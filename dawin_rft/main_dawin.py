import torch
import torch.nn.functional as F
import wandb
import os, time, copy, pdb, random
import argparse
import numpy as np
from tqdm import tqdm
import datasets
from datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA
import clip
from clip.model import convert_weights

from utils import get_model_from_sd
from utils import get_feats_logits_and_labels, compute_accuracy, compute_sample_loss_uncertainty_acc
#from mixturemodel import BetaMixtureModel
from mixturemodel_revised import BetaMixtureModel

def interpolation(alpha, weight1, weight2):
    return {key: (1 - alpha) * weight1[key] + alpha * weight2[key] for key in weight1.keys()}

class DaWin(object):
    def __init__(self, basemodel, preprocess, args, zssd, ftsd):
        self.basemodel = copy.deepcopy(basemodel)
        self.preprocess = preprocess
        self.args = args
        self.zs_sd = zssd
        self.ft_sd = ftsd
        if args.wise_alpha:
            self.merged_sd = interpolation(args.wise_alpha, zssd, ftsd)
        else:
            self.merged_sd = interpolation(0.5, zssd, ftsd) # by default
        self.expertise = args.expertise
        self.temperature_scales = {
            'ViT-B/32':[0.9576, 0.9733],
            'ViT-B/16':[0.96, 1.0066],
            'ViT-L/14':[1.0767, 0.9545],
         }[args.model]
        

    def build_datastore(self,pseudo_label=0,dsname=None,dataloader=None,project_fn=None):
        # Load feature / logits 
        # First run requires extracting features and logits for entire samples from the evaluation dataset.
        self.args.prefix = 'zs'
        print(f"load {self.args.prefix} cache")
        _, zs_logits, labels, project_fn = get_feats_logits_and_labels(self.basemodel, self.preprocess, self.args, dataset_name=dsname, state_dict=self.zs_sd)
        self.args.prefix = 'ft'
        print(f"load {self.args.prefix} cache")
        _, ft_logits, _, _ = get_feats_logits_and_labels(self.basemodel, self.preprocess, self.args, dataset_name=dsname, state_dict=self.ft_sd)

        labels_ = copy.deepcopy(labels)        
        if pseudo_label == 1: # soft label from output ensemble
            zs_prob = F.softmax(project_fn(zs_logits,device=self.args.device)/self.temperature_scales[0], dim=1)
            ft_prob = F.softmax(project_fn(ft_logits,device=self.args.device)/self.temperature_scales[1], dim=1)
            labels = (zs_prob + ft_prob)/2
        elif pseudo_label == 2: # soft label from merged model
            _, m_logits, _, _ = get_feats_logits_and_labels(self.basemodel, self.preprocess, self.args, dataset_name=dsname, state_dict=self.merged_sd)
            labels = F.softmax(project_fn(m_logits,device=self.args.device),dim=1)
        elif pseudo_label == 3: # hard label from output ensemble
            zs_prob = F.softmax(project_fn(zs_logits,device=self.args.device)/self.temperature_scales[0], dim=1)
            ft_prob = F.softmax(project_fn(ft_logits,device=self.args.device)/self.temperature_scales[1], dim=1)
            labels = torch.argmax(zs_prob + ft_prob, dim=1)
        elif pseudo_label == 4: # hard label from merged model
            _, m_logits, _, _ = get_feats_logits_and_labels(self.basemodel, self.preprocess, self.args, dataset_name=dsname, state_dict=self.merged_sd)
            labels = torch.argmax(project_fn(m_logits,device=self.args.device),dim=1)
        else: # use entropy or X-entropy
            pass

        zs_loss, zs_accs, zs_ent = compute_sample_loss_uncertainty_acc(project_fn(zs_logits,device=self.args.device), labels, self.temperature_scales[0],pseudo_label=pseudo_label)
        ft_loss, ft_accs, ft_ent = compute_sample_loss_uncertainty_acc(project_fn(ft_logits,device=self.args.device), labels, self.temperature_scales[1],pseudo_label=pseudo_label)
        
        # domain-wise offset adjustment
        if self.args.offset_adjustment:
            if 'ent' in self.expertise:
                print('good - oa')
                self.coef_bias = (zs_ent.std()/zs_ent.mean() + ft_ent.std()/ft_ent.mean()) / 2
                #self.coef_bias = (zs_ent.std()/zs_ent.mean() + zs_ent.std()/zs_ent.mean()) / 2
                self.coef_biasw = (zs_ent.mean() + ft_ent.mean()) / zs_ent.mean()
            elif 'loss' in self.expertise:
                self.coef_bias = (zs_loss.std()/zs_loss.mean() + ft_loss.std()/ft_loss.mean()) / 2
                self.coef_biasw = (zs_loss.mean() + ft_loss.mean()) / zs_loss.mean()
            else:
                pass
        else:
            self.coef_bias = 0.0
            self.coef_biasw = 2.0
        
        if self.expertise == 'negexp_loss_ratio':
            negexp_zs_loss = (-zs_loss).exp()
            negexp_ft_loss = (-ft_loss).exp()
            coef = (negexp_ft_loss + self.coef_bias/self.coef_biasw) / (negexp_zs_loss + negexp_ft_loss + self.coef_bias)
        elif self.expertise == 'negexp_ent_ratio':
            print('good - expertise')
            negexp_zs_ent = (-zs_ent).exp()
            negexp_ft_ent = (-ft_ent).exp()
            coef = (negexp_ft_ent + self.coef_bias/self.coef_biasw) / (negexp_zs_ent + negexp_ft_ent + self.coef_bias)
        elif self.expertise == 'maxlogit_ratio':
            zs_max_logits = zs_logits.max(dim=1)[0].reshape(-1,1)
            ft_max_logits = ft_logits.max(dim=1)[0].reshape(-1,1)
            coef = (ft_max_logits + self.coef_bias/self.coef_biasw) / (zs_max_logits + ft_max_logits + self.coef_bias)
        elif self.expertise == 'expconf_ratio':
            zs_max_conf = F.softmax(zs_logits/self.temperature_scales[0], dim=1).max(dim=1)[0].exp().reshape(-1,1)
            ft_max_conf = F.softmax(zs_logits/self.temperature_scales[1], dim=1).max(dim=1)[0].exp().reshape(-1,1)
            coef = (ft_max_conf + self.coef_bias/self.coef_biasw) / (zs_max_conf + ft_max_conf + self.coef_bias)
        elif self.expertise == 'expconfdiff_ratio':
            zs_prob = F.softmax(zs_logits/self.temperature_scales[0], dim=1)
            sorted_zs, _ = torch.sort(zs_prob, descending=True)
            zs_conf_diff = (sorted_zs[:,0] - sorted_zs[:,1]).exp().reshape(-1,1)
            ft_prob = F.softmax(zs_logits/self.temperature_scales[1], dim=1)
            sorted_ft, _ = torch.sort(ft_prob, descending=True)
            ft_conf_diff = (sorted_ft[:,0] - sorted_ft[:,1]).exp().reshape(-1,1)
            coef = (ft_conf_diff + self.coef_bias/self.coef_biasw) / (zs_conf_diff + ft_conf_diff + self.coef_bias)
        elif self.expertise == 'selective_entropy':
            coef = (zs_ent > ft_ent).int().reshape(-1,1)
        elif self.expertise == 'selective_loss':
            coef = (zs_loss > ft_loss).int().reshape(-1,1)
        
        
        if self.args.ensemble_coef:
            if self.args.ensemble_coef.split(';')[0] == 'constant':
                print("conducting output ensemble with constant coef")
                coef = self.args.ensemble_coef.split(';')[1]
            else:
                print("conducting output ensemble with dynamic coef")
                coef = coef.reshape(-1,1)
            zs_prob = F.softmax(project_fn(zs_logits,device=self.args.device)/self.temperature_scales[0], dim=1)
            ft_prob = F.softmax(project_fn(ft_logits,device=self.args.device)/self.temperature_scales[1], dim=1)
            acc = compute_accuracy((1 - coef) * zs_prob + coef * ft_prob, labels_)
            return acc
        elif 'selective' in self.expertise:
            acc = (((1 - coef) * zs_accs + coef * ft_accs).sum() / len(coef)).item()
            return acc
        else:
            pass

        # you can save the estimated coefficients for analysis
        if self.args.save_outputs:
            results = coef.cpu().numpy()
            with open(f'analysis/{dsname}_{self.args.wb_runname}_coef.npy','wb') as f:
                np.save(f, results)
        self.datastore = {}
        self.datastore['coefficient'] = coef


    def eval(self, w1, w2, dataset, loader, dataset_name=None):
        print("get coefficient")
        coefs = self.datastore['coefficient']
        coefs = coefs * self.args.wise_alpha if self.args.wise_alpha else coefs

        if self.args.baseline == 'uniform':
            coefs = torch.rand_like(self.datastore['coefficient'])
        elif self.args.baseline == 'gaussian':
            coefs = torch.normal(mean=0.5, std=0.2, size=self.datastore['coefficient'].shape)
            coefs = torch.clamp(coefs, min=0.01, max=0.99)
        elif self.args.baseline == 'wise_gaussian':
            coefs = torch.normal(mean=0.8, std=0.1, size=self.datastore['coefficient'].shape)
            coefs = torch.clamp(coefs, min=0.01, max=0.99)
        else:
            pass

        if (len(coefs.shape) > 1) and (coefs.size(1) > 1):
            coefs = coefs.mean(axis=1)
        coefs = coefs.squeeze()
        
        coefs_label, bmm, is_correct = None, None, None
        if self.args.bmm_ncluster:
            print('good - bmm')
            coefs_np = coefs.cpu().numpy().reshape(-1,1)
            bmm = BetaMixtureModel(n_mixtures=self.args.bmm_ncluster, random_seed=self.args.seed)
            bmm.fit(coefs_np)
            for i in range(bmm.n_mixtures):
                a,b = bmm.beta_params_[i, 0],bmm.beta_params_[i, 1]
                print(f'beta means of {i}th cluster: {a/(a+b):.3f}')
            coefs_label = bmm.predict(coefs_np)

        print("start evaluation")
        top1, correct, n = 0., 0., 0.
        tot_logits, tot_labels = torch.tensor([]).cuda(), torch.tensor([]).cuda()
        n_eval_batches = len(loader)
        bmm_means = np.array([])
        for i, batch in enumerate(tqdm(loader)):
            inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
            if 'image_paths' in batch:
                image_paths = batch['image_paths']

            #* Cluster-wise Dynamic Merging
            if (self.args.bmm_ncluster > 0) and (self.args.eval_batch_size > 1):
                print('good - eval')
                if i < (n_eval_batches - 1): batch_idx = coefs_label[self.args.eval_batch_size*i:self.args.eval_batch_size*(i+1)]
                else:                        batch_idx = coefs_label[self.args.eval_batch_size*i:]

                y_recon, logits = torch.tensor([]).cuda(), torch.tensor([]).cuda()
                for j in np.unique(batch_idx):
                    sub_idcs = batch_idx == j
                    if sum(sub_idcs).item() > 0:
                        merged_sd = interpolation(bmm.beta_params_[j, 0] / (bmm.beta_params_[j, 0] + bmm.beta_params_[j, 1]), w1, w2)
                        model = get_model_from_sd(merged_sd, self.basemodel)
                        with torch.no_grad():
                            logits_sub = model(inputs[sub_idcs])
                        y_recon = torch.cat([y_recon, labels[sub_idcs]])
                        logits = torch.cat([logits, logits_sub])
                labels = y_recon
            #* Fully Sample-wise Dynamic Merging (very slow!!!!!)
            else:
                merged_sd = interpolation(coefs[i], w1, w2)
                model = get_model_from_sd(merged_sd, self.basemodel)
                if args.half_prec:
                    convert_weights(model)
                    inputs = inputs.half()
                with torch.no_grad():
                    logits = model(inputs)
                
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                labels = dataset.project_labels(labels, device)
            if isinstance(logits, list):
                logits = logits[0]
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, labels, image_paths, None)
                correct += acc1
                n += num_total
            else:
                correct_current = pred.eq(labels.view_as(pred)).sum().item()
                correct += correct_current
                n += labels.size(0)

            tot_labels = torch.cat([tot_labels, labels])
            tot_logits = torch.cat([tot_logits, logits])
            tot_labels = tot_labels.type(torch.cuda.LongTensor)
        
        # save evaluation outputs to analyze correlation between entropy and cross entropy, and more.
        # per-dataset, you can get an array of (entropy, x-entropy, indicator for correct/incorrect)
        if self.args.save_outputs:
            loss, is_correct, ent = compute_sample_loss_uncertainty_acc(tot_logits, tot_labels)
            results = torch.cat([ent.reshape(-1,1), loss.reshape(-1,1), is_correct.int().reshape(-1,1)],axis=1).cpu().numpy()
            with open(f'analysis/{dataset_name}_{self.args.wb_runname}_eval.npy','wb') as f:
                np.save(f, results)
        return correct / n
    
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtype",type=str,default='') # whether temperature scaling mode or not
    parser.add_argument("--data-location",type=str,default='data/')
    parser.add_argument("--dataset_name",type=str,default='ImageNet')
    parser.add_argument("--source_dataset",type=str,default='ImageNet') # ImageNet2p or ImageNet98p
    parser.add_argument("--eval_dataset",type=str,default='') # specifying evaluation set -- default: evaluation on all the imagenet variants
    parser.add_argument("--model-location",type=str,default='checkpoints/',)
    parser.add_argument("--cache-dir",type=str,default='cache/',)
    
    parser.add_argument("--expertise", type=str, default='negexp_ent_ratio',) # acc, acc1, lsvr_f, ssvr_cv, sve_f
    parser.add_argument("--wise-alpha", type=float, default=0.0,)
    
    parser.add_argument('--offset-adjustment', action='store_true') # offset adjustment flag
    parser.add_argument("--baseline", type=str, default='',)

    parser.add_argument("--ensemble_coef", type=str, default='',)
    parser.add_argument("--half_prec", type=int, default=0,) # not implemented yet
    parser.add_argument("--seed", type=int, default=1, )
    parser.add_argument("--pseudo_label", type=int, default=0, )
    parser.add_argument("--bmm_ncluster", type=int, default=0, )
    
    parser.add_argument("--save-outputs", type=int, default=0, )
    parser.add_argument("--wb_project", type=str, default='',)
    parser.add_argument("--wb_runname", type=str, default='',)
    parser.add_argument("--zs-path", type=str, default='checkpoints/zeroshot.pt', )
    parser.add_argument("--ft-path", type=str, default='checkpoints/finetune.pt', )
    parser.add_argument("--model", type=str, default='ViT-B/32',)
    parser.add_argument("--eval-batch-size", type=int, default=1024,)
    parser.add_argument("--workers",type=int,default=8,)

    args = parser.parse_args()

    def set_seed(SEED):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        random.seed(SEED)

    set_seed(args.seed)
    if args.wb_project:
        wandb_args = {"project": args.wb_project}
        wandb_args["name"] = args.wb_runname if args.wb_runname else None
        wandb.init(**wandb_args, config=vars(args), save_code=False)
    args.batch_size = args.eval_batch_size
    device, args.device = 'cuda', 'cuda'

    #* 1. load the backbone model & dataset
    base_model, preprocess = clip.load(args.model, 'cpu', jit=False)
    image_sz = base_model.visual.input_resolution 
    base_model.eval()

    #* 2. load sd, and init dawin
    ft_state_dict = torch.load(args.ft_path, map_location=torch.device(args.device))
    zs_state_dict = torch.load(args.zs_path, map_location=torch.device(args.device))
    
    print("init training-free data-driven merger")
    merger = DaWin(base_model, preprocess, args, zs_state_dict, ft_state_dict)

    #* 3. evaluation
    results={}
    eval_datasets = [ImageNet, ImageNetV2, ImageNetR, ImageNetA, ImageNetSketch, ObjectNet]
    # user-define eval datasets
    if args.eval_dataset:
        eval_datasets = []
        for eds in args.eval_dataset.split(';'):
            if   eds == 'ImageNetSketch': eval_datasets.append(ImageNetSketch)
            elif eds == 'ImageNetR': eval_datasets.append(ImageNetR)
            elif eds == 'ImageNetA': eval_datasets.append(ImageNetA)
            elif eds == 'ImageNet': eval_datasets.append(ImageNet)
            elif eds == 'ImageNetV2': eval_datasets.append(ImageNetV2)
            elif eds == 'ImageNet2p': eval_datasets.append(ImageNet2p)
            elif eds == 'ObjectNet': eval_datasets.append(ObjectNet)
            else: pass

    for i, dataset_cls in enumerate(eval_datasets):
        print(f'Evaluating on {dataset_cls.__name__}.')
        dataset = dataset_cls(preprocess, args.data_location, args.eval_batch_size, args.workers)
        project_fn = getattr(dataset, 'project_logits', None)

        loader = dataset.test_loader
        if (args.ensemble_coef) or ('selective' in args.expertise):
            top1 = merger.build_datastore(args.pseudo_label, dsname=dataset_cls.__name__)
        else:
            merger.build_datastore(pseudo_label=args.pseudo_label, dsname=dataset_cls.__name__, dataloader=loader, project_fn=project_fn)
            top1 = merger.eval(zs_state_dict, ft_state_dict, dataset, loader=loader, dataset_name=dataset_cls.__name__)
            
        print(top1)
        results[dataset_cls.__name__] = top1
        if args.wb_project:
            print(f'{dataset_cls.__name__}_Acc: {top1:.4f}')
            wandb.log({f'{dataset_cls.__name__}_Acc':top1})

    try:
        results['OODavg'] = 1./5 * (results['ImageNetV2'] + results['ImageNetR'] + results['ImageNetSketch'] + results['ObjectNet'] + results['ImageNetA'])
        print(f'OODavg_Acc: {results["OODavg"]:.4f}')
        if args.wb_project:
            wandb.log({f'OODavg_Acc':results["OODavg"]})
    except:
        pass