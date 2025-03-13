import os
import time
import sys
import tqdm

import torch
from task_vectors import TaskVector
from eval import eval_single_dataset_dynamic
from args import parse_arguments
import wandb
import copy
import torch.nn.functional as F
from utils import DirichletMixtureModel
from sklearn.cluster import KMeans
import numpy as np
import pdb

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

args = parse_arguments()

if args.train_dataset is None:
    args.train_dataset = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
if args.eval_datasets is None:
    args.eval_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
train_datasets = args.train_dataset
exam_datasets = args.eval_datasets

model = 'ViT-B-32'

if args.wb_project:
    wandb_args = {"project": args.wb_project}
    wandb_args["name"] = args.wb_runname if args.wb_runname else None
    wandb.init(**wandb_args, config=vars(args), save_code=False)

args.data_location = '../data'
args.model = model
args.save = '../checkpoints/' + model
args.logs_path = '../logs/' + model
pretrained_checkpoint = '../checkpoints/'+model+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_Task_wise_AdaMerging.txt'.format(str_time_))
args.log = log

task_vectors = [TaskVector(pretrained_checkpoint, '../checkpoints/'+model+'/'+dataset_name+'/finetuned.pt') for dataset_name in train_datasets]
n_tasks = len(task_vectors)

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

def load_weights_explicit(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)
    return mod

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model

        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model(images)
        return features

from heads import get_classification_head
class DaWin(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets):
        super(DaWin, self).__init__()
        self.paramslist = paramslist
        self.model = copy.deepcopy(model)
        self.individuals = {i:copy.deepcopy(model) for i in range(len(paramslist)-1)}

        self.names = names
        self.pretrain_lambdas = torch.ones(1, 1)
        prior = 0.3
        rlambdas = torch.ones(1, len(paramslist)-1) * prior 
        self.lambdas_raw = torch.nn.Parameter(rlambdas)
        
        self.classifier = []
        for dataset_name in exam_datasets:
            classification_head = get_classification_head(args, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(args.device))
            self.classifier.append(layer_name)
        
        for t in range(n_tasks):
            temp_plist = [self.paramslist[0]] + [self.paramslist[t+1]]
            params = tuple(sum(p) for _, p in enumerate(zip(*temp_plist)))
            params = tuple(p.cuda(0) for p in params)
            model = load_weights_explicit(self.individuals[t], self.names, params)
            self.individuals[t] = model
        
        self.dmm_coef_labels = None

    def lambdas(self, inp=None, y=None, dataset_name=None, prob=False):        
        if inp is None:
            return torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        negent = torch.tensor([]).to(inp.device)
        for t in range(n_tasks):
            with torch.no_grad():
                feats = self.individuals[t](inp)
                layer_name = 'classifier_{}'.format(dataset_name)
                classification_head = getattr(self, layer_name)
                out = classification_head(feats)
            if y is None:
                negent = torch.cat([negent, -softmax_entropy(out).reshape(-1,1)],axis=1)
            else:
                negent = torch.cat([negent, -F.cross_entropy(out,y,reduction='none').reshape(-1,1)],axis=1)
        if prob: # for Dirichlet mixture
            lambdass = F.softmax(negent, dim=1)
        else:
            lambdass = torch.cat([torch.ones(out.size(0),1).to('cuda:0'), (0.3 * n_tasks) * F.softmax(negent, dim=1)],axis=1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_image_encoder(self):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp, y, dataset_name,direct=False):
        alph = self.lambdas(inp, y, dataset_name)
        batch_feature = torch.tensor([]).to(inp.device)
        for i in range(inp.shape[0]):
            alph_ = alph[i]
            params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph_.cpu()))) for j, p in enumerate(zip(*self.paramslist)))
            params = tuple(p.cuda(0) for p in params)
            load_weights(self.model, self.names, params)
            with torch.no_grad():
                feature = self.model(inp)
            batch_feature = torch.cat([batch_feature, feature])
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(batch_feature)

        return out

pretrained_model = torch.load(pretrained_checkpoint)
pretrained_model_dic = pretrained_model.state_dict()

model = ModelWrapper(pretrained_model, exam_datasets)
model = model.to(args.device)
_, names = make_functional(model)

paramslist = []
paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())] # pretrain
paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items())  for i, tv in enumerate(task_vectors)] # task vectors
torch.cuda.empty_cache()
dawin_mtl_model = DaWin(paramslist, model, names, exam_datasets)

from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle

metric_dict = {}
Total_ACC = 0.
evalds = exam_datasets

# coefficient computation
device = args.device
print(f'compute entire coefficient and perform DMM modeling')

entire_coef_dict = {}
dmm_coef_labels = {}
dmms = {}
for dataset_name in evalds:
    print(f"on {dataset_name}")
    fn = f'../cache/CLIP_VITB16_{dataset_name}_coefficients.pt'
    if os.path.exists(fn):
        print(f'load cache: {fn}')
        coef = torch.load(fn)
    else:
        print(f'cache is not founded from {fn}, extract from scratch ...')
        dataset = get_dataset(
                    dataset_name,
                    dawin_mtl_model.model.model.val_preprocess,
                    location=args.data_location,
                    batch_size=args.batch_size
                )
        dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
        coef = torch.tensor([]).cuda()
        startt0 = time.time()
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                data = maybe_dictionarize(data)
                inp = data['images'].to(device)
                y = data['labels'].to(device)
                coef_ = dawin_mtl_model.lambdas(inp, None, dataset_name, prob=True)
                coef = torch.cat([coef, coef_])
        coef = coef.cpu()
        torch.save(coef, fn)

    
    # membership inference from Dirichlet Mixture Model
    print('membership inference of DMM')
    coefs_np = coef.numpy()
    if args.save_coef:
        os.makedirs('assets',exist_ok=True)
        np.save(f'assets/{args.wb_runname}_coef.npy', coefs_np)
    
    entire_coef_dict[f'{dataset_name}'] = coefs_np

    if args.clustering == 'dmm':
        dmm = DirichletMixtureModel(n_components=1, random_seed=0)
        dmm.fit(coefs_np, n_init=5, method="kmeans", max_iter=500, tol=1e-5)
        dmm_coef_labels[f'{dataset_name}'] = dmm.predict(coefs_np)
        print(f'estimated DMM params: {dmm.params_}')
        dawin_mtl_model.dmm_coef_labels = dmm_coef_labels
    elif args.clustering == 'kmeans':
        dmm = KMeans(n_clusters=1,random_state=0,n_init='auto').fit(coefs_np)
        dmm_coef_labels[f'{dataset_name}'] = dmm.labels_
        print(f'estimated params: {dmm.cluster_centers_}')
    else:
        raise ValueError('not implemented yet')
    dmms[f'{dataset_name}'] = dmm

dawin_mtl_model.dmm_coef_labels = dmm_coef_labels


for dataset_name in exam_datasets:
    startt = time.time()
    metrics = eval_single_dataset_dynamic(dawin_mtl_model,dataset_name,args,dmms)
    
    Total_ACC += metrics['top1']*100
    log.info('Eval: init: ' + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
    metric_dict[f"{str(dataset_name)}_Acc"] = metrics['top1']*100

    log.info('Eval: init: ' + ' Avg ACC:' + str(Total_ACC / len(exam_datasets)) + '\n')    
    metric_dict[f"Avg_Acc"] = Total_ACC / len(exam_datasets)
    if args.wb_project:
        wandb.log(metric_dict)