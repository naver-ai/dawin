import os
import time
import sys
import tqdm

from eval import eval_single_dataset, eval_single_dataset_head, eval_single_dataset_preprocess_head
from args import parse_arguments
from merging_cofficient import get_merging_cofficients
import wandb
import copy
import pdb
import torch.nn.functional as F

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

n_tasks = len(exam_datasets)
model = 'ViT-B-32'
if args.set_alphas != 'from_scratch':
    args.eval_only = 1

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
log = create_log_dir(args.logs_path, 'log_{}_Task_wise_AdaMergingPP.txt'.format(str_time_))
args.log = log

from ties_merging_utils import *

ft_checks = [torch.load('../checkpoints/'+model+'/'+dataset_name+'/finetuned.pt').state_dict() for dataset_name in exam_datasets]
ptm_check = torch.load(pretrained_checkpoint).state_dict()

check_parameterNamesMatch(ft_checks + [ptm_check])

remove_keys = []

print(f"Flattening out Checkpoints")
flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

tv_flat_checks = flat_ft - flat_ptm

assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check)
assert all([check_state_dicts_equal(vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i])for i in range(len(ft_checks))])

K = 20
merge_func = "dis-sum"

selected_entries, merged_tv = ties_merging_split(tv_flat_checks, reset_thresh=K, merge_func=merge_func,)

ties_task_vectors = []
for vector_ in selected_entries:
    t_state_dict = vector_to_state_dict(vector_, ptm_check, remove_keys=remove_keys)
    ref_model = torch.load(pretrained_checkpoint)
    ref_model.load_state_dict(t_state_dict, strict=False)
    ties_task_vectors.append(ref_model.state_dict())

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
    # Remove all the parameters in the model
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
class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets, set_alphas='from_scratch'):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = copy.deepcopy(model)

        self.names = names
        self.pretrain_lambdas = torch.ones(1, 1)
        prior = 0.3
        rlambdas = torch.ones(1, len(paramslist)-1) * prior  # (1 * tasks)
        self.lambdas_raw = torch.nn.Parameter(rlambdas)
        self.set_alphas = set_alphas

        self.classifier = []
        for dataset_name in exam_datasets:
            classification_head = get_classification_head(args, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(args.device))
            self.classifier.append(layer_name)

    def set_pt_lambdas(self):
        ralpha = get_merging_cofficients('tw_adamergingpp', args.model)
        self.alpha = torch.Tensor(ralpha)
        return self.alpha

    def lambdas(self, inp=None, y=None, dataset_name=None):
        if self.set_alphas == 'from_scratch':
            task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
            lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        elif self.set_alphas == 'pre-trained':
            lambdass = self.set_pt_lambdas()
        else:
            lambdass = None
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

    def forward(self, inp, dataset_name):
        
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)

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
paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.items())  for i, tv in enumerate(ties_task_vectors)] # task vectors
torch.cuda.empty_cache()
adamerging_mtl_model = AdaMerging(paramslist, model, names, exam_datasets, args.set_alphas)

print('init lambda:')
print(adamerging_mtl_model.lambdas())
print('collect_trainable_params:')
print(list(adamerging_mtl_model.collect_trainable_params()))

epochs = 500
optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)

from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle

metric_dict = {}
Total_ACC = 0.
evalds = exam_datasets if args.eval_datasets is None else args.eval_datasets
if args.eval_only:
    for dataset_name in evalds:
        image_encoder = adamerging_mtl_model.get_image_encoder()
        classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
        metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
        Total_ACC += metrics['top1'] * 100
        log.info('Eval: init: ' + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
        metric_dict[f"{str(dataset_name)}_Acc"] = metrics['top1']*100
        log.info('Eval: init: ' + ' Avg ACC:' + str(Total_ACC / len(exam_datasets)) + '\n')    
        metric_dict[f"Avg_Acc"] = Total_ACC / len(exam_datasets)
        if args.wb_project:
            wandb.log(metric_dict)
else:
    for epoch in range(epochs):
        losses = 0.
        for dataset_name in exam_datasets:
            dataset = get_dataset(dataset_name, pretrained_model.val_preprocess, location=args.data_location, batch_size=16)
            dataloader = get_dataloader_shuffle(dataset)

            for i, data in enumerate(tqdm.tqdm(dataloader)):
                data = maybe_dictionarize(data)
                x = data['images'].to(args.device)
                y = data['labels'].to(args.device)

                outputs = adamerging_mtl_model(x, dataset_name)
                loss = softmax_entropy(outputs).mean(0)
                losses += loss

                if i > 0: # Execute only one step
                    break

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(list(adamerging_mtl_model.lambdas().data))

        if ((epoch+1) % 500) == 0:
            log.info(str(list(adamerging_mtl_model.lambdas().data)))

            Total_ACC = 0.
            for dataset_name in exam_datasets:
                image_encoder = adamerging_mtl_model.get_image_encoder()
                classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
                metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
                Total_ACC += metrics['top1']
                log.info('Eval: Epoch: ' + str(epoch) + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
            log.info('Eval: Epoch: ' + str(epoch) + ' Avg ACC:' + str(Total_ACC / len(exam_datasets)) + '\n')
