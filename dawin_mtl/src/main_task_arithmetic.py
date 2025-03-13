import os
import numpy as np
import time
import sys

from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments
import wandb

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

model = 'ViT-B-32'

if args.train_dataset is None:
    args.train_dataset = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
if args.eval_datasets is None:
    args.eval_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
train_datasets = args.train_dataset
exam_datasets = args.eval_datasets

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
log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))

task_vectors = [
    TaskVector(pretrained_checkpoint, '../checkpoints/'+model+'/'+dataset_name+'/finetuned.pt') for dataset_name in train_datasets
]


startt = time.time()
task_vector_sum = sum(task_vectors)
scaling_coef_ = 0.3
image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef_)
log.info('*'*20 + 'scaling_coef:' + str(scaling_coef_) + '*'*20)

print(f'Use {args.device}')
accs = []
metric_dict = {}
for dataset in exam_datasets:
    metrics = eval_single_dataset(image_encoder, dataset, args)
    log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
    accs.append(metrics.get('top1')*100)
    metric_dict[f"{str(dataset)}_Acc"] = metrics.get('top1')*100
log.info('Avg ACC:' + str(np.mean(accs)) + '%')
metric_dict[f"Avg_Acc"] = np.mean(accs)
runtime = time.time() - startt
if args.wb_project:
    wandb.log(metric_dict)