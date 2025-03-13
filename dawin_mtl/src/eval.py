import os
import json
import tqdm

import torch
import numpy as np

import utils
from datasets.common import get_dataloader, maybe_dictionarize
from heads import get_classification_head
from modeling import ImageClassifier

from datasets.registry import get_dataset
from scipy.special import softmax
import pdb

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)
        
def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

#! DaWin
def eval_single_dataset_dynamic(interpolator, dataset_name, args, dmms=None):
    interpolator.eval()
    coefs_label = interpolator.dmm_coef_labels[f'{dataset_name}']

    dataset = get_dataset(
        dataset_name,
        interpolator.model.model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    
    try:
        n_tasks = len(args.train_dataset.split(','))
    except:
        n_tasks = len(args.train_dataset)
    
    #pdb.set_trace()
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        
        n_eval_batches = len(dataloader)
        tot_logits, tot_labels = torch.tensor([]).cuda(), torch.tensor([]).cuda()

        #! domain-wise merging (our default setup)
        #* 0.3 (task arithmetic default scaler) * N_task
        try:
            alph_ = torch.cat([torch.tensor([1.0]),torch.tensor([0.3 * n_tasks * dmms[dataset_name].params_[0]]).squeeze()])
        except:
            alph_ = torch.cat([torch.tensor([1.0]),torch.tensor([0.3 * n_tasks * dmms[dataset_name].params_[0]]).reshape(1)])
                
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph_))) for j, p in enumerate(zip(*interpolator.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(interpolator.model, interpolator.names, params)
        
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            if args.ncluster == 0:
                logits = interpolator(x, None, dataset_name)
            else:
                #! batch-wise merging
                # if i < (n_eval_batches - 1): batch_idx = coefs_label[args.batch_size*i:args.batch_size*(i+1)]
                # else:                        batch_idx = coefs_label[args.batch_size*i:]
                # y_recon, features = torch.tensor([]).cuda(), torch.tensor([]).cuda()
                # for j in np.unique(batch_idx):
                #     sub_idcs = batch_idx == j
                #     if sum(sub_idcs).item() > 0:
                #         if args.clustering == 'dmm':
                #             try:
                #                 alph_ = torch.cat([torch.tensor([1.0]),torch.tensor([0.3 * n_tasks * dmms[dataset_name].params_[j]]).squeeze()])
                #             except:
                #                 alph_ = torch.cat([torch.tensor([1.0]),torch.tensor([0.3 * n_tasks * dmms[dataset_name].params_[j]]).reshape(1)])
                #         else: # kMeans
                #             alph_ = torch.cat([torch.tensor([1.0]),torch.tensor([0.3 * n_tasks * softmax(dmms[dataset_name].cluster_centers_[j])]).squeeze()])

                #         params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph_))) for j, p in enumerate(zip(*interpolator.paramslist)))
                #         params = tuple(p.cuda(0) for p in params)
                #         load_weights(interpolator.model, interpolator.names, params)
                #         with torch.no_grad():
                #             feature_sub = interpolator.model(x[sub_idcs])
                #         y_recon = torch.cat([y_recon, y[sub_idcs]])
                #         features = torch.cat([features, feature_sub])

                with torch.no_grad():
                    features = interpolator.model(x)

            layer_name = 'classifier_{}'.format(dataset_name)
            classification_head = getattr(interpolator, layer_name)
            logits = classification_head(features)

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            
            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
    return metrics
    
def eval_single_dataset(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    print(f"{dataset_name} tot test samples: {len(dataloader) * args.batch_size}")

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            
            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
    return metrics

def eval_single_dataset_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)

    model.eval()

    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    print(f"{dataset_name} tot test samples: {len(dataloader) * args.batch_size}")

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')

    return metrics

def eval_single_dataset_preprocess_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)

    model.eval()

    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')

    return metrics

def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info