import argparse
import os
import torch
import clip
import os
from datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA
from utils import get_model_from_sd, test_model_on_dataset
import wandb

def interpolation(alpha, weight1, weight2):
    return {key: (1 - alpha) * weight1[key] + alpha * weight2[key] for key in weight1.keys()}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('checkpoints/soups'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--eval-single-model", type=str, default='',
    )
    parser.add_argument(
        "--wiseft-alpha", type=float, default=-0.1,
    )
    parser.add_argument(
        "--wiseft-ftpath", type=str, default='',
    )
    parser.add_argument(
        "--wiseft-zspath", type=str, default='',
    )
    parser.add_argument(
        "--model", type=str, default='ViT-B/32',
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--wb_project", type=str, default='', # weight and bias
    )
    parser.add_argument(
        "--wb_runname", type=str, default='', # weight and bias
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    if args.wb_project:
        wandb_args = {"project": args.wb_project}
        wandb_args["name"] = args.wb_runname if args.wb_runname else None
        wandb.init(**wandb_args, config=vars(args), save_code=False)

    base_model, preprocess = clip.load(args.model, 'cpu', jit=False)

    if args.eval_single_model:
        model_path = [os.path.join(args.model_location, f'{args.eval_single_model}')]
            
        assert os.path.exists(model_path)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model = get_model_from_sd(state_dict, base_model)

        results = {'model_name' : f'{args.eval_single_model}'}
        for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:
            print(f'Evaluating model-{args.eval_single_model} on {dataset_cls.__name__}.')
            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)
        
        results['OODavg'] = 1./5 * (results['ImageNetV2'] + 
            results['ImageNetR'] + results['ImageNetSketch'] + 
            results['ObjectNet'] + results['ImageNetA'])

        if args.wb_project:
            wandb.log({k+'_Acc': v for k, v in results.items()})

    
    if args.wiseft_alpha >= 0.0:
        alpha = args.wiseft_alpha
        ft_state_dict = torch.load(args.wiseft_ftpath, map_location=torch.device('cuda'))
        zs_state_dict = torch.load(args.wiseft_zspath, map_location=torch.device('cuda'))
        
        merged = interpolation(alpha, zs_state_dict, ft_state_dict)
        model = get_model_from_sd(merged, base_model)

        results = {'model_name' : f'wiseft_{alpha:.2f}'}
        for i, dataset_cls in enumerate([ImageNet, ImageNetV2, ImageNetR, ImageNetA, ImageNetSketch, ObjectNet]):
            print(f'Evaluating on {dataset_cls.__name__} with coef {alpha}.')
            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)

        results['OODavg'] = 1./5 * (results['ImageNetV2'] + 
            results['ImageNetR'] + results['ImageNetSketch'] + 
            results['ObjectNet'] + results['ImageNetA'])

        if args.wb_project:
            wandb.log({k+'_Acc': v for k, v in results.items()})