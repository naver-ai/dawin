import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import os
import pickle
from tqdm import tqdm
import clip
import copy
import copy
import datasets
from clip.model import convert_weights

    
def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifier.cpu(), f)


def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None, lp_flag=False):
        super(ModelWrapper, self).__init__()
        self.model = model # model: CLIP class
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        if initial_weights is None:
            initial_weights = torch.zeros_like(self.classification_head.weight)
            torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
        self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
        self.classification_head.bias = torch.nn.Parameter(
            torch.zeros_like(self.classification_head.bias))
        if lp_flag:
            for name, param in self.model.named_parameters():
                param.requires_grad_(False)

        #! discriminator for CLIP and other backbone
        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, return_features=False):
        features = self.model.encode_image(images)

        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        if return_features:
            return logits, features
        return logits
    
    def save(self, filename):
        print(f'Saving classifier to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classifier from {filename}')
        return torch_load(filename)

def get_model_from_sd(state_dict, base_model):
    feature_dim = state_dict['classification_head.weight'].shape[1]
    num_classes = state_dict['classification_head.weight'].shape[0]
    
    model = ModelWrapper(copy.deepcopy(base_model), feature_dim, num_classes, normalize=True)
    for p in model.parameters():
        p.data = p.data.float()
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    #devices = [x for x in range(torch.cuda.device_count())]
    return model #torch.nn.DataParallel(model,  device_ids=devices)

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None, shape=[512, 1000]):
        if weights is not None:
            output_size, input_size = weights.shape
            super().__init__(input_size, output_size)
        else:
            super().__init__(shape[0], shape[1])
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())

        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename, logger=None):
        print(f'Loading classification head from {filename}')
        if logger != None:
            logger.info(f'Loading classification head from {filename}')
        return torch_load(filename)

class ImageClassifier(torch.nn.Module):
    def __init__(self,
                 image_encoder,
                 classification_head,
                 process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs, out_feat=False):
        if self.process_images:
            feats = self.image_encoder(inputs)
        outputs = self.classification_head(feats)
        if out_feat:
            return outputs, feats
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return torch_load(filename)


def maybe_dictionarize_batch(batch):
    if isinstance(batch, dict):
        return batch
    if len(batch) == 2:
        return {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        return {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')


def test_model_on_dataset(model, dataset,half_prec=0):
    if model is not None:
        model.eval()
    device = 'cuda'
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        end = time.time()
        loader = dataset.test_loader
        coefs = torch.tensor([]).to(device)
        dname = type(dataset).__name__
        if dname == 'ImageNet2p':
            loader = dataset.train_loader
            # assert to make sure the imagenet held-out minival logic is consistent across machines.
            assert dataset.train_dataset.__getitem__(dataset.sampler.indices[1000])['image_paths'].endswith('n01675722_4108.JPEG')

        for i, batch in enumerate(tqdm(loader)):
            batch = maybe_dictionarize_batch(batch)
            # pdb.set_trace()
            inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
            if half_prec:
                inputs = inputs.half()

            data_time = time.time() - end
            y = labels
            if 'image_paths' in batch:
                image_paths = batch['image_paths']

            logits, features = model(inputs, return_features=True)

            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            if isinstance(logits, list):
                logits = logits[0]

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths, None)
                correct += acc1
                n += num_total
            else:
                correct_current = pred.eq(y.view_as(pred)).sum().item()
                correct += correct_current
                n += y.size(0)

            batch_time = time.time() - end
            end = time.time()
            if i % 20 == 0:
                percent_complete = 100.0 * i / len(loader)
                print(
                    f"[{percent_complete:.0f}% {i}/{len(loader)}]\t"
                    f"Acc: {100 * (correct/n):.2f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                )

        top1 = correct / n

    
        return top1


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def identity(x,device='cuda'):
    return x

def compute_sample_loss_uncertainty(logits, labels, temperature=1.0,pseudo_label=0,prob=False):
    if pseudo_label in [1,2]:
        loss = (-labels * F.log_softmax(logits,1)).sum(1)
        #pdb.set_trace()
    else:
        loss = F.cross_entropy(logits, labels, reduction='none')

    if prob:
        uncertainty = - (logits * torch.log(logits)).sum(dim=1)
    else:
        uncertainty = - (F.softmax(logits/temperature,dim=1) * F.log_softmax(logits/temperature, dim=1)).sum(dim=1)
    return loss.reshape(-1,), uncertainty.reshape(-1,)

def compute_accuracy(logits, labels):
    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(labels.view_as(pred)).sum().item()
    n = labels.size(0)
    acc = correct / n

    return acc

def compute_sample_accuracy(logits, labels):
    pred = logits.argmax(dim=1, keepdim=True)
    return pred.eq(labels.view_as(pred))

def compute_sample_uncertainty(logits, temperature=1.0):
    uncertainty = - (F.softmax(logits/temperature,dim=1) * F.log_softmax(logits/temperature, dim=1)).sum(dim=1)
    return uncertainty.reshape(-1,)

def compute_sample_loss_uncertainty_acc(logits, labels, temperature=1.0, pseudo_label=0):
    if pseudo_label in [1,2]:
        loss = (-labels * F.log_softmax(logits,1)).sum(1)
        labels = torch.argmax(labels, dim=1).reshape(-1,1)
    else:
        loss = F.cross_entropy(logits, labels, reduction='none')
    pred = logits.argmax(dim=1, keepdim=True)

    uncertainty = - (F.softmax(logits/temperature,dim=1) * F.log_softmax(logits/temperature, dim=1)).sum(dim=1)
    return loss.reshape(-1,), pred.eq(labels.view_as(pred)), uncertainty.reshape(-1,)

def get_feats_logits_and_labels(model, preprocess, args, dataset_name=None,state_dict=None,use_cache=True):    
    if args.model == 'ViT-B/32': mpf = 'CLIPVITB32_'
    if args.model == 'ViT-B/16': mpf = 'CLIPVITB16_'
    if args.model == 'ViT-L/14': mpf = 'CLIPVITL14_'
    
    args.prefix = mpf + args.prefix
    
    if args.half_prec: args.prefix += '_fp16_'

    save_path = os.path.join(args.cache_dir, args.prefix + '_' + dataset_name + '_cache.pt')
    device = args.device

    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class(
        preprocess,
        args.data_location,
        args.batch_size,
        args.workers
    )
    project_fn = getattr(dataset, 'project_logits', None)
    dataloader = dataset.test_loader
    if dataset_name == 'ImageNet2p':
        dataloader = dataset.train_loader

    if os.path.exists(save_path):
        cache_data = torch.load(save_path)
        print(f'use cache from {save_path}')
        logits = cache_data['logits'].to(device)
        labels = cache_data['labels'].to(device)
        feats = cache_data['feats'].to(device)
    else:
        print(f"do not find cache in {save_path}")
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
        
        if state_dict is not None:
            model = get_model_from_sd(state_dict, model)

        model.eval()
        logits, labels, feats = [], [], []
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                data = maybe_dictionarize_batch(data)
                x = data['images'].to(device)
                if args.half_prec:
                    x = x.half()

                label = data['labels'].to(device)
                # if 'image_paths' in data:
                #     image_paths = data['image_paths']
                logit, feat = model(x, return_features=True)

                labels.append(label)
                logits.append(logit)
                feats.append(feat)

                if project_fn is not None:
                    logit = project_fn(logit, device)
                if hasattr(dataset, 'project_labels'):
                    label = dataset.project_labels(label, device)
                if isinstance(logit, list):
                    logit = logit[0]
                pred = logit.argmax(dim=1, keepdim=True).to(device)
    
        labels = torch.cat(labels)
        logits = torch.cat(logits)
        feats = torch.cat(feats)
        print(f'successfully build the cache file at {save_path}')
        torch.save({'logits': logits.cpu(), 'labels': labels.cpu(), 'feats': feats.cpu()}, save_path)

    if project_fn is not None:
        return feats, logits, labels, project_fn
    else:
        return feats, logits, labels, identity


def get_feats_logits(model, args, dataloader, project_fn):
    model.eval()
    logits = []
    top1, correct, n = 0., 0., 0.
    device = args.device
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            data = maybe_dictionarize_batch(data)
            x = data['images'].to(device)
            if args.half_prec:
                x = x.half()

            if 'image_paths' in data:
                image_paths = data['image_paths']
            #pdb.set_trace()
            logit, _ = model(x, return_features=True)

            if project_fn is not None:
                logit = project_fn(logit, device)
            if isinstance(logit, list):
                logit = logit[0]
            pred = logit.argmax(dim=1, keepdim=True).to(device)
            logits.append(logit)

        logits = torch.cat(logits)
        
    if project_fn is not None:
        return logits, project_fn
    else:
        return logits, identity


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster