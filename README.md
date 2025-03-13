## [DaWin: Training-free Dynamic Weight Interpolation for Robust Adaptation](https://arxiv.org/abs/2410.03782)

> [Changdae Oh<sup>1,2*](https://changdaeoh.github.io/), [Sharon Li<sup>2](https://pages.cs.wisc.edu/~sharonli/), [Kyungwoo Song<sup>3&dagger;](https://gtshs2.github.io/), [Sangdoo Yun<sup>1&dagger;](https://sangdooyun.github.io/), [Dongyoon Han<sup>1&dagger;](https://dongyoonhan.github.io/), <br>
> <sup> <sup>*</sup> Work done during an internship at NAVER AI Lab, &dagger; corresponding authors </sup> <br>
> <sup>1</sup>[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic/ai-lab), <sup>2</sup>[University of Wisconsin--Madison](https://www.wisc.edu/), <sup>3</sup>[Yonsei University](https://www.yonsei.ac.kr/en_sc/index.jsp)


[![paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2410.03782)
[![Paper](https://img.shields.io/badge/Paper-ICLR_2025-blue)](https://openreview.net/forum?id=L8e7tBf4pP)

<br>

<img width="1262" alt="image" src="https://github.com/user-attachments/assets/c29d8d71-5986-43cf-8dca-a978384bdafe">


### Abstract
>Adapting a pre-trained foundation model on downstream tasks should ensure robustness against distribution shifts without the need to retrain the whole model. Although existing weight interpolation methods are simple yet effective, we argue their static nature limits downstream performance while achieving efficiency. In this work, we propose DaWin, a training-free dynamic weight interpolation method that leverages the entropy of individual models over each unlabeled test sample to assess model expertise, and compute per-sample interpolation coefficients dynamically. Unlike previous works that typically rely on additional training to learn such coefficients, our approach requires no training. Then, we propose a mixture modeling approach that greatly reduces inference overhead raised by dynamic interpolation. We validate DaWin on the large-scale visual recognition benchmarks, spanning 14 tasks across robust fine-tuning -- ImageNet and derived five distribution shift benchmarks -- and multi-task learning with eight classification tasks. Results demonstrate that DaWin achieves significant performance gain in considered settings, with minimal computational overhead. We further discuss DaWin's analytic behavior to explain its empirical success.


### Updates
* (2025/03/14): Our code is now available! 
* (2025/01/22): Our manuscript has been accepted at [ICLR 2025](https://iclr.cc/)🎉🎉; 
* (2024/10/09): A short version of the preprint has been accepted at [NeurIPS 2024 Workshop on Adaptive Foundation Models](https://adaptive-foundation-models.org/)🎉
* (2024/10/03): Code is under internal review.
* (2024/10/03): [Preprint](https://arxiv.org/abs/2410.03782) has been uploaded.

---

## Installation
> conda env create -f dawin.yaml 

## Application-specific Instructions
* To reproduce multi-task learning experiments, refer to `dawin_rft/` and corresponding `README.md` for detailed instructions.
* To reproduce multi-task learning experiments, refer to `dawin_mtl/` and corresponding `README.md` for detailed instructions.

---

## Quick Start with Pseudo-code
```python
def interpolation(alpha, weight1, weight2):
    return {key: (1 - alpha) * weight1[key] + alpha * weight2[key] for key in weight1.keys()}

basemodel, _ = clip.load(args.model, 'cpu', jit=False)

# load state_dict of individual models to be interpolated 
sd_pt = torch.load(args.pt_path, map_location=torch.device(args.device))
sd_ft = torch.load(args.ft_path, map_location=torch.device(args.device))

# get entropy from the outputs of each model for all samples
logit_pt = get_logits(basemodel, dataset_name=args.dsname, state_dict=sd_pt)
logit_ft = get_logits(basemodel, dataset_name=args.dsname, state_dict=sd_ft)
ent_pt = - (F.softmax(logit_pt,dim=1) * F.log_softmax(logit_pt, dim=1)).sum(dim=1)
ent_ft = - (F.softmax(logit_ft,dim=1) * F.log_softmax(logit_ft, dim=1)).sum(dim=1)

# exponentiated negative entropy as model expertise to weigh the interpolation
expertise_pt = (-ent_pt).exp()
expertise_ft = (-ent_ft).exp()
lambdas = (expertise_ft) / (expertise_pt + expertise_ft)

# sample-wise interpolation (w/o Beta Mixture Modeling)
eval_dataloader = torch.utils.data.DataLoader(..., batch_size=1, shuffle=False)
correct, n = 0., 0.
for i, (inputs, labels) in enumerate(eval_dataloader):
    inputs, labels = inputs.cuda(), labels.cuda()
    merged_sd = interpolation(lambdas[i], sd_pt, sd_ft)
    model = get_model_from_sd(merged_sd, basemodel)
    logits = model(inputs)

    preds = logits.argmax(dim=1, keepdim=True).to(device)
    correct_current = preds.eq(labels.view_as(preds)).sum().item()
    correct += correct_current
    n += labels.size(0)

top1_acc = correct / n
```

---

## How to cite
```
@inproceedings{
oh2025dawin,
title={DaWin: Training-free Dynamic Weight Interpolation for Robust Adaptation},
author={Changdae Oh and Yixuan Li and Kyungwoo Song and Sangdoo Yun and Dongyoon Han},
booktitle={International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=L8e7tBf4pP}
}
```

## License
```
DaWin
Copyright (c) 2025-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
