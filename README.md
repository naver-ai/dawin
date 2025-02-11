## [DaWin: Training-free Dynamic Weight Interpolation for Robust Adaptation](https://arxiv.org/abs/2410.03782)

> [Changdae Oh<sup>1,2*](https://changdaeoh.github.io/), [Sharon Li<sup>2](https://pages.cs.wisc.edu/~sharonli/), [Kyungwoo Song<sup>3&dagger;](https://gtshs2.github.io/), [Sangdoo Yun<sup>1&dagger;](https://sangdooyun.github.io/), [Dongyoon Han<sup>1&dagger;](https://dongyoonhan.github.io/), <br>
> <sup> <sup>*</sup> Work done during an internship at NAVER AI Lab, &dagger; corresponding authors </sup> <br>
> <sup>1</sup>[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic/ai-lab), <sup>2</sup>[University of Wisconsin--Madison](https://www.wisc.edu/), <sup>3</sup>[Yonsei University](https://www.yonsei.ac.kr/en_sc/index.jsp)


[![paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2410.03782)

<br>

<img width="1262" alt="image" src="https://github.com/user-attachments/assets/c29d8d71-5986-43cf-8dca-a978384bdafe">


### Abstract
>Adapting a pre-trained foundation model on downstream tasks should ensure robustness against distribution shifts without the need to retrain the whole model. Although existing weight interpolation methods are simple yet effective, we argue their static nature limits downstream performance while achieving efficiency. In this work, we propose DaWin, a training-free dynamic weight interpolation method that leverages the entropy of individual models over each unlabeled test sample to assess model expertise, and compute per-sample interpolation coefficients dynamically. Unlike previous works that typically rely on additional training to learn such coefficients, our approach requires no training. Then, we propose a mixture modeling approach that greatly reduces inference overhead raised by dynamic interpolation. We validate DaWin on the large-scale visual recognition benchmarks, spanning 14 tasks across robust fine-tuning -- ImageNet and derived five distribution shift benchmarks -- and multi-task learning with eight classification tasks. Results demonstrate that DaWin achieves significant performance gain in considered settings, with minimal computational overhead. We further discuss DaWin's analytic behavior to explain its empirical success.


### Updates
* (2025/01/22): Our manuscript has been accepted at ICLR 2025🎉🎉; Full code will be released very soon!
* (2024/10/09): A short version of the preprint has been accepted at [NeurIPS 2024 Workshop on Adaptive Foundation Models](https://adaptive-foundation-models.org/)🎉
* (2024/10/03): Code is under internal review.
* (2024/10/03): [Preprint](https://arxiv.org/abs/2404.09490) has been uploaded.
