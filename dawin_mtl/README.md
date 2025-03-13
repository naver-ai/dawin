# DaWin - Multi-task Learning Experiments

Set the path first
```
cd dawin_mtl
mkdir cache checkpoints
export PYTHONPATH="$PYTHONPATH:$PWD"
```

## Prepare datasets
* Following previous works, we evaluate DaWin on eight visual recognition benchmarks: SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, and DTD.
* Refer to dataset processing in the [task_vectors](https://github.com/mlfoundations/task_vectors).
* Or you can download the processed data from [Baidu Cloud disk](https://pan.baidu.com/s/1w0Z2UVv3NVmqDhjH8WTOJQ?pwd=kvg6) or [HugggingFace](https://huggingface.co/collections/tanganke/image-classification-datasets-662abda7d75efe6b0e6b43da).
* After download datasets, you may need to run `split_dataset.py` to appropriately set splits per dataset
* Then, make a symbolic link here `ln -s YOURDATAPATH ./dawin/dawin_mtl/data`

## Prepare checkpoints
* You can mannually download the eight fine-tuned models' checkpoints (and corresponding classifier heads) from the [here](https://github.com/mlfoundations/task_vectors#checkpoints).
* [Public Google Drive url of Task Arithmetic authors](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw)
* Or you can refer to `python download_ckpt.py` script to download all the required checkpoints.

## Run
Run the following commands at `src/` to reproduce Table 6 -- main MTL experiments.

* Run DaWin (Ours)
    > `python main_dawin.py`
* Run Individual Fine-tuned Models' Evaluation
    > `python main_individuals.py`
* Run Weight Averaging [paper1](https://arxiv.org/abs/2203.05482), [paper2](https://arxiv.org/abs/2208.05592)
    > `python main_weight_avg.py`
* Run Task Atithmetic [paper](https://arxiv.org/abs/2212.04089)
    > `python main_task_arithmetic.py`
* Run TIES-MERGING [paper](https://arxiv.org/abs/2306.01708)
    > `python main_ties_merging.py`
* Run Task-wise AdaMerging [paper](https://arxiv.org/abs/2310.02575)
    > `python main_task_wise_adamerging.py`
* Run Task-wise AdaMerging++ [paper](https://arxiv.org/abs/2310.02575)
    > `python main_task_wise_adamergingpp.py`
* Run Layer-wise AdaMerging [paper](https://arxiv.org/abs/2310.02575)
    > `python main_layer_wise_adamerging.py`
* Run Layer-wise AdaMerging++ [paper](https://arxiv.org/abs/2310.02575)
    > `python main_layer_wise_adamergingpp.py`

## Acknowledgement
This repository is built on top of [AdaMerging](https://github.com/EnnengYang/AdaMerging) and other code blocks are somewhat borrowed from the [ModelSoup](https://github.com/mlfoundations/model-soups) project, we appreciate the authors' endeavors.