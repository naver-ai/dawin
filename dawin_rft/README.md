# DaWin - Robust Fine-tuning Experiments

## Setup
* Start with installation for the environment with `conda env create -f dawin.yml`
* **Directory structure**
    * The directory structure should seem like below.
    ```
    ├── cache # after the first run, cache files will be saved here
    │   ├── CLIPVITB32_ft_ImageNet_cache.pt
    │   ├── CLIPVITB32_zs_ImageNet_cache.pt
    │   ├── CLIPVITB32_ft_ImageNetV2_cache.pt
    │   ├── CLIPVITB32_zs_ImageNetV2_cache.pt
    │   ├── CLIPVITB32_ft_{*}_cache.pt # {*} denote datasetname
    │   ├── CLIPVITB32_zs_{*}_cache.pt
    ├── checkpoints
    │   ├── zeroshot_clipvitb32.pt
    │   ├── finetune_clipvitb32.pt
    │   ├── *.pt # checkpoints for baseline methods
    ├── data
    │   ├── imagenet
    │   ├── imagenet-r
    │   ├── *
    ├── datasets
    │   ├── imagenet.py
    │   ├── imagenet_r.py
    │   ├── *
    ├── script
    │   ├── dawin.sh
    │   ├── dawin_samplewise.sh
    │   ├── dawin_applications.sh
    │   ├── dawin_ablation.sh
    │   ├── dawin_baselines.sh
    ├── main_dawin.py
    ├── main.py
    ├── mixturemodel.py
    ├── utils.py
    ├── *.py
    ```
    * Place your checkpoints (zero-shot, fine-tuned, and other baseline models') in the `checkpoints/`
    * Place your datasets in `data/`

    ``` linux
    mkdir cache checkpoints data analysis
    ln -s 'YOURDATAPATH' ./dawin/dawin_rft/data
    ```

* **Model checkpoints**
  * We reached out the authors of [Model Stock](https://github.com/naver-ai/model-stock) to get their checkpoints including individual fine-tuned models, model soup, and model stock weights that are used for the robust fine-tuning experiments
  * Here are the fine-tuned model weights we use for merging: [google drive](https://drive.google.com/drive/folders/1NZ9kxXmsFWuY6oSAozTaaZwSISlZ_RgX?usp=sharing)
  * You can also get weights of other baseline methods from the official [Model Stock repository](https://github.com/naver-ai/model-stock/blob/main/notebooks/model_stock_eval.ipynb).
* **Dataset**
  * For the robust fine-tuning experiments, we adopted ImageNet distribution shifts benchmarks:
    * In-distribution (ID): [ImageNet-1K](https://www.image-net.org/download.php)
    * Out-of-distribution (OOD): [ImageNetV2](https://imagenetv2.org/), [ImageNet-Rendition](https://github.com/hendrycks/imagenet-r), [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch), and [ObjectNet](https://objectnet.dev/)
  * Refer to [`datasets.md`](https://github.com/mlfoundations/wise-ft/blob/master/datasets.md) of WiSE-FT repository to prepare these datasets. 


## Run baseline methods
* After preparing the model checkpoints, you could evaluate individual checkpoints (`*.pt`) through the commend as below
  > `main.py --eval-single-model *.pt`
* You could also reproduce the WiSE-FT results as below 
  > `main.py --wiseft-alpha 0.5 --wiseft-zspath checkpoints/zeroshot.pt --wiseft-ftpath checkpoints/finetune.pt`

## Run DaWin
We provide the scripts to reproduce main results, ablation study, and extended applications (cover Table 1, 2, 3, 4, 5, Figure 5, and some tables in Appendix).
* You could reproduce DaWin's evaluation across ImageNet variants with CLIP ViT-B/32 as below
  ```
  CUDA_VISIBLE_DEVICES=0 python3 main_dawin.py --seed 1 \
  --cache-dir cache/ --data-location data/ --model-location checkpoints/ \
  --model ViT-B/32 \
  --zs-path checkpoints/zeroshot_clipvitb32.pt --ft-path checkpoints/finetune_clipvitb32.pt \
  --eval-batch-size 1024 --bmm_ncluster 3 \
  --offset-adjustment \
  --expertise negexp_ent_ratio
  ```
  * refer to `sh script/vitb32_dawin.sh 0`
  * When you run the DaWin method for the first time, this produces cache files (at `./cache/`) containing logits and predicts of zero-shot and fine-tuned CLIP. Subsequent runs will be faster by loading these cache files.
* To reproduce the ablation study over expertise metric (such as pseudo label-based X-entropy), run the command below
  ```
  for psl in 1 2 3 4
  do

  CUDA_VISIBLE_DEVICES=0 python3 main_dawin.py --seed 1 \
  --cache-dir cache/ --data-location data/ --model-location checkpoints/ \
  --model ViT-B/32 \
  --zs-path checkpoints/zeroshot_clipvitb32.pt --ft-path checkpoints/finetune_clipvitb32.pt \
  --eval-batch-size 1024 --bmm_ncluster 3 \
  --offset-adjustment \
  --expertise negexp_loss_ratio --pseudo_label $psl

  done
  ```
  * refer to `script/dawin_ablation.sh`
* To reproduce application scenarios -- dynamic classifier selection and dynamic output ensemble -- of DaWin, run the commands below
  ```
  CUDA_VISIBLE_DEVICES=0 python3 main_dawin.py --seed 1 \
  --cache-dir cache/ --data-location data/ --model-location checkpoints/ \
  --model ViT-B/32 \
  --zs-path checkpoints/zeroshot_clipvitb32.pt --ft-path checkpoints/finetune_clipvitb32.pt \
  --eval-batch-size 1024 \
  --expertise selective_entropy
  ```
  * refer to `script/applications.sh`
* To simulate the sample-wise dynamic interpolation approaches, run the command below 
  ```
  CUDA_VISIBLE_DEVICES=0 python3 main_dawin.py --seed 1 \
  --cache-dir cache/ --data-location data/ --model-location checkpoints/ \
  --model ViT-B/32 \
  --zs-path checkpoints/zeroshot_clipvitb32.pt --ft-path checkpoints/finetune_clipvitb32.pt \
  --eval-batch-size 1 --bmm_ncluster 0 \
  --expertise negexp_ent_ratio --offset-adjustment
  ```
  * refer to `dawin_samplewise.sh` 
  * Due to its reliance on the sample-wise merging operation, it takes about 2.5 hours to evaluate 50,000 images (e.g., ImageNet test set) on a single NVIDIA A100 GPU. Evalution for entire ImageNet variants require roughly 10 hours.

## Acknowledgement
Some code blocks are borrowed from the projects below, we appreciate the authors' endeavors.
- WiSE-FT: https://github.com/mlfoundations/wise-ft
- Model Soups: https://github.com/mlfoundations/model-soups
