cd ..

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

for expt in maxlogit_ratio expconf_ratio expconfdiff_ratio
do

CUDA_VISIBLE_DEVICES=0 python3 main_dawin.py --seed 1 \
--cache-dir cache/ --data-location data/ --model-location checkpoints/ \
--model ViT-B/32 \
--zs-path checkpoints/zeroshot_clipvitb32.pt --ft-path checkpoints/finetune_clipvitb32.pt \
--eval-batch-size 1024 --bmm_ncluster 3 \
--expertise $expt

done