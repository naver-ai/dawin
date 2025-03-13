cd ..

CUDA_VISIBLE_DEVICES=0 python3 main_dawin.py --seed 1 \
--cache-dir cache/ --data-location data/ --model-location checkpoints/ \
--model ViT-B/32 \
--zs-path checkpoints/zeroshot_clipvitb32.pt --ft-path checkpoints/finetune_clipvitb32.pt \
--eval-batch-size 1024 \
--expertise selective_entropy

CUDA_VISIBLE_DEVICES=0 python3 main_dawin.py --seed 1 \
--cache-dir cache/ --data-location data/ --model-location checkpoints/ \
--model ViT-B/32 \
--zs-path checkpoints/zeroshot_clipvitb32.pt --ft-path checkpoints/finetune_clipvitb32.pt \
--eval-batch-size 1024 \
--expertise negexp_ent_ratio --offset-adjustment \
--ensemble_coef dynamic