#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

save_path=/data1/tgajpati/ss2_ty/Reimplementing-UniMatch/exp/idx_3

mkdir -p $save_path

export CUDA_VISIBLE_DEVICES=2

python ray_tune_idx_3.py \
    --project_name=ss2-ssl-idx-3 \
    --model_name=fixmatch-w-cutmix-1 \
    --search_alg=bohb \
    --enable_logging \
    \
    --dataset=idx_3 \
    --nclass=1 \
    \
    --num_samples=20 \
    --num_epochs=200 \
    --epochs_before_eval=5 \
    --save_path=$save_path \
    2>&1 | tee $save_path/$now.log
    
    # --use_checkpoint \
    # --fast_debug \
