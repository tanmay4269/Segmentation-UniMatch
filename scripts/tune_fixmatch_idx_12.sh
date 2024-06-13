#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

save_path=/data1/tgajpati/ss2_ty/Reimplementing-UniMatch/exp/idx_12

mkdir -p $save_path

export CUDA_VISIBLE_DEVICES=2
export TUNE_DISABLE_STRICT_METRIC_CHECKING=1

python ray_tune.py \
    --project_name=ss2-ssl-idx-12 \
    --model_name=debug-fixmatch-w-cutmix-1 \
    --search_alg=bohb \
    \
    --dataset=idx_12 \
    --nclass=3 \
    \
    --num_samples=4 \
    --num_epochs=20 \
    --save_path=$save_path \
    2>&1 | tee $save_path/$now.log
    
    # --use_checkpoint \
    # --enable_logging \