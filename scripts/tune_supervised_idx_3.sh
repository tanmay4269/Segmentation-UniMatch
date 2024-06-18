#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

save_path=/data1/tgajpati/ss2_ty/Segmentation-UniMatch/exp/idx_3/supervised/

mkdir -p $save_path

export CUDA_VISIBLE_DEVICES=0

python tune_supervised_idx_3.py \
    --project_name=ss2-supervised-idx-3 \
    --model_name=kinda-exhaustive-search-1 \
    --search_alg=hyperopt \
    --enable_logging \
    \
    --dataset=idx_3 \
    --nclass=1 \
    \
    --num_samples=90 \
    --num_epochs=200 \
    --epochs_before_eval=5 \
    --save_path=$save_path \
    2>&1 | tee $save_path/$now.log
    
    # --use_checkpoint \
    # --fast_debug \
