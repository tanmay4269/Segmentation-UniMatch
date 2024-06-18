#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

save_path=/data1/tgajpati/ss2_ty/Segmentation-UniMatch/exp/idx_3/fixmatch/

mkdir -p $save_path

export CUDA_VISIBLE_DEVICES=1

python tune_fixmatch_idx_3.py \
    --project_name=ss2-ssl-idx-3 \
    --model_name=fixmatch-loss-types \
    --search_alg=hyperopt \
    --enable_logging \
    \
    --dataset=idx_3 \
    --nclass=1 \
    \
    --num_samples=150 \
    --num_epochs=200 \
    --epochs_before_eval=5 \
    --save_path=$save_path \
    2>&1 | tee $save_path/$now.log
    
    # --use_checkpoint \
    # --fast_debug \
