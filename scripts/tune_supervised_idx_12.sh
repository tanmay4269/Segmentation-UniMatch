#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

project_name=ss2-supervised-idx-12
model_name=augment-labeled-2

method=supervised
dataset=idx_12
nclass=3

export CUDA_VISIBLE_DEVICES=0
search_alg=hyperopt
num_samples=90
num_epochs=120
epochs_before_eval=5

# num_samples=1
# epochs_before_eval=1

####################

save_path=$(pwd)/exp/idx_$dataset/$method/
mkdir -p $save_path

python tune_${method}_${dataset}.py \
    --project_name=$project_name \
    --model_name=$model_name \
    --search_alg=$search_alg \
    --enable_logging \
    \
    --dataset=$dataset \
    --nclass=$nclass \
    \
    --num_samples=$num_samples \
    --num_epochs=$num_epochs \
    --epochs_before_eval=$epochs_before_eval \
    --save_path=$save_path \
    2>&1 | tee $save_path/$now.log
    
    # --use_checkpoint \
    # --fast_debug \
