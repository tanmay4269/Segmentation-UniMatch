#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

model_name=sl-tune-no-pretraining

method=supervised
dataset=3
nclass=1

export CUDA_VISIBLE_DEVICES=0
search_alg=hyperopt
num_samples=10
num_epochs=300
epochs_before_eval=5

# num_samples=1
# epochs_before_eval=1

####################
project_name=ss2-idx-$dataset

save_path=$(pwd)/exp/idx_$dataset/$method/
mkdir -p $save_path

python tune_${method}_idx_${dataset}.py \
    --project_name=$project_name \
    --model_name=$model_name \
    --search_alg=$search_alg \
    --enable_logging \
    \
    --dataset=idx_$dataset \
    --nclass=$nclass \
    \
    --num_samples=$num_samples \
    --num_epochs=$num_epochs \
    --epochs_before_eval=$epochs_before_eval \
    --save_path=$save_path \
    2>&1 | tee $save_path/$now.log
    
    # --use_checkpoint \
    # --fast_debug \
