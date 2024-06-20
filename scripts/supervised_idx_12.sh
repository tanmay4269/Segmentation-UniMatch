#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

model_name=sl-no-aug-pretrain-1
# model_name=debug

method=supervised
dataset=12
nclass=3

export CUDA_VISIBLE_DEVICES=2
num_epochs=300
epochs_before_eval=5

# num_epochs=5
# epochs_before_eval=1

####################
project_name=ss2-idx-$dataset

save_path=$(pwd)/exp/idx_$dataset/$method/$now/
mkdir -p $save_path

python ${method}.py \
    --project_name=$project_name \
    --model_name=$model_name \
    --enable_logging \
    \
    --dataset=idx_$dataset \
    --nclass=$nclass \
    \
    --num_epochs=$num_epochs \
    --epochs_before_eval=$epochs_before_eval \
    --save_path=$save_path \
    2>&1 | tee $save_path/$now.log

