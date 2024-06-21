#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

model_name=sll-fm-no-aug-ur10-lr2e-4

method=fixmatch
dataset=3
nclass=1

export CUDA_VISIBLE_DEVICES=2
dataroot=/data1/tgajpati/ss2_ty/tmp/Segmentation-UniMatch/dataset/kerogens
# dataroot=/home/tvg/Segmentation-UniMatch/dataset/kerogens
num_epochs=200
epochs_before_eval=5

# num_epochs=10
# epochs_before_eval=1
    # --enable_logging \

####################
project_name=ss2-idx-$dataset

save_path=$(pwd)/exp/idx_$dataset/$method/$now/
mkdir -p $save_path

python ${method}.py \
    --project_name=$project_name \
    --model_name=$model_name \
    --enable_logging \
    \
    --dataroot=$dataroot \
    --dataset=idx_$dataset \
    --nclass=$nclass \
    \
    --num_epochs=$num_epochs \
    --epochs_before_eval=$epochs_before_eval \
    --save_path=$save_path \
    2>&1 | tee $save_path/$now.log

