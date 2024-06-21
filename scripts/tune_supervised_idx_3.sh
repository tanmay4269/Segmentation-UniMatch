#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

model_name=sl-tune-pretrained-gray
method=supervised

# dataroot=/data1/tgajpati/ss2_ty/tmp/Segmentation-UniMatch/dataset/kerogens/
dataroot=/home/tvg/Segmentation-UniMatch/dataset/kerogens
dataset=3
nclass=1

export CUDA_VISIBLE_DEVICES=0
search_alg=hyperopt
num_samples=9
num_epochs=200
epochs_before_eval=5

num_samples=2
num_epochs=20
epochs_before_eval=1
    # --enable_logging \

####################
project_name=ss2-idx-$dataset

save_path=$(pwd)/exp/idx_$dataset/$method/$now/
mkdir -p $save_path

python tune_${method}_idx_${dataset}.py \
    --project_name=$project_name \
    --model_name=$model_name \
    --search_alg=$search_alg \
    \
    --dataroot=$dataroot \
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
