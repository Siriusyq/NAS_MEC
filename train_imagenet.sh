#!/bin/bash
# mpirun -np 8 \
#     -H localhost:8\
#     -bind-to none -map-by slot \
#     -x NCCL_DEBUG=INFO -x PATH \
#     -mca pml ob1 -mca btl ^openib \

    # python train_imagenet.py \
    #     --arch proxyless_gpu \
    #     --train-dir $HOME/Dataset/imagenet/train \
    #     --val-dir $HOME/Dataset/imagenet/val \
    #     --fp16-allreduce \
    #     --color-jitter \
    #     --label-smoothing \
    #     --epochs 300 
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python train_imagenet.py \
        --arch proxyless_condition1_100_140 \
        --batch-size 256 \
        --train-dir $HOME/Dataset/imagenet/train \
        --val-dir $HOME/Dataset/imagenet/val \
        --fp16-allreduce \
        --color-jitter \
        --label-smoothing \
        --epochs 300 