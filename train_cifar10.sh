#!/bin/bash

    # mpirun -np 8 \
    #     -H localhost:8\
    #     -bind-to none -map-by slot \
    #     -x NCCL_DEBUG=INFO -x PATH \
    #     -mca pml ob1 -mca btl ^openib \
    python train_cifar10.py \
        --arch proxyless_gpu \
        --save_path $HOME/Dataset/cifar10 \
        --fp16-allreduce \
        --color-jitter \
        --label-smoothing \
        --epochs 300 
