#!/bin/bash

bin/language-modeling-with-boxes train \
 --batch_size=4096 --box_type=BoxTensor \
 --data_device=gpu \
 --dataset='h_and_m' \
 --embedding_dim=64 \
 --eval_file=./data/similarity_datasets/ \
 --int_temp=1.9678289474987882 \
 --log_frequency=50 \
 --loss_fn=max_margin \
 --lr=0.004204091643267762 \
 --margin=5 \
 --model_type=Word2BoxConjunction \
 --n_gram=5 \
 --negative_samples=10 \
 --num_epochs=10 \
 --subsample_thresh=0.001 \
 --vol_temp=0.33243242379830407 \
 --save_model \
 --add_pad
