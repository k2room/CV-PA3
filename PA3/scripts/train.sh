#!/usr/bin/env bash
#--dataset_root /jisu/dataset/iHarmony4resized256/ \

python train.py \
--dataset_root /home/work/CV-PA3/PA3/dataset_processing/ \
--name experiment_IN_train \
--checkpoints_dir /home/work/CV-PA3/PA3/checkpoints/IN \
--model rainnet \
--netG rainnet \
--dataset_mode iharmony4 \
--is_train 1 \
--gan_mode wgangp \
--normD instance \
--normG RAIN \
--preprocess None \
--niter 100 \
--niter_decay 100 \
--input_nc 3 \
--batch_size 11 \
--num_threads 6 \
--lambda_L1 100 \
--print_freq 400 \
--gpu_ids 0,1 \
#--continue_train \
#--load_iter 87 \
#--epoch 88 \
