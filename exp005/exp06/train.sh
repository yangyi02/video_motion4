#!/bin/bash

source ../../set_path.sh

CUDA_VISIBLE_DEVICES=1 python ../demo.py --train --data=chair --tensorboard_path=../../tensorboard/exp06_005 --batch_size=32 --image_size=128 --motion_range=2 --num_frame=2 --net_depth=13 --train_epoch=2000 --save_interval=2002 --test_interval=200 --test_epoch=20 --learning_rate=0.001 2>&1 | tee train.log
