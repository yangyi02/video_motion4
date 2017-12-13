#!/bin/bash

source ../../set_path.sh

CUDA_VISIBLE_DEVICES=1 python ../demo.py --train --data=chair --tensorboard_path=../../tensorboard/exp08_004 --batch_size=32 --image_size=128 --motion_range=2 --num_frame=3 --net_depth=13 --bg_move --train_epoch=6000 --save_interval=6001 --test_interval=200 --test_epoch=20 --learning_rate=0.001 2>&1 | tee train.log
