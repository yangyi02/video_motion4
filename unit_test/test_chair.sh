#!/bin/bash

source ../set_path.sh

python ../base_demo.py --test_gt --data=chair --batch_size=32 --image_size=32 --image_channel=1 --motion_range=2 --num_frame=3 --test_epoch=1 --display --save_display --save_display_dir=./

python ../base_demo.py --test_gt --data=chair --batch_size=32 --image_size=32 --motion_range=2 --num_frame=3 --test_epoch=1 --display --save_display --save_display_dir=./

python ../base_demo.py --train --data=chair --batch_size=32 --image_size=32 --motion_range=2 --num_frame=3 --train_epoch=5 --test_interval=5 --test_epoch=1 --save_dir=./ 

python ../base_demo.py --test --data=chair --init_model=model.pth --batch_size=32 --image_size=32 --motion_range=2 --num_frame=3 --test_epoch=1 --display --save_display --save_display_dir=./

sh trim.sh

rm model.pth
