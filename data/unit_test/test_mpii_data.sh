#!/bin/bash

source ../../set_path.sh

python ../../data/real/mpii_data.py --batch_size=1 --image_size=32 --resolution=128 --image_channel=1 --num_frame=1 --save_display --save_display_dir=./

python ../../data/real/mpii_data.py --batch_size=2 --image_size=64 --resolution=128 --num_frame=2 --save_display --save_display_dir=./

python ../../data/real/mpii_data.py --batch_size=3 --image_size=128 --resolution=256 --num_frame=3 --min_diff_thresh=0.02 --max_diff_thresh=0.2 --save_display --save_display_dir=./

python ../../data/real/mpii_data.py --batch_size=4 --image_size=256 --resolution=256 --num_frame=4 --min_diff_thresh=0.02 --max_diff_thresh=0.2 --diff_div_thresh=1.1 --save_display --save_display_dir=./

sh trim.sh
