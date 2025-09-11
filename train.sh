#!/bin/bash
source activate /iag_ad_01/ad/yuanweizhong/miniconda/vggt/
export http_proxy=http://proxy.hs.com:3128
export https_proxy=http://proxy.hs.com:3128
export HTTP_PROXY=http://proxy.hs.com:3128
export HTTPS_PROXY=http://proxy.hs.com:3128
# python robot.py
cd training
time torchrun --nproc_per_node=8 --master_port=29502 launch.py
cd ..
python robot.py