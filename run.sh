#!/bin/bash
source activate /iag_ad_01/ad/yuanweizhong/miniconda/vggt/
python -m ptvsd --host 0.0.0.0 --port 5691 demo_colmap.py --scene_dir='/iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser/camera/example_dataset/example_scene_sample' --use_ba
# python demo_colmap.py --scene_dir='/iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser/camera/example_dataset/example_scene_sample' --use_ba