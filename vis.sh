source activate /iag_ad_01/ad/yuanweizhong/miniconda/vggt
# time python vis.py --scene_dir /iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser/camera/example_dataset/example_scene1/images --intrinsics_dir /iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser/camera/example_dataset/example_scene1/intrins --extrinsics_dir /iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser/camera/example_dataset/example_scene1/poses

root_path=/iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser/camera/example_dataset/example_scene3 
time python -m ptvsd --host 0.0.0.0 --port 5692 vis.py --scene_dir ${root_path}/images --intrinsics_dir ${root_path}/intrins --poses_dir ${root_path}/poses --query_frame_num 1
# time python -m pdb vis.py --scene_dir ${root_path}/images --intrinsics_dir ${root_path}/intrins --poses_dir ${root_path}/poses
