# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo with GT Camera")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--intrinsics_dir", type=str, required=True, help="Directory containing intrinsic parameter txt files")
    parser.add_argument("--extrinsics_dir", type=str, required=True, help="Directory containing extrinsic parameter txt files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=5, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=2048, help="Maximum number of query points")
    parser.add_argument("--fine_tracking", action="store_true", default=True, help="Use fine tracking")
    return parser.parse_args()

def run_VGGT_depth_only(model, images, dtype, resolution=518):
    """只使用VGGT预测深度图，不预测相机参数"""
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # 只预测深度图，不预测相机参数
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return depth_map, depth_conf

def extract_timestamp_from_filename(filename):
    """从文件名中提取时间戳，支持多种格式"""
    # 移除扩展名
    basename = os.path.splitext(filename)[0]

    # 尝试提取数字作为时间戳
    import re
    numbers = re.findall(r'\d+', basename)
    if numbers:
        # 返回最长的数字串作为时间戳
        return max(numbers, key=len)
    else:
        # 如果没有数字，返回文件名本身
        return basename

def read_intrinsics_from_txt(intrinsics_dir, image_paths):
    """
    从txt文件中读取内参矩阵
    每个txt文件包含一个3x3的内参矩阵
    """
    intrinsics = []

    # 获取所有内参文件
    intrinsic_files = glob.glob(os.path.join(intrinsics_dir, "*.txt"))
    intrinsic_dict = {}

    # 读取所有内参文件并建立时间戳映射
    for intrinsic_file in intrinsic_files:
        timestamp = extract_timestamp_from_filename(os.path.basename(intrinsic_file))
        intrinsic_matrix = np.loadtxt(intrinsic_file)
        assert intrinsic_matrix.shape == (3, 3), f"Intrinsic matrix in {intrinsic_file} should be 3x3"
        intrinsic_dict[timestamp] = intrinsic_matrix

    # 为每张图像匹配对应的内参
    for image_path in image_paths:
        image_timestamp = extract_timestamp_from_filename(os.path.basename(image_path))

        if image_timestamp in intrinsic_dict:
            intrinsics.append(intrinsic_dict[image_timestamp])
        else:
            # 如果找不到对应的内参，使用第一个可用的内参
            print(f"Warning: No intrinsic found for {image_path}, using first available intrinsic")
            intrinsics.append(list(intrinsic_dict.values())[0])

    return np.array(intrinsics)

def read_extrinsics_from_txt(extrinsics_dir, image_paths):
    """
    从txt文件中读取外参矩阵
    每个txt文件包含一个4x4的姿态矩阵
    """
    extrinsics = []

    # 获取所有外参文件
    extrinsic_files = glob.glob(os.path.join(extrinsics_dir, "*.txt"))
    extrinsic_dict = {}

    # 读取所有外参文件并建立时间戳映射
    for extrinsic_file in extrinsic_files:
        timestamp = extract_timestamp_from_filename(os.path.basename(extrinsic_file))
        extrinsic_matrix = np.loadtxt(extrinsic_file)
        assert extrinsic_matrix.shape == (4, 4), f"Extrinsic matrix in {extrinsic_file} should be 4x4"
        extrinsic_dict[timestamp] = extrinsic_matrix

    # 为每张图像匹配对应的外参
    for image_path in image_paths:
        image_timestamp = extract_timestamp_from_filename(os.path.basename(image_path))

        if image_timestamp in extrinsic_dict:
            # 取前3行（旋转矩阵和平移向量），符合COLMAP格式
            extrinsics.append(extrinsic_dict[image_timestamp][:3, :])
        else:
            # 如果找不到对应的外参，使用第一个可用的外参
            print(f"Warning: No extrinsic found for {image_path}, using first available extrinsic")
            extrinsics.append(list(extrinsic_dict.values())[0][:3, :])

    return np.array(extrinsics)

def demo_fn(args):
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")

    # Load VGGT model
    model = VGGT()
    pretrained_dict = torch.load('/iag_ad_01/ad/yuanweizhong/ckpt/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9/model.pt')
    model.load_state_dict(pretrained_dict)
    model.eval()
    model = model.to(device)
    print("Model loaded")

    # Get image paths
    image_dir = os.path.join(args.scene_dir)
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")

    # 按文件名排序确保一致性
    image_path_list.sort()
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images
    vggt_fixed_resolution = 518
    img_load_resolution = 1024
    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # 读取真值相机内参和外参
    print("Reading intrinsic parameters...")
    gt_intrinsic = read_intrinsics_from_txt(args.intrinsics_dir, image_path_list)
    print(f"Loaded {len(gt_intrinsic)} intrinsic matrices")

    print("Reading extrinsic parameters...")
    gt_extrinsic = read_extrinsics_from_txt(args.extrinsics_dir, image_path_list)
    print(f"Loaded {len(gt_extrinsic)} extrinsic matrices")

    print("Loaded GT camera parameters and poses")

    # 只使用VGGT预测深度图
    depth_map, depth_conf = run_VGGT_depth_only(model, images, dtype, vggt_fixed_resolution)

    # 使用真值内外参生成3D点云
    points_3d = unproject_depth_map_to_point_map(depth_map, gt_extrinsic, gt_intrinsic)

    # VGGT+BA重建流程
    image_size = np.array([img_load_resolution, img_load_resolution])
    scale = img_load_resolution / vggt_fixed_resolution
    shared_camera = True  # 使用共享相机

    with torch.cuda.amp.autocast(dtype=dtype):
        # 预测轨迹
        pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
            images,
            conf=depth_conf,
            points_3d=points_3d,
            masks=None,
            max_query_pts=args.max_query_pts,
            query_frame_num=args.query_frame_num,
            keypoint_extractor="aliked+sp",
            fine_tracking=args.fine_tracking,
        )
        torch.cuda.empty_cache()

    # 缩放内参矩阵从518到1024
    gt_intrinsic_scaled = gt_intrinsic.copy()
    gt_intrinsic_scaled[:, :2, :] *= scale

    track_mask = pred_vis_scores > args.vis_thresh

    # 使用真值内外参进行COLMAP重建
    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
        points_3d,
        gt_extrinsic,          # 使用真值外参
        gt_intrinsic_scaled,   # 使用真值内参（已缩放）
        pred_tracks,
        image_size,
        masks=track_mask,
        max_reproj_error=args.max_reproj_error,
        shared_camera=shared_camera,
        camera_type=args.camera_type,
        points_rgb=points_rgb,
    )

    breakpoint()
    if reconstruction is None:
        raise ValueError("No reconstruction can be built with BA")

    # Bundle Adjustment
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)
    print("Bundle Adjustment completed")

    # 重命名和缩放相机参数
    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=img_load_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    # 保存重建结果
    print(f"Saving reconstruction to {args.scene_dir}/sparse/gt_camera_ba")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse/gt_camera_ba")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # 保存点云
    trimesh.PointCloud(points_3d, colors=points_rgb).export(
        os.path.join(args.scene_dir, "sparse/gt_camera_ba/points.ply")
    )

    print("Reconstruction completed successfully!")
    return True

def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction

if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)