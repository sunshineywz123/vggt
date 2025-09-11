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
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import open3d as o3d
import roma
from scipy.spatial.distance import pdist
import ipdb
import sys
if 1:
    import ptvsd
    ptvsd.enable_attach(address=('0.0.0.0', 5691), redirect_output=True)

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
def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf

def vis_o3d_pcd_1(cloud,color = [1,1,1]):    
    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries([pcd])

def vis_o3d_pcd_2(cloud1,cloud2,color1 = [1,1,1],color2 = [1,1,1]):
    
    pcd1=o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(cloud1)
    pcd1.paint_uniform_color(color1)
    pcd2=o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(cloud2)
    pcd2.paint_uniform_color(color2)

    o3d.visualization.draw_geometries([pcd1,pcd2])
    

def visualize_camera_poses(poses_tensor1, poses_tensor2, size=0.1, show_coordinate_frame=True):
    
    # 创建Open3D可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 转换为NumPy数组
    poses1 = poses_tensor1.numpy()
    poses2 = poses_tensor2.numpy()
    
    # 添加第一个相机的位姿
    for i, pose in enumerate(poses1):
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        camera.transform(pose)
        camera.paint_uniform_color([1, 0, 0])  # 红色表示第一个相机
        vis.add_geometry(camera)
    
    # 添加第二个相机的位姿
    for i, pose in enumerate(poses2):
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        camera.transform(pose)
        camera.paint_uniform_color([0, 0, 1])  # 蓝色表示第二个相机
        vis.add_geometry(camera)
    
    # 可选：添加全局坐标系
    # if show_coordinate_frame:
    #     world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    #     vis.add_geometry(world_frame)
    
    vis.run()
    vis.destroy_window()

def visualize_camera_poses1(poses_tensor1, size=0.1, show_coordinate_frame=True):
    
    # 创建Open3D可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 转换为NumPy数组
    poses1 = poses_tensor1.numpy()
    
    
    # 添加第一个相机的位姿
    for i, pose in enumerate(poses1):
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        camera.transform(pose)
        camera.paint_uniform_color([1, i/3, 0])  # 红色表示第一个相机
        vis.add_geometry(camera)
    

    
    # 可选：添加全局坐标系
    # if show_coordinate_frame:
    #     world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    #     vis.add_geometry(world_frame)
    
    vis.run()
    vis.destroy_window()

def get_med_dist_between_poses(poses):
    
    return np.median(pdist([p[:3, 3].numpy() for p in poses]))

def align_multiple_poses(src_poses, target_poses):
    N = len(src_poses)
    assert src_poses.shape == target_poses.shape == (N, 4, 4)

    def center_and_z(poses):
        eps = get_med_dist_between_poses(poses) / 10
        return torch.cat((poses[:, :3, 3], poses[:, :3, 3] + eps*poses[:, :3, 2]))
    R, T, s = roma.rigid_points_registration(center_and_z(src_poses), center_and_z(target_poses), compute_scaling=True)
    # import open3d as o3d

    return s, R, T

def signed_log1p(x):
    sign = torch.sign(x)
    return sign * torch.log1p(torch.abs(x))

def rotate_points_with_srt(source_points, s, R, t):

    # 1. 处理缩放（先应用缩放）
    if isinstance(s, (float, int)):
        # 标量缩放：所有维度等比例缩放
        scaled_points = source_points * s
    else:
        # 各维度独立缩放（假设s是3维向量）
        scaled_points = source_points * s.reshape(1, -1)  # (N,3) * (1,3)
    
    # 2. 应用旋转变换（矩阵乘法）
    rotated_points = torch.matmul(scaled_points, R.T)  # (N,3) @ (3,3) = (N,3)
    
    # 3. 应用平移变换
    rotated_points = rotated_points + t.reshape(1, -1)  # (N,3) + (1,3)
    
    return rotated_points

def rotate_cameras_with_srt(poses, s,R,t):


    
    # 创建4×4的SRT变换矩阵
    srt_matrix = torch.eye(4, device=poses.device)
    srt_matrix[:3, :3] = s * R  # 缩放和旋转
    srt_matrix[:3, 3] = t       # 平移
    
    # 对每个相机位姿应用SRT变换
    # 注意：相机位姿通常是从世界坐标系到相机坐标系的变换
    # 因此，我们需要右乘SRT变换矩阵
    transformed_poses = srt_matrix @ poses
    
    return transformed_poses


def read_colmap_camera(colmap_camera_path):
    with open(colmap_camera_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        line = lines[3:][0]
        image_W,image_H,focal_x,focal_y,cx,cy = [float(i) for i in line.split()[2:]]

    return image_W,image_H,focal_x,focal_y,cx,cy

def read_colmap_gt(colmap_images_path):
    # colmap_images_path = "sparse_DTU/set_23_24_33/scan40/sparse/0/images.txt"
    # colmap_images_path = "sparse_DTU/wo_pose/scan24/sparse/0/images.txt"
    with open(colmap_images_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = lines[4:]
        poses = torch.zeros(int(len(lines)/2),7)
        for idx,line in enumerate(lines):
            if idx % 2 == 0:
                line_splited = line.split()
                image_idx = int(line_splited[-1][:4])
                pose = torch.tensor([float(line_splited[2]),float(line_splited[3]),float(line_splited[4]),float(line_splited[1]),float(line_splited[5]),float(line_splited[6]),float(line_splited[7])])
                poses[image_idx] = pose

    poses_R = []
    for pose in poses:
        q_x, q_y, q_z,q_w,t_x,t_y,t_z = pose

        R = torch.eye(4)
        R_3 = torch.tensor([
            [1 - 2 * q_y ** 2 - 2 * q_z ** 2, 2 * (q_x * q_y - q_w * q_z), 2 * (q_x * q_z + q_w * q_y)],
            [2 * (q_x * q_y + q_w * q_z), 1 - 2 * q_x ** 2 - 2 * q_z ** 2, 2 * (q_y * q_z - q_w * q_x)],
            [2 * (q_x * q_z - q_w * q_y), 2 * (q_y * q_z + q_w * q_x), 1 - 2 * q_x ** 2 - 2 * q_y ** 2]
            ])
        t = torch.tensor([t_x,t_y,t_z])

        R[:3, :3] = R_3
        R[:3, 3] = t

        poses_R.append(R.inverse())
    return torch.stack(poses_R,dim = 0)[:,:3,:]
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

    gt_pose = gt_extrinsic
    last_row = torch.tensor([0, 0, 0, 1]).expand(gt_pose.shape[0], 1, 4)
    gt_pose44 = torch.cat([torch.tensor(gt_pose), last_row], dim=1)
    print("Loaded GT camera parameters and poses")
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    extrinsic44 = torch.cat([torch.tensor(extrinsic), last_row], dim=1)
    extrinsic44 = torch.inverse(extrinsic44)
    s, R, T = align_multiple_poses(extrinsic44.double(),gt_pose44.double())
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    # conf_thres_value = args.conf_thres_value
    # conf_mask = depth_conf > conf_thres_value
    # conf_mask = randomly_limit_trues(conf_mask,max_points_for_colmap)
    
    points_3d_trans2gt = rotate_points_with_srt(torch.tensor(points_3d).float(),s,R,T).numpy()

    # VGGT+BA重建流程
    image_size = np.array([img_load_resolution, img_load_resolution])
    scale = img_load_resolution / vggt_fixed_resolution
    shared_camera = True  # 使用共享相机

    with torch.cuda.amp.autocast(dtype=dtype):
        # 预测轨迹
        pred_tracks, pred_vis_scores, pred_confs, points_3d_trans2gt, points_rgb = predict_tracks(
            images,
            conf=depth_conf,
            points_3d=points_3d_trans2gt,
            masks=None,
            max_query_pts=args.max_query_pts,
            query_frame_num=args.query_frame_num,
            keypoint_extractor="aliked+sp",
            fine_tracking=args.fine_tracking,
        )
        torch.cuda.empty_cache()

    # breakpoint()
    # 缩放内参矩阵从518到1024
    gt_intrinsic_scaled = gt_intrinsic.copy()
    gt_intrinsic_scaled[:, :2, :] *= scale

    track_mask = pred_vis_scores > args.vis_thresh

    # import ipdb;ipdb.set_trace()
    # 使用真值内外参进行COLMAP重建
    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
        points_3d_trans2gt,
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
    trimesh.PointCloud(points_3d_trans2gt, colors=points_rgb).export(
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
        if 1:
            demo_fn(args)
        else:
            try:
                demo_fn(args)
            except:
                _,_,tb = sys.exc_info()
                ipdb.post_mortem(tb)