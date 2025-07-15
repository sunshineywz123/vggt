# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
from IPython import embed
import open3d as o3d
import time
import open3d as o3d
import numpy as np

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

import cv2
import os
from tqdm import tqdm
import torch.nn.functional as F
import sys
import ipdb
from vggt.utils.load_fn import load_and_preprocess_video
from pathlib import Path
import json
def vis(
    pred_dict: dict,
    output_dir: str,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
    pad_masks: np.ndarray = None,
    ori_images: np.ndarray = None,
    start_index: int = 0,
    n_frames: int = 230,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    
    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # 3.3.1 若经过padding，将 depth 从
    if len(pad_masks) != 0:
        # 后续步骤同上
        valid_cols = np.where(pad_masks[0].astype(bool).any(axis=0))[0]
        depth_map = depth_map[:, :, valid_cols, :]
        depth_conf = depth_conf[:, :, valid_cols]

    frame_height = ori_images.shape[1]
    frame_width = ori_images.shape[2]
    # 3.4 深度图上采样到原始尺寸
    # depth_map 是 numpy 数组，需用 cv2.resize 逐帧上采样
    upsampled_depth = []
    for i in range(depth_map.shape[0]):
        upsampled = cv2.resize(
            depth_map[i, :, :, 0], 
            (frame_width, frame_height), 
            interpolation=cv2.INTER_LINEAR
        )
        upsampled_depth.append(upsampled[..., None])  # 保持最后一维
    depth_map = np.stack(upsampled_depth, axis=0)
    # 3.7 调整相机内参以适应新的图像尺寸
    ratio_x = frame_width / images[0, 0].shape[-1]
    ratio_y = frame_height / images[0, 0].shape[-2]
    intrinsic_frame = intrinsics_cam.copy()

    # print(f"before intrinsic_frame: {intrinsic_frame}\n")
    # import ipdb;ipdb.set_trace()
    # crop intrinsic if padded
    # if len(pad_masks) != 0:
    #     # 将相机内参矩阵中的主点x坐标设置为有效列数的一半,以适应裁剪后的图像尺寸
    #     intrinsic_frame[:,0,2] = len(valid_cols) // 2
    intrinsic_frame[:, 0, :] *= ratio_x
    intrinsic_frame[:, 1, :] *= ratio_y
    intrinsic_frame[:,0,2]=frame_width/2
    # print(f"intrinsic_frame: {intrinsic_frame}\n")
    
    intrinsics_cam=intrinsic_frame

    #save intrinsic and poses
    #intrinsics_cam.shape = (S, 3, 3) 存为N×9
    #extrinsics_cam.shape = (S, 3, 4) 存为N×12
    if start_index==0:
        np.savetxt(os.path.join(output_dir,'pred_intrinsics.txt'),intrinsics_cam.reshape(-1,9))
        np.savetxt(os.path.join(output_dir,'pred_traj.txt'),extrinsics_cam.reshape(-1,12))
    else:
        #续写'pred_traj.txt'
        with open(os.path.join(output_dir, 'pred_traj.txt'), 'ab') as f:
            np.savetxt(f, extrinsics_cam.reshape(-1, 12))
        with open(os.path.join(output_dir, 'pred_intrinsics.txt'), 'ab') as f:
            np.savetxt(f, intrinsics_cam.reshape(-1, 9))
    # save images

    # if not os.path.exists(os.path.join(output_dir,'crop_images')):
    #     os.makedirs(os.path.join(output_dir,'crop_images'),exist_ok=True)
    # for i in tqdm(range(images.shape[0]),'images'):
    #     cv2.imwrite(os.path.join(output_dir,f'crop_images/frame_{int(start_index+i):06d}.jpg'),images.transpose(0, 2, 3, 1)[i][:,:,::-1]*255)
    # import ipdb;ipdb.set_trace()
    # intrinsic = np.loadtxt(path + 'pred_intrinsics.txt')
    # poses = np.loadtxt(path + 'pred_traj.txt')

    # K = torch.from_numpy(intrinsic[:num_frames, :]).reshape(num_frames, 3, 3)
    # R_matrix = quat_to_matrix(np.concatenate([poses[:num_frames, 5:], poses[:num_frames, 4:5]], -1))
    # t = poses[:num_frames, 1:4]

    # 5. 保存深度图
    if not os.path.exists(os.path.join(output_dir, "depth")):
        os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    if len(depth_map.shape) == 2:  # 单帧情况
        depth = depth_map
        depth_uint16 = (depth*1000).astype(np.uint16)
        depth_png_path = os.path.join(output_dir, "depth/depth.png")
        cv2.imwrite(depth_png_path, depth_uint16)
    else:  # 多帧情况
        for i, depth in enumerate(depth_map):
            depth_uint16 = (depth*1000).astype(np.uint16)
            depth_png_path = os.path.join(output_dir, "depth", f"{int(start_index+i):04d}.png")
            cv2.imwrite(depth_png_path, depth_uint16)
    
    # import ipdb;ipdb.set_trace()
    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map
    # t_sky_1 = time.time()
    # # Apply sky segmentation if enabled
    # if mask_sky and image_folder is not None:
    #     conf = apply_sky_segmentation(conf, image_folder)
    # t_sky_2 = time.time()
    # print('mask sky time: ',t_sky_2 - t_sky_1)
    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    # 7.2 准备颜色数据
    # if len(pad_masks) != 0:
    #     # unpad colors
    #     # colors = images.transpose(0, 2, 3, 1)
    #     colors = ori_images
    #     colors = colors[:, :, valid_cols, :]
    # else:
    #     colors = ori_images
    colors = ori_images/255
        # colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape
    # import ipdb;ipdb.set_trace()

    for i in tqdm(range(world_points.shape[0]),'world_points'):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_points[i].reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(colors[i].reshape(-1, 3))
        save_path = os.path.join(output_dir,f'world_points_{i}.ply') # 'vggt_pcd_autocast.ply'
        o3d.io.write_point_cloud(save_path, pcd)

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)
    
    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    
    
    # save point_cloud 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_centered[init_conf_mask])
    pcd.colors = o3d.utility.Vector3dVector(colors_flat[init_conf_mask]/255)
    save_path = os.path.join(output_dir,'vggt_pcd.ply') # 'vggt_pcd_autocast.ply'
    o3d.io.write_point_cloud(save_path,pcd)

    return 1


# Helper functions for sky segmentation


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)
            #sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores
    # sky conf *= 0, else conf *= 1 
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")


def main(args,start_index=0,n_frames=230):
    """
    Main function for the VGGT demo with viser for 3D visualization.

    This function:
    1. Loads the VGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser

    Command-line arguments:
    --image_folder: Path to folder containing input images
    --use_point_map: Use point map instead of depth-based points
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --mask_sky: Apply sky segmentation to filter out sky points
    """
    t1 = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = VGGT()
    #_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    #model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.load_state_dict(torch.load("/mnt/afs/shendonghui/Task/vggt/model.pt"))

    model.eval()
    model = model.to(device)

    t2 = time.time()
    print('init model time: ',t2-t1)
    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    #args.image_folder是文件
    if os.path.isfile(args.image_folder):
        image_names =None
        print(f"Found video_path:{args.image_folder} ")
    else:
        image_names = glob.glob(os.path.join(args.image_folder, "*"))
        print(f"Found {len(image_names)} images")

    # 1. 加载并预处理输入数据
    # 根据输入类型（视频或图像列表）选择不同的加载方式
    if isinstance(image_names, list):
        images, ori_images, pad_masks = load_and_preprocess_images(image_names,mode='pad')
    else:
        images, ori_images, pad_masks = load_and_preprocess_video(args.image_folder,start_index,n_frames)
    images=images.to(device)
    print(f"Preprocessed images shape: {images.shape}")
    t3 = time.time()
    print('process img time: ',t3-t2)

    print("Running inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = model(images)
            
    t4 = time.time()
    print('inference time: ',t4-t3)
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    print("Starting viser visualization...")

    if os.path.isfile(args.image_folder):
        output_dir = args.image_folder.replace(Path(args.image_folder).parent.name,'')[:-4]
    else:
        output_dir = args.image_folder.replace(os.path.basename(args.image_folder),'')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    vis(
        predictions,
        output_dir=output_dir,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
        pad_masks=pad_masks,
        ori_images=ori_images,
        start_index=start_index,
        n_frames=n_frames
    )
    t5 = time.time()
    print('post-process time: ',t5 - t4)
    print("Visualization complete")


if __name__ == "__main__":
    if 0:
        args = parser.parse_args()
        if os.path.isfile(args.image_folder):
            image_names =None
            print(f"Found video_path:{args.image_folder} ")
        else:
            image_names = glob.glob(os.path.join(args.image_folder, "*"))
            print(f"Found {len(image_names)} images")
        video_path = args.image_folder
        #根据视频json文件，获取视频的开始帧索引和结束帧索引
        video_json_path = str(Path(video_path).parent)+  "/videos_bak.json"
        if os.path.exists(video_json_path):
            with open(video_json_path, 'r') as f:
                video_data = json.load(f)
            for data in tqdm(video_data,desc="Processing video data"):
                if data["video_path"] == video_path:
                    start_index = data["start_index"]
                    n_frames = data["n_frames"]
                    print(f"video_path: {video_path}, start_index: {start_index}, n_frames: {n_frames}\n")
                    main(args,start_index,n_frames)
    else:
        try:
            args = parser.parse_args()
            main(args)
        except:
            type, value, traceback = sys.exc_info()
            ipdb.post_mortem(traceback)


# conda activate /mnt/afs/miniconda/envs/Frame2Scene/
# cd /mnt/afs/yuanweizhong/vggt/
# python inference.py --image_folder examples/gugong/images --mask_sky
# time python inference.py --image_folder examples/gugong_tmp/images --mask_sky
# time python inference.py --image_folder /mnt/afs_james/dataset/human_dataset/humanvid/videos/853976-hd_1920_1080_25fps.mp4 --mask_sky
# time python inference.py --image_folder /mnt/afs_james/dataset/human_dataset/humanvid/videos/3197663-hd_1080_1920_25fps.mp4 --mask_sky 


# /mnt/afs/xieweijian/ads-cli --threads 50 cp /mnt/afs/yuanweizhong/vggt/1.txt  s3://AD8B978DE05D4643A847B4F110C08571:685C3EA2B40C4EC09EDDF62C4185E9FD@aigc_3dv1.10.118.0.160/test/1.txt
