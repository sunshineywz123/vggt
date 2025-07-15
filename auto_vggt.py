import os
import shutil
import re
import subprocess

# 数据集根目录
datasets_root = "/iag_ad_01/ad/yuanweizhong/datasets/senseauto"

# 定义生成 center_camera_fov30_100 的函数
def generate_center_camera_fov30_100(source_dir, dest_dir, num_files_to_copy=100, extension='.jpg'):
    os.makedirs(dest_dir, exist_ok=True)
    all_entries = os.listdir(source_dir)

    # 筛选出符合条件的文件
    files = [
        f for f in all_entries 
        if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(extension)
    ]

    # 按文件名中的数字排序
    def sort_key(filename):
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return float('inf')

    files.sort(key=sort_key)

    # 复制前 num_files_to_copy 个文件
    files_to_copy = files[:num_files_to_copy]
    for filename in files_to_copy:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(source_path, dest_path)

# 遍历数据集目录
for dataset in os.listdir(datasets_root):
    dataset_path = os.path.join(datasets_root, dataset)
    target_folder = os.path.join(dataset_path, "camera", "center_camera_fov30_100")
    source_folder = os.path.join(dataset_path, "camera", "center_camera_fov30")
    
    # 检查目标目录是否存在
    if os.path.isdir(target_folder):
        # 如果目标目录存在，运行推理脚本
        command = f'python inference.py --image_folder "{target_folder}" --mask_sky'
        print(f"启动命令: {command}")
        subprocess.run(command, shell=True)
    else:
        # 如果目标目录不存在，生成 center_camera_fov30_100
        if os.path.isdir(source_folder):
            print(f"{target_folder} 不存在，正在")
            generate_center_camera_fov30_100(source_folder, target_folder)
            print(f"{target_folder} 已生成，开始运行")
            command = f'python inference.py --image_folder "{target_folder}" --mask_sky'
            subprocess.run(command, shell=True)
        else:
            print(f"跳过: {source_folder} 不存在，无法生成 {target_folder}")