import os
import shutil
import re

source_dir = "/iag_ad_01/ad/yuanweizhong/datasets/2025_07_05_17_38_13_AutoCollect_pilotGtRawParser/camera/center_camera_fov30"
dest_dir = "/iag_ad_01/ad/yuanweizhong/datasets/2025_07_05_17_38_13_AutoCollect_pilotGtRawParser/camera/center_camera_fov30_100"
num_files_to_copy = 100
extension = '.jpg'

os.makedirs(dest_dir, exist_ok=True)

all_entries = os.listdir(source_dir)

files = [
    f for f in all_entries 
    if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(extension)
]

def sort_key(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')

files.sort(key=sort_key)

files_to_copy = files[:num_files_to_copy]

for filename in files_to_copy:
    source_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(dest_dir, filename)
    shutil.copy2(source_path, dest_path)