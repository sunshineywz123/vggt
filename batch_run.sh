#!/bin/bash
# cd /mnt/afs_james/dataset/human_dataset
# 1. 切分数据
# python clip_humanvid.py
# cp /mnt/afs_james/dataset/human_dataset/humanvid/videos/videos.json /mnt/afs_james/dataset/human_dataset/humanvid/videos/videos_bak.json
# conda activate /mnt/afs/miniconda/envs/Frame2Scene/
# cd /mnt/afs/yuanweizhong/vggt/
# 设置要遍历的目录
directory="/mnt/afs_james/dataset/human_dataset/humanvid/videos/"


# # 使用 find 命令查找所有 .mp4 文件
# find "$directory" -name "*.mp4" -print0 | while IFS= read -r -d $'\0' file; do
#   # 获取文件名（不包含扩展名）
#   # 使用 ffmpeg 转换视频
#   echo "转换文件: $file"
#   time python inference.py --image_folder $file --mask_sky  || continue
# done


# 2. vggt处理数据
# 先收集所有.mp4文件到数组
mapfile -d '' files < <(find "$directory" -name "*.mp4" -print0)
total=${#files[@]}
half=$((total * 7   / 16))
# half=$((total * 7   / 8))
# 打印所有files
# for file in "${files[@]}"; do
#   echo "$file"
# done > files.txt
# 只处理后一半
for ((i=half; i<total; i++)); do
  file="${files[i]}"
  echo "转换文件: $file"
  time python inference.py --image_folder "$file" --mask_sky || continue
done

# 3. 生成最终的json文件
# cd /mnt/afs_james/dataset/human_dataset
# python process_humanvid_json.py
