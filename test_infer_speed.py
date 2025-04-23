import torch
import glob
import os
import time
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT()
model.load_state_dict(torch.load("/mnt/afs/shendonghui/Task/vggt/model.pt"))

model.eval()  # Ensure model is in evaluation mode
model = model.to(device)

# Load and preprocess example images (replace with your own image paths)
frame_num = 17
image_names = glob.glob(os.path.join("/mnt/afs/shendonghui/Task/vggt/examples/test_0314/IMG_8778_result/pano_results/render_video", "*"))

image_names = image_names[:frame_num]
images = load_and_preprocess_images(image_names).to(device, non_blocking=True)
images = images[:, :, :336, :518].clone()    # match the setting of Table


with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        
        ########################## 
        # Multiple warm-up iterations for better GPU optimization
        #for _ in range(3):
        #    _, _ = model.aggregator(images)
        #torch.cuda.synchronize()  # Ensure warm-up is complete
        #########################
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        # Benchmark with multiple runs for more accurate timing
        # for rough estimate, just run once
        num_runs = 1
        start_time.record()
        for _ in range(num_runs):
            t0 = time.time()
            aggregated_tokens_list, ps_idx = model.aggregator(images)
            t1 = time.time()
            predictions = model(images)
            t2 = time.time()

        end_time.record()
        torch.cuda.synchronize()
        print('agg_only time:',t1-t0)
        print('predict time: ',t2-t1)
        
        runtime_ms = start_time.elapsed_time(end_time) / num_runs  # Average time per run
        runtime_sec = runtime_ms / 1000  # Convert ms to seconds
        print(f"Time taken (avg of {num_runs} runs): {runtime_ms:.2f} ms ({runtime_sec:.4f} s)")
