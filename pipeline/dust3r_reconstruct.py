"""DUSt3R room-scale 3D reconstruction from panorama perspective crops."""
import sys, os, time, math
import numpy as np
sys.path.insert(0, '/home/ubuntu/dust3r')
os.chdir('/home/ubuntu/dust3r')

import torch
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

IMAGE_DIR = '/home/ubuntu/gs_pipeline/images/'
OUTPUT_PLY = '/home/ubuntu/gs_pipeline/dust3r_room.ply'
DEVICE = 'cpu'
IMAGE_SIZE = 512

# Select images from connected rooms
SELECTED_ROOMS = ['01-living-room-d', '02-living-room-c', '03-living-room-b']
VIEWS = ['front', 'right', 'left', 'back']

image_files = []
for room in SELECTED_ROOMS:
    for view in VIEWS:
        fname = f"{IMAGE_DIR}/{room}_{view}.jpg"
        if os.path.exists(fname):
            image_files.append(fname)

print(f"Selected {len(image_files)} images from {len(SELECTED_ROOMS)} rooms")

# Load model
print("\nLoading DUSt3R model...")
t0 = time.time()
model = AsymmetricCroCo3DStereo.from_pretrained('naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt')
model = model.to(DEVICE)
model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s")

# Load images
print("\nLoading images...")
images = load_images(image_files, size=IMAGE_SIZE)
n = len(images)
print(f"Loaded {n} images")

# Create pairs using index tuples to avoid tensor comparison
pair_indices = set()
# Sequential pairs
for i in range(n):
    for j in range(i+1, min(i+4, n)):
        pair_indices.add((i, j))
# Cross-room same-view pairs
room_size = len(VIEWS)
for v in range(room_size):
    for r1 in range(len(SELECTED_ROOMS)):
        for r2 in range(r1+1, len(SELECTED_ROOMS)):
            i = r1 * room_size + v
            j = r2 * room_size + v
            if i < n and j < n:
                pair_indices.add((i, j))

pairs = [(images[i], images[j]) for i, j in sorted(pair_indices)]
print(f"\nCreated {len(pairs)} image pairs")
print(f"Estimated time: {len(pairs) * 57 / 60:.0f} minutes on CPU")

# Run inference
print("\nRunning DUSt3R inference...")
t1 = time.time()
with torch.no_grad():
    output = inference(pairs, model, DEVICE, batch_size=1)
print(f"Inference done in {time.time()-t1:.1f}s ({(time.time()-t1)/60:.1f} min)")

# Global alignment
print("\nRunning global alignment...")
scene = global_aligner(output, device=DEVICE, mode=GlobalAlignerMode.PointCloudOptimizer)
loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
print(f"Global alignment done, final loss: {loss}")

# Extract 3D points and colors
print("\nExtracting 3D reconstruction...")
pts3d = scene.get_pts3d()
imgs = scene.imgs
masks = scene.get_masks()

all_points = []
all_colors = []
for i in range(len(pts3d)):
    pts = pts3d[i].detach().cpu().numpy()
    mask = masks[i].detach().cpu().numpy()
    img = imgs[i]
    h, w = pts.shape[:2]
    pts_flat = pts.reshape(-1, 3)
    mask_flat = mask.reshape(-1)
    colors_flat = (img.reshape(-1, 3) * 255).astype(np.uint8)
    valid = mask_flat > 0.5
    pts_valid = pts_flat[valid]
    colors_valid = colors_flat[valid]
    step = 2
    all_points.append(pts_valid[::step])
    all_colors.append(colors_valid[::step])
    print(f"  Image {i+1}/{len(pts3d)}: {pts_valid.shape[0]:,} valid -> {len(pts_valid[::step]):,} subsampled")

all_points = np.concatenate(all_points, axis=0).astype(np.float32)
all_colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
print(f"\nTotal: {len(all_points):,} 3D points")

# Write Gaussian Splatting PLY
print("\nWriting Gaussian Splatting PLY...")
import struct

SH_C0 = 0.28209479177387814
n_pts = len(all_points)
f_dc = (all_colors.astype(np.float32) / 255.0 - 0.5) / SH_C0
scale_val = math.log(0.008)
opacity_val = 5.0

header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n_pts}\n"
header += "property float x\nproperty float y\nproperty float z\n"
header += "property float nx\nproperty float ny\nproperty float nz\n"
header += "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
for i in range(45):
    header += f"property float f_rest_{i}\n"
header += "property float opacity\n"
header += "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
header += "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
header += "end_header\n"

# Use numpy for fast binary write
zero3 = np.zeros(3, dtype=np.float32)
zero45 = np.zeros(45, dtype=np.float32)
scales = np.full(3, scale_val, dtype=np.float32)
rot = np.array([1,0,0,0], dtype=np.float32)
opacity = np.array([opacity_val], dtype=np.float32)

with open(OUTPUT_PLY, 'wb') as f:
    f.write(header.encode())
    for i in range(n_pts):
        f.write(all_points[i].tobytes())  # xyz
        f.write(zero3.tobytes())  # normals
        f.write(f_dc[i].astype(np.float32).tobytes())  # f_dc
        f.write(zero45.tobytes())  # f_rest
        f.write(opacity.tobytes())  # opacity
        f.write(scales.tobytes())  # scale
        f.write(rot.tobytes())  # rotation

file_size = os.path.getsize(OUTPUT_PLY)
print(f"\nOutput: {OUTPUT_PLY}")
print(f"Size: {file_size/1048576:.1f} MB")
print(f"Gaussians: {n_pts:,}")
print("Done!")
