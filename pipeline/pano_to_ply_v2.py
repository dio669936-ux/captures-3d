"""
Generate 3D point cloud (.ply) from equirectangular 360° panorama.
Uses Depth Anything V2 for high-quality depth estimation.

Key improvements over v1:
- Better depth model (Depth Anything V2 vs MiDaS small)
- Proper equirectangular → 3D spherical projection
- Hemisphere-aware processing (avoid pole distortion)
- Denser point sampling for solid appearance

Usage:
    python pano_to_ply_v2.py <panorama.jpg> <output.ply> [--max-points 1000000]
"""

import sys
import os
import cv2
import numpy as np
import struct
import math


def load_depth_model():
    """Load Depth Anything V2 Small model."""
    import torch
    from transformers import pipeline

    print("Loading Depth Anything V2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    depth_pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    return depth_pipe


def estimate_depth_dav2(img_bgr, depth_pipe):
    """Estimate depth using Depth Anything V2. Returns metric-like depth map."""
    from PIL import Image

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    result = depth_pipe(pil_img)
    depth_pil = result["depth"]
    # Get raw predicted depth (the PIL image is 0-255 uint8, use it as float)
    depth_map = np.array(depth_pil).astype(np.float32)

    # Also try to get the raw prediction tensor if available
    if "predicted_depth" in result:
        raw = result["predicted_depth"]
        if hasattr(raw, 'numpy'):
            depth_map = raw.squeeze().numpy().astype(np.float32)
        else:
            depth_map = np.array(raw).astype(np.float32)

    # Resize to match input
    if depth_map.shape[:2] != img_bgr.shape[:2]:
        depth_map = cv2.resize(depth_map, (img_bgr.shape[1], img_bgr.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

    return depth_map


def pano_to_pointcloud(img_bgr, depth_map, subsample=2):
    """Convert equirectangular panorama + depth → 3D points.

    Proper spherical projection:
    - Each pixel (u,v) maps to direction (theta, phi) on unit sphere
    - theta = yaw (0 to 2π), phi = pitch (-π/2 to π/2)
    - With depth d, 3D position = d * direction_vector
    """
    h, w = img_bgr.shape[:2]

    # Depth Anything V2 returns disparity-like map (higher = closer)
    # Convert to distance: distance ∝ 1/disparity
    depth = depth_map.astype(np.float64)
    depth_clipped = np.clip(depth, 1.0, None)  # Avoid division by zero

    # Inverse depth → distance
    distance = 1.0 / depth_clipped

    # Normalize so median distance = 3.5m (typical indoor room)
    median_d = np.median(distance)
    if median_d > 0:
        distance = distance / median_d * 3.5

    # Clamp to reasonable indoor range
    distance = np.clip(distance, 0.3, 10.0)

    # Create subsampled pixel grid
    rows = np.arange(0, h, subsample)
    cols = np.arange(0, w, subsample)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # Pixel → spherical coordinates
    # theta: 0→2π (left→right of equirect)
    # phi: π/2→-π/2 (top→bottom = up→down)
    theta = (col_grid.astype(np.float64) / w) * 2 * np.pi
    phi = (0.5 - row_grid.astype(np.float64) / h) * np.pi

    # Sample depth
    d = distance[row_grid, col_grid]

    # Spherical → Cartesian (right-hand, Y-up)
    x = d * np.cos(phi) * np.sin(theta)
    y = d * np.sin(phi)
    z = d * np.cos(phi) * np.cos(theta)

    # Sample colors from original image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = img_rgb[row_grid, col_grid, 0]
    g = img_rgb[row_grid, col_grid, 1]
    b = img_rgb[row_grid, col_grid, 2]

    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    colors = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)

    # Filter out poles (top/bottom 8%) - heavily distorted in equirectangular
    row_frac = row_grid.ravel().astype(float) / h
    mask = (row_frac > 0.08) & (row_frac < 0.92)

    # Filter extreme distances
    d_flat = d.ravel()
    mask &= (d_flat > 0.3) & (d_flat < 8.0)

    return points[mask], colors[mask]


def write_ply_binary(filename, points, colors):
    """Write binary PLY file."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    # Pack all data at once for speed
    data = np.empty(n, dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
    ])
    data['x'] = points[:, 0].astype(np.float32)
    data['y'] = points[:, 1].astype(np.float32)
    data['z'] = points[:, 2].astype(np.float32)
    data['r'] = colors[:, 0].astype(np.uint8)
    data['g'] = colors[:, 1].astype(np.uint8)
    data['b'] = colors[:, 2].astype(np.uint8)

    with open(filename, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    print(f"Wrote {n:,} points to {filename}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python pano_to_ply_v2.py <panorama.jpg> <output.ply> [--max-points N]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_ply = sys.argv[2]
    max_points = 1000000

    for i, arg in enumerate(sys.argv):
        if arg == "--max-points" and i + 1 < len(sys.argv):
            max_points = int(sys.argv[i + 1])

    # Load image
    print(f"Loading: {input_path}")
    img_full = cv2.imread(input_path)
    if img_full is None:
        print(f"Error: cannot read {input_path}")
        sys.exit(1)

    h_full, w_full = img_full.shape[:2]
    print(f"  Size: {w_full}x{h_full}")
    aspect = w_full / h_full
    if abs(aspect - 2.0) > 0.1:
        print(f"  Warning: aspect ratio {aspect:.2f} ≠ 2.0, may not be equirectangular")

    # Resize for depth estimation (Depth Anything works well at 518px height)
    depth_w = 1024
    depth_h = 512
    img_depth = cv2.resize(img_full, (depth_w, depth_h), interpolation=cv2.INTER_AREA)

    # Higher res for color sampling
    color_w = min(w_full, 4096)
    color_h = color_w // 2
    img_color = cv2.resize(img_full, (color_w, color_h), interpolation=cv2.INTER_AREA)
    del img_full

    # Load depth model
    depth_pipe = load_depth_model()

    # Estimate depth
    print("\nEstimating depth with Depth Anything V2...")
    depth_map = estimate_depth_dav2(img_depth, depth_pipe)
    print(f"  Depth range: {depth_map.min():.1f} - {depth_map.max():.1f}")

    # Upscale depth to color resolution
    depth_hr = cv2.resize(depth_map, (color_w, color_h), interpolation=cv2.INTER_LINEAR)

    # Calculate subsample rate
    usable_pixels = color_w * color_h * 0.84  # 84% after pole removal
    subsample = max(1, int(math.sqrt(usable_pixels / max_points)))
    print(f"\nGenerating point cloud (subsample={subsample}, target={max_points:,})...")

    points, colors = pano_to_pointcloud(img_color, depth_hr, subsample=subsample)
    print(f"  Raw points: {len(points):,}")

    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
        print(f"  Subsampled to: {max_points:,}")

    # Write output
    write_ply_binary(output_ply, points, colors)
    file_size = os.path.getsize(output_ply) / (1024 * 1024)
    print(f"\nDone! {output_ply} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
