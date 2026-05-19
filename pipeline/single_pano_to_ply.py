"""
Generate a 3D point cloud (.ply) from a single equirectangular 360° panorama.

Uses MiDaS depth estimation + spherical backprojection.
No COLMAP, no GPU, no training needed.

Usage:
    python single_pano_to_ply.py <panorama.jpg> <output.ply> [--max-points 1000000] [--resolution 2048]
"""

import sys
import os
import cv2
import numpy as np
import math
import struct


def load_midas():
    """Load MiDaS model for depth estimation (CPU)."""
    import torch
    print("Loading MiDaS depth model...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    return midas, transform


def estimate_depth(img_bgr, midas, transform):
    """Run MiDaS depth estimation. Returns inverse-depth map at image resolution."""
    import torch
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img_bgr.shape[:2],
            mode="bilinear", align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()


def equirect_to_3d_points(img_bgr, inv_depth, subsample=2):
    """Convert equirectangular image + depth to 3D point cloud."""
    h, w = img_bgr.shape[:2]

    inv_depth_clipped = np.clip(inv_depth, 1e-3, None)
    distance = 1.0 / inv_depth_clipped

    # Normalize: median distance = 4.0 meters (typical room)
    median_d = np.median(distance)
    if median_d > 0:
        distance = distance / median_d * 4.0

    distance = np.clip(distance, 0.1, 20.0)

    rows = np.arange(0, h, subsample)
    cols = np.arange(0, w, subsample)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # Equirectangular → spherical
    yaw = (col_grid.astype(np.float64) / w) * 2 * np.pi
    pitch = (0.5 - row_grid.astype(np.float64) / h) * np.pi

    d = distance[row_grid, col_grid]

    # Spherical → Cartesian (Y-up)
    x = d * np.cos(pitch) * np.sin(yaw)
    y = d * np.sin(pitch)
    z = d * np.cos(pitch) * np.cos(yaw)

    # Colors
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = img_rgb[row_grid, col_grid, 0]
    g = img_rgb[row_grid, col_grid, 1]
    b = img_rgb[row_grid, col_grid, 2]

    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    colors = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)

    # Filter extreme distances and top/bottom poles (often distorted in 360°)
    mask = (d.ravel() < 18.0) & (d.ravel() > 0.15)
    # Exclude top 10% and bottom 10% of image (ceiling/floor poles often distorted)
    row_frac = row_grid.ravel().astype(float) / h
    mask &= (row_frac > 0.08) & (row_frac < 0.92)

    points = points[mask]
    colors = colors[mask]

    return points, colors


def write_ply(filename, points, colors):
    """Write point cloud as binary PLY file."""
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
    with open(filename, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fff", points[i, 0], points[i, 1], points[i, 2]))
            f.write(struct.pack("<BBB", int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])))
    print(f"Wrote {n:,} points to {filename}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python single_pano_to_ply.py <panorama.jpg> <output.ply> [--max-points N] [--resolution W]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_ply = sys.argv[2]

    max_points = 800000
    process_width = 2048

    for i, arg in enumerate(sys.argv):
        if arg == "--max-points" and i + 1 < len(sys.argv):
            max_points = int(sys.argv[i + 1])
        if arg == "--resolution" and i + 1 < len(sys.argv):
            process_width = int(sys.argv[i + 1])

    # Load image
    print(f"Loading: {input_path}")
    img_full = cv2.imread(input_path)
    if img_full is None:
        print(f"Error: cannot read {input_path}")
        sys.exit(1)
    h_full, w_full = img_full.shape[:2]
    print(f"  Original size: {w_full}x{h_full}")

    # Resize for processing
    process_height = process_width // 2  # Keep 2:1 aspect ratio
    img = cv2.resize(img_full, (process_width, process_height), interpolation=cv2.INTER_AREA)
    print(f"  Processing at: {process_width}x{process_height}")

    # For color sampling, use a higher-res version
    color_width = min(w_full, 4096)
    color_height = color_width // 2
    img_color = cv2.resize(img_full, (color_width, color_height), interpolation=cv2.INTER_AREA)

    # Free full image memory
    del img_full

    # Load MiDaS
    midas, transform = load_midas()

    # Estimate depth
    print("\nEstimating depth...")
    inv_depth = estimate_depth(img, midas, transform)
    print(f"  Depth range (inverse): {inv_depth.min():.1f} - {inv_depth.max():.1f}")

    # Resize depth to match color image
    inv_depth_hr = cv2.resize(inv_depth, (color_width, color_height), interpolation=cv2.INTER_LINEAR)

    # Calculate subsample rate
    total_pixels = color_width * color_height * 0.84  # ~84% after pole removal
    subsample = max(1, int(math.sqrt(total_pixels / max_points)))
    print(f"\nGenerating point cloud (subsample={subsample})...")

    points, colors = equirect_to_3d_points(img_color, inv_depth_hr, subsample=subsample)
    print(f"  Generated: {len(points):,} points")

    # Random subsample if too many
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
        print(f"  Subsampled to: {max_points:,} points")

    # Write PLY
    write_ply(output_ply, points, colors)
    file_size = os.path.getsize(output_ply) / (1024 * 1024)
    print(f"\nDone! {output_ply} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
