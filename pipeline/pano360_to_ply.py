#!/usr/bin/env python3
"""
Improved 360° Panorama → 3D Point Cloud Pipeline
Mimics DreamScene360's approach: tangent crop depth estimation + global alignment

Instead of running depth estimation on the distorted equirectangular image directly,
this pipeline:
1. Projects equirectangular into multiple overlapping perspective (tangent) crops
2. Runs depth estimation on each crop (perspective images = what models are trained on)
3. Blends/aligns depth maps back to equirectangular space
4. Backprojects with proper spherical geometry → 3D point cloud
"""

import sys
import os
import math
import argparse
import numpy as np
import cv2
from pathlib import Path

def equirect_to_perspective(img, fov_deg, theta_deg, phi_deg, out_h, out_w):
    """Extract a perspective crop from equirectangular image.
    
    Args:
        img: equirectangular image (H, W, 3)
        fov_deg: field of view in degrees
        theta_deg: horizontal angle (yaw) in degrees, 0 = center
        phi_deg: vertical angle (pitch) in degrees, 0 = horizon
        out_h, out_w: output image size
    Returns:
        perspective crop (out_h, out_w, 3)
    """
    h, w = img.shape[:2]
    fov = np.radians(fov_deg)
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    # Intrinsic matrix
    f = out_w / (2 * np.tan(fov / 2))
    cx, cy = out_w / 2, out_h / 2

    # Create pixel grid
    u = np.arange(out_w) - cx
    v = np.arange(out_h) - cy
    u, v = np.meshgrid(u, v)

    # Direction vectors in camera space
    x = u
    y = v
    z = np.full_like(u, f, dtype=np.float64)

    # Normalize
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/norm, y/norm, z/norm

    # Rotation matrices (yaw then pitch)
    # Yaw (around Y axis)
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    # Pitch (around X axis)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    R = Ry @ Rx

    # Stack and rotate
    dirs = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    dirs = dirs @ R.T  # rotate

    # Convert to spherical coordinates
    lon = np.arctan2(dirs[..., 0], dirs[..., 2])  # [-pi, pi]
    lat = np.arcsin(np.clip(dirs[..., 1], -1, 1))  # [-pi/2, pi/2]

    # Map to equirectangular pixel coordinates
    map_x = ((lon / np.pi + 1) / 2 * w).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * h).astype(np.float32)

    # Wrap horizontally
    map_x = map_x % w

    # Sample
    crop = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return crop


def perspective_depth_to_equirect(depth_crop, fov_deg, theta_deg, phi_deg, crop_h, crop_w, eq_h, eq_w):
    """Map a perspective depth crop back to equirectangular coordinates.
    
    Returns:
        eq_depth_contrib: (eq_h, eq_w) depth values mapped back
        eq_weight: (eq_h, eq_w) weight mask for blending
    """
    fov = np.radians(fov_deg)
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    f = crop_w / (2 * np.tan(fov / 2))
    cx, cy = crop_w / 2, crop_h / 2

    # For each equirectangular pixel, check if it falls in this crop's FOV
    eq_u = np.arange(eq_w)
    eq_v = np.arange(eq_h)
    eq_u, eq_v = np.meshgrid(eq_u, eq_v)

    # Equirectangular to spherical
    lon = (eq_u / eq_w * 2 - 1) * np.pi
    lat = (0.5 - eq_v / eq_h) * np.pi

    # Spherical to 3D direction
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    # Inverse rotation
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    R = Ry @ Rx
    R_inv = R.T

    dirs = np.stack([x, y, z], axis=-1)
    dirs_cam = dirs @ R_inv.T

    # Project to crop pixel coordinates
    z_cam = dirs_cam[..., 2]
    valid = z_cam > 0.01

    px = (dirs_cam[..., 0] / z_cam * f + cx)
    py = (dirs_cam[..., 1] / z_cam * f + cy)

    # Check bounds
    margin = 2
    valid &= (px >= margin) & (px < crop_w - margin)
    valid &= (py >= margin) & (py < crop_h - margin)

    eq_depth = np.zeros((eq_h, eq_w), dtype=np.float32)
    eq_weight = np.zeros((eq_h, eq_w), dtype=np.float32)

    # Sample depth from crop
    px_int = np.clip(px.astype(int), 0, crop_w - 1)
    py_int = np.clip(py.astype(int), 0, crop_h - 1)

    eq_depth[valid] = depth_crop[py_int[valid], px_int[valid]]

    # Weight: higher in center of crop, lower at edges (for smooth blending)
    dist_from_center = np.sqrt((px - cx)**2 + (py - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    weight = np.clip(1.0 - dist_from_center / max_dist, 0, 1) ** 2
    eq_weight[valid] = weight[valid]

    return eq_depth, eq_weight


def run_depth_estimation(images, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
    """Run depth estimation on a list of images using HuggingFace pipeline."""
    from transformers import pipeline as hf_pipeline
    import torch

    print(f"Loading depth model: {model_name}")
    pipe = hf_pipeline(
        "depth-estimation",
        model=model_name,
        device="cpu",
        torch_dtype=torch.float32
    )

    depths = []
    for i, img in enumerate(images):
        print(f"  Estimating depth for crop {i+1}/{len(images)}...", end='\r')
        # Convert BGR to RGB PIL
        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = pipe(pil_img)
        depth = np.array(result["depth"])
        # Normalize to 0-1
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        depths.append(depth.astype(np.float32))
    print()
    return depths


def backproject_equirect_to_3d(img, depth, num_points=500000, depth_scale=10.0):
    """Backproject equirectangular image + depth to 3D colored point cloud.
    
    Uses proper spherical geometry with depth-dependent radius.
    Improved: log-scale depth, outlier removal, polar exclusion.
    """
    h, w = img.shape[:2]

    # Create coordinate grids
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)

    # Convert to spherical coordinates
    lon = (u / w * 2 - 1) * np.pi  # [-pi, pi]
    lat = (0.5 - v / h) * np.pi   # [pi/2, -pi/2]

    # Exclude extreme polar regions (top/bottom ~15 degrees) - usually ceiling/floor artifacts
    polar_mask = np.abs(lat) < np.radians(75)

    # Depth to radius using log scale for better dynamic range
    # Monocular depth: higher value = closer object
    # Use log transform for more natural depth distribution
    depth_clipped = np.clip(depth, 0.01, 0.99)
    # Log-disparity transform: maps depth to more perceptually uniform radius
    radius = np.exp(-depth_clipped * 3.0) * depth_scale

    # Clip extreme radii
    r_median = np.median(radius)
    radius = np.clip(radius, r_median * 0.1, r_median * 3.0)

    # Spherical to Cartesian
    x = radius * np.cos(lat) * np.sin(lon)
    y = radius * np.sin(lat)
    z = radius * np.cos(lat) * np.cos(lon)

    # Flatten and apply polar mask
    mask = polar_mask.ravel()
    points = np.stack([x.ravel()[mask], y.ravel()[mask], z.ravel()[mask]], axis=1)
    colors_flat = img.reshape(-1, 3)[mask][:, ::-1] / 255.0  # BGR → RGB

    # Remove outliers using statistical distance from median
    center = np.median(points, axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    dist_median = np.median(dists)
    dist_std = np.std(dists)
    inlier_mask = dists < (dist_median + 2.5 * dist_std)
    points = points[inlier_mask]
    colors_flat = colors_flat[inlier_mask]

    # Subsample if needed
    if len(points) > num_points:
        indices = np.random.choice(len(points), size=num_points, replace=False)
        points = points[indices]
        colors_flat = colors_flat[indices]

    return points, colors_flat


def save_ply(path, points, colors):
    """Save point cloud as PLY file."""
    n = len(points)
    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    with open(path, 'wb') as f:
        f.write(header.encode())
        for i in range(n):
            f.write(np.array(points[i], dtype=np.float32).tobytes())
            f.write(colors_uint8[i].tobytes())

    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"Saved {path} ({n:,} points, {size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="360° Panorama → 3D Point Cloud (improved pipeline)")
    parser.add_argument("input", help="Input equirectangular panorama image")
    parser.add_argument("--output", "-o", default="output.ply", help="Output PLY file path")
    parser.add_argument("--num-crops", type=int, default=14, help="Number of perspective crops")
    parser.add_argument("--crop-size", type=int, default=512, help="Size of each perspective crop")
    parser.add_argument("--fov", type=float, default=90, help="Field of view for crops (degrees)")
    parser.add_argument("--num-points", type=int, default=600000, help="Number of output points")
    parser.add_argument("--depth-scale", type=float, default=8.0, help="Depth scale factor")
    parser.add_argument("--eq-size", type=int, default=2048, help="Equirectangular processing width")
    args = parser.parse_args()

    print("=" * 60)
    print("360° Panorama → 3D Point Cloud (Tangent Crop Pipeline)")
    print("=" * 60)

    # Load image
    print(f"\n1. Loading panorama: {args.input}")
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Cannot load {args.input}")
        sys.exit(1)

    orig_h, orig_w = img.shape[:2]
    print(f"   Original size: {orig_w}x{orig_h}")

    # Resize for processing
    eq_w = args.eq_size
    eq_h = eq_w // 2
    img_eq = cv2.resize(img, (eq_w, eq_h), interpolation=cv2.INTER_AREA)
    print(f"   Processing size: {eq_w}x{eq_h}")

    # Generate crop configurations
    # Cover full sphere with overlapping perspective crops
    print(f"\n2. Generating {args.num_crops} perspective crops (FOV={args.fov}°)")

    crop_configs = []
    # Horizontal ring at horizon
    n_horiz = max(6, args.num_crops - 4)
    for i in range(n_horiz):
        theta = i * 360.0 / n_horiz - 180
        crop_configs.append((theta, 0))

    # Add top and bottom views
    crop_configs.append((0, 60))
    crop_configs.append((180, 60))
    crop_configs.append((0, -60))
    crop_configs.append((180, -60))

    # Trim to requested count
    crop_configs = crop_configs[:args.num_crops]

    # Extract perspective crops
    crops = []
    for i, (theta, phi) in enumerate(crop_configs):
        crop = equirect_to_perspective(img_eq, args.fov, theta, phi, args.crop_size, args.crop_size)
        crops.append(crop)
        print(f"   Crop {i+1}: theta={theta:.0f}°, phi={phi:.0f}°")

    # Run depth estimation on each crop
    print(f"\n3. Running Depth Anything V2 on {len(crops)} crops...")
    crop_depths = run_depth_estimation(crops)

    # Map each crop's depth back to equirectangular space and blend
    print(f"\n4. Blending depth maps back to equirectangular...")
    eq_depth_sum = np.zeros((eq_h, eq_w), dtype=np.float64)
    eq_weight_sum = np.zeros((eq_h, eq_w), dtype=np.float64)

    for i, ((theta, phi), depth) in enumerate(zip(crop_configs, crop_depths)):
        # Resize depth to crop size if needed
        if depth.shape != (args.crop_size, args.crop_size):
            depth = cv2.resize(depth, (args.crop_size, args.crop_size))

        d_contrib, w_contrib = perspective_depth_to_equirect(
            depth, args.fov, theta, phi, args.crop_size, args.crop_size, eq_h, eq_w
        )
        eq_depth_sum += d_contrib * w_contrib
        eq_weight_sum += w_contrib
        print(f"   Blended crop {i+1}/{len(crops)}", end='\r')

    print()

    # Normalize blended depth
    eq_weight_sum = np.maximum(eq_weight_sum, 1e-8)
    aligned_depth = (eq_depth_sum / eq_weight_sum).astype(np.float32)

    # Fill any remaining gaps with simple depth
    gaps = eq_weight_sum < 0.01
    if gaps.any():
        print(f"   Filling {gaps.sum():,} gap pixels...")
        aligned_depth = cv2.inpaint(aligned_depth, (gaps.astype(np.uint8) * 255), 5, cv2.INPAINT_TELEA)

    # Smooth the blended depth to reduce seam artifacts between crops
    aligned_depth = cv2.GaussianBlur(aligned_depth, (7, 7), 2.0)

    # Save debug depth visualization
    depth_vis = (aligned_depth * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    debug_path = args.output.replace('.ply', '_depth.png')
    cv2.imwrite(debug_path, depth_vis)
    print(f"   Saved depth visualization: {debug_path}")

    # Backproject to 3D
    print(f"\n5. Backprojecting to 3D ({args.num_points:,} points)...")
    points, colors = backproject_equirect_to_3d(
        img_eq, aligned_depth,
        num_points=args.num_points,
        depth_scale=args.depth_scale
    )

    # Save PLY
    print(f"\n6. Saving PLY...")
    save_ply(args.output, points, colors)

    print(f"\nDone! Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
