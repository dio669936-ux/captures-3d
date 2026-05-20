"""
Generate a 3D point cloud (.ply) from equirectangular 360° panoramas using depth estimation.

Pipeline:
1. Load each panorama
2. Estimate depth with MiDaS
3. Backproject each pixel to 3D using equirectangular projection + depth
4. Subsample for manageable file size
5. Export as .ply (compatible with Gaussian Splatting viewers)

Usage:
    python depth_to_ply.py <panoramas_dir> <output.ply> [--max-points 500000]
"""

import sys
import os
import cv2
import numpy as np
import math
import struct


def load_midas():
    """Load MiDaS small model for depth estimation (CPU)."""
    import torch
    print("Loading MiDaS depth model...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    return midas, transform


def estimate_depth(img_bgr, midas, transform):
    """Run MiDaS depth estimation. Returns inverse-depth map."""
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


def equirect_to_3d_points(img_bgr, inv_depth, subsample=4):
    """Convert equirectangular image + depth to 3D point cloud.

    Each pixel in equirectangular maps to a direction (yaw, pitch).
    With depth, we place each pixel at a 3D position.

    Args:
        img_bgr: BGR image (H, W, 3)
        inv_depth: Inverse depth map (H, W), higher = closer
        subsample: Take every Nth pixel to reduce point count

    Returns:
        points: (N, 3) xyz coordinates
        colors: (N, 3) RGB colors (0-255)
    """
    h, w = img_bgr.shape[:2]

    # Convert inverse depth to distance
    inv_depth_clipped = np.clip(inv_depth, 1e-3, None)
    distance = 1.0 / inv_depth_clipped

    # Normalize so median distance = 3.0 meters (approximate room scale)
    median_d = np.median(distance)
    if median_d > 0:
        distance = distance / median_d * 3.0

    # Clamp extreme values
    distance = np.clip(distance, 0.1, 15.0)

    # Create pixel grid (subsampled)
    rows = np.arange(0, h, subsample)
    cols = np.arange(0, w, subsample)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # Equirectangular to spherical coordinates
    # x-axis = longitude (yaw): 0 to 2*pi
    # y-axis = latitude (pitch): pi/2 to -pi/2 (top to bottom)
    yaw = (col_grid.astype(np.float64) / w) * 2 * np.pi  # 0 to 2*pi
    pitch = (0.5 - row_grid.astype(np.float64) / h) * np.pi  # pi/2 to -pi/2

    # Sample depth at subsampled positions
    d = distance[row_grid, col_grid]

    # Spherical to Cartesian
    # Convention: Y-up, Z-forward at yaw=0
    x = d * np.cos(pitch) * np.sin(yaw)
    y = d * np.sin(pitch)
    z = d * np.cos(pitch) * np.cos(yaw)

    # Sample colors
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = img_rgb[row_grid, col_grid, 0]
    g = img_rgb[row_grid, col_grid, 1]
    b = img_rgb[row_grid, col_grid, 2]

    # Flatten
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    colors = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)

    # Filter out extreme distance points (sky, floor artifacts)
    mask = (d.ravel() < 12.0) & (d.ravel() > 0.2)
    points = points[mask]
    colors = colors[mask]

    return points, colors


def estimate_movement_orb(img1, img2):
    """Estimate camera yaw rotation between two equirectangular frames."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        return 0.0

    good = matches[:min(50, len(matches))]
    h, w = gray1.shape
    dx_sum = 0
    for m in good:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        dx = (pt2[0] - pt1[0]) / w * 360
        if dx > 180: dx -= 360
        if dx < -180: dx += 360
        dx_sum += dx
    return dx_sum / len(good)


def write_ply(filename, points, colors):
    """Write point cloud as PLY file."""
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
        print("Usage: python depth_to_ply.py <panoramas_dir> <output.ply> [--max-points N]")
        sys.exit(1)

    panorama_dir = sys.argv[1]
    output_ply = sys.argv[2]

    max_points = 500000
    if "--max-points" in sys.argv:
        idx = sys.argv.index("--max-points")
        max_points = int(sys.argv[idx + 1])

    panorama_files = sorted([
        f for f in os.listdir(panorama_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Found {len(panorama_files)} panorama frames")

    # Load MiDaS
    midas, transform = load_midas()

    # Build trajectory (camera positions)
    print("\nEstimating camera trajectory...")
    positions = [(0.0, 0.0, 0.0)]  # x, y, z in 3D
    headings = [0.0]
    prev_small = None

    for i, fname in enumerate(panorama_files):
        img = cv2.imread(os.path.join(panorama_dir, fname))
        if img is None:
            continue
        small = cv2.resize(img, (960, 480))
        if prev_small is not None:
            dx = estimate_movement_orb(prev_small, small)
            heading = headings[-1] - dx
            headings.append(heading)
            step = 1.5  # ~1.5 meters per frame interval
            x = positions[-1][0] + step * math.sin(math.radians(heading))
            z = positions[-1][2] - step * math.cos(math.radians(heading))
            positions.append((x, 0.0, z))
        prev_small = small
        print(f"  Trajectory: frame {i+1}/{len(panorama_files)}")

    # Process each panorama → 3D points
    print("\nGenerating 3D point cloud...")
    all_points = []
    all_colors = []

    # Calculate subsample rate to stay within max_points
    # Each panorama at subsample=4 on 1024x512 gives ~32K points
    # With 17 panoramas = ~544K points
    target_per_frame = max_points // len(panorama_files)
    # At resolution 1024x512 with subsample s: points ≈ (1024/s) * (512/s)
    # target = 1024*512 / s^2 → s = sqrt(1024*512/target)
    subsample = max(2, int(math.sqrt(1024 * 512 / max(target_per_frame, 1000))))
    print(f"  Subsample rate: {subsample} (target {target_per_frame:,} pts/frame)")

    for i, fname in enumerate(panorama_files):
        img = cv2.imread(os.path.join(panorama_dir, fname))
        if img is None:
            continue

        # Resize for depth estimation
        small = cv2.resize(img, (1024, 512))
        inv_depth = estimate_depth(small, midas, transform)

        # Get 3D points
        pts, cols = equirect_to_3d_points(small, inv_depth, subsample=subsample)

        # Rotate points according to camera heading
        heading_rad = math.radians(headings[i] if i < len(headings) else 0)
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        rot_pts = pts.copy()
        rot_pts[:, 0] = pts[:, 0] * cos_h - pts[:, 2] * sin_h
        rot_pts[:, 2] = pts[:, 0] * sin_h + pts[:, 2] * cos_h

        # Translate to camera position
        cx, cy, cz = positions[i] if i < len(positions) else (0, 0, 0)
        rot_pts[:, 0] += cx
        rot_pts[:, 1] += cy
        rot_pts[:, 2] += cz

        all_points.append(rot_pts)
        all_colors.append(cols)

        print(f"  Frame {i+1}/{len(panorama_files)}: {len(pts):,} points")

    # Combine all points
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    print(f"\nTotal: {len(all_points):,} points")

    # Random subsample if still too many
    if len(all_points) > max_points:
        idx = np.random.choice(len(all_points), max_points, replace=False)
        all_points = all_points[idx]
        all_colors = all_colors[idx]
        print(f"Subsampled to {max_points:,} points")

    # Write PLY
    write_ply(output_ply, all_points, all_colors)
    print(f"\nDone! Output: {output_ply}")
    print(f"View in: https://captures-3d-ognvwkqc.devinapps.com (upload PLY file)")
    print(f"Or view in: https://antimatter15.com/splat/ (drag & drop PLY)")


if __name__ == "__main__":
    main()
