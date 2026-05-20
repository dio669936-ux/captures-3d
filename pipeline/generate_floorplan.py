"""
Generate a floor plan from equirectangular 360° panoramas using depth estimation.

Pipeline:
1. Use MiDaS to estimate depth from each panorama
2. Sample wall-distance at horizon (pitch≈0) for all yaw angles
3. Project wall points to 2D using camera trajectory
4. Accumulate wall points into a top-down occupancy map
5. Draw floor plan with walls, walking path, and hotspots

Usage:
    python generate_floorplan.py <panoramas_dir> <output_dir>
"""

import sys
import os
import json
import cv2
import numpy as np
import math


def equirect_to_perspective(equirect_img, fov, theta, phi, out_h, out_w):
    """Extract a perspective view from an equirectangular image."""
    h, w = equirect_img.shape[:2]
    f = out_w / (2 * math.tan(math.radians(fov / 2)))

    u = np.arange(out_w) - out_w / 2
    v = np.arange(out_h) - out_h / 2
    u, v = np.meshgrid(u, v)

    x = u
    y = -v
    z = np.full_like(u, f, dtype=np.float64)
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/norm, y/norm, z/norm

    theta_r = math.radians(theta)
    phi_r = math.radians(phi)

    cos_p, sin_p = math.cos(phi_r), math.sin(phi_r)
    x1 = x
    y1 = y * cos_p - z * sin_p
    z1 = y * sin_p + z * cos_p

    cos_t, sin_t = math.cos(theta_r), math.sin(theta_r)
    x2 = x1 * cos_t + z1 * sin_t
    y2 = y1
    z2 = -x1 * sin_t + z1 * cos_t

    lon = np.arctan2(x2, z2)
    lat = np.arcsin(np.clip(y2, -1, 1))

    px = (lon / math.pi + 1) * w / 2
    py = (-lat / (math.pi/2) + 1) * h / 2
    px = np.clip(px, 0, w - 1).astype(np.float32)
    py = np.clip(py, 0, h - 1).astype(np.float32)

    return cv2.remap(equirect_img, px, py, cv2.INTER_LINEAR)


def extract_nadir_view(equirect_img, size=512):
    """Extract the nadir (floor/downward) view from equirectangular image."""
    return equirect_to_perspective(equirect_img, fov=120, theta=0, phi=-90,
                                   out_h=size, out_w=size)


def estimate_movement(img1, img2):
    """Estimate camera movement between two equirectangular frames using ORB."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        return 0.0, 0.0

    good_matches = matches[:min(50, len(matches))]
    h, w = gray1.shape
    dx_sum = 0

    for m in good_matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        dx = (pt2[0] - pt1[0]) / w * 360
        if dx > 180: dx -= 360
        if dx < -180: dx += 360
        dx_sum += dx

    return dx_sum / len(good_matches), 0.0


def build_trajectory(panorama_files, panorama_dir):
    """Build camera trajectory from sequential panorama frames."""
    positions = [(0.0, 0.0)]
    headings = [0.0]
    prev_img = None

    for i, fname in enumerate(panorama_files):
        img_path = os.path.join(panorama_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        small = cv2.resize(img, (960, 480))

        if prev_img is not None:
            dx, _ = estimate_movement(prev_img, small)
            heading = headings[-1] - dx
            headings.append(heading)

            step = 30
            x = positions[-1][0] + step * math.sin(math.radians(heading))
            y = positions[-1][1] - step * math.cos(math.radians(heading))
            positions.append((x, y))

        prev_img = small
        print(f"  Trajectory: frame {i+1}/{len(panorama_files)}")

    return positions, headings


def load_midas():
    """Load MiDaS small model for depth estimation (CPU)."""
    import torch
    print("  Loading MiDaS depth model...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    return midas, transform


def estimate_depth(img_bgr, midas, transform):
    """Run MiDaS depth estimation on a BGR image.

    Returns inverse-depth map (higher = closer) at original resolution.
    """
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


def horizon_depth_to_wall_points(inv_depth, n_samples=360):
    """Extract wall positions from the equirectangular depth map.

    Samples the depth at the horizon (pitch=0, middle row) at n_samples
    evenly-spaced yaw angles.  Returns (n_samples, 2) array of (x, z)
    wall coordinates relative to the camera.
    """
    h, w = inv_depth.shape

    # Sample a band around the horizon (middle ±5% of height) and average
    band_h = max(1, int(h * 0.05))
    mid = h // 2
    band = inv_depth[mid - band_h : mid + band_h, :]  # shape (2*band_h, w)
    horizon_inv = np.mean(band, axis=0)  # shape (w,)

    # Convert inverse depth to distance (clip to avoid div-by-zero)
    horizon_inv = np.clip(horizon_inv, 1e-3, None)
    distance = 1.0 / horizon_inv
    # Normalize: median wall distance ≈ 2x the trajectory step (step=30)
    # so walls extend well beyond camera positions to form room boundaries
    median_d = np.median(distance)
    if median_d > 0:
        distance = distance / median_d * 60.0
    # Clamp very far distances (open areas / sky) to avoid outliers
    distance = np.clip(distance, 0, 200.0)

    # Resample to n_samples evenly-spaced yaw angles
    cols = np.linspace(0, w - 1, n_samples, endpoint=False).astype(int)
    dist_sampled = distance[cols]

    # Yaw angle per sample (0 to 2π)
    yaw = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

    # Convert polar (yaw, distance) to Cartesian wall coords
    wx = dist_sampled * np.sin(yaw)
    wz = -dist_sampled * np.cos(yaw)

    return np.stack([wx, wz], axis=1)


def accumulate_wall_map(all_wall_points, positions, canvas_size=800, margin=80):
    """Accumulate wall points from all cameras into a 2D occupancy grid.

    Returns:
        wall_grid: (canvas_size, canvas_size) float32 density map
        transform: (scale, offset_x, offset_y) to convert world→pixel
    """
    # Shift wall points to global coordinates
    global_pts = []
    for pts, (cx, cz) in zip(all_wall_points, positions):
        shifted = pts.copy()
        shifted[:, 0] += cx
        shifted[:, 1] += cz
        global_pts.append(shifted)

    all_pts = np.concatenate(global_pts, axis=0)

    # Also include camera positions for bounding box
    cam_arr = np.array(positions)
    all_for_bounds = np.concatenate([all_pts, cam_arr], axis=0)

    min_x, min_z = all_for_bounds.min(axis=0)
    max_x, max_z = all_for_bounds.max(axis=0)

    range_x = max(max_x - min_x, 0.1)
    range_z = max(max_z - min_z, 0.1)
    scale = min((canvas_size - 2 * margin) / range_x,
                (canvas_size - 2 * margin) / range_z)
    off_x = canvas_size / 2 - (min_x + max_x) / 2 * scale
    off_z = canvas_size / 2 - (min_z + max_z) / 2 * scale

    grid = np.zeros((canvas_size, canvas_size), dtype=np.float32)

    # Draw wall points as small circles for visibility
    dot_radius = max(2, int(scale * 0.15))
    for x, z in all_pts:
        px = int(x * scale + off_x)
        pz = int(z * scale + off_z)
        if 0 <= px < canvas_size and 0 <= pz < canvas_size:
            cv2.circle(grid, (px, pz), dot_radius, 1.0, -1)

    # Also draw wall outlines per camera for connected structure
    for pts_shifted in global_pts:
        pixels = []
        for x, z in pts_shifted:
            px = int(x * scale + off_x)
            pz = int(z * scale + off_z)
            px = np.clip(px, 0, canvas_size - 1)
            pz = np.clip(pz, 0, canvas_size - 1)
            pixels.append((px, pz))
        for j in range(len(pixels) - 1):
            cv2.line(grid, pixels[j], pixels[j + 1], 0.5, 1)
        if pixels:
            cv2.line(grid, pixels[-1], pixels[0], 0.5, 1)

    return grid, (scale, off_x, off_z)


def create_floorplan_image(wall_grid, positions, nadir_views, transform_params,
                           canvas_size=800, hotspot_radius=12):
    """Render the floor plan image with walls, trajectory, and hotspot markers."""
    scale, off_x, off_z = transform_params

    # Blur the wall grid to create smooth wall outlines
    blurred = cv2.GaussianBlur(wall_grid, (11, 11), 3.0)

    # Normalize to 0-255
    if blurred.max() > 0:
        norm = (blurred / blurred.max() * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(blurred, dtype=np.uint8)

    # Threshold to get wall mask
    _, wall_mask = cv2.threshold(norm, 30, 255, cv2.THRESH_BINARY)

    # Create colored canvas
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[:] = (30, 26, 26)  # dark background

    # Draw wall density as a heatmap overlay
    wall_color = cv2.applyColorMap(norm, cv2.COLORMAP_BONE)
    # Only show where there is signal
    mask3 = np.stack([norm > 5] * 3, axis=-1)
    canvas[mask3] = wall_color[mask3]

    # Draw wall edges (contours) with different thresholds for detail
    for thresh_val, thickness, color in [
        (60, 2, (220, 220, 220)),
        (30, 1, (160, 160, 160)),
    ]:
        _, thresh_mask = cv2.threshold(norm, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        # Filter out tiny contours
        contours = [c for c in contours if cv2.contourArea(c) > 20]
        cv2.drawContours(canvas, contours, -1, color, thickness, cv2.LINE_AA)

    # Convert camera positions to pixel coordinates
    normalized = []
    for (cx, cz) in positions:
        px = int(cx * scale + off_x)
        pz = int(cz * scale + off_z)
        normalized.append((px, pz))

    # Draw nadir thumbnails at camera positions
    thumb_size = 36
    for i, (nx, ny) in enumerate(normalized):
        if i < len(nadir_views) and nadir_views[i] is not None:
            thumb = cv2.resize(nadir_views[i], (thumb_size, thumb_size))
            x1 = max(0, nx - thumb_size // 2)
            y1 = max(0, ny - thumb_size // 2)
            x2 = min(canvas_size, x1 + thumb_size)
            y2 = min(canvas_size, y1 + thumb_size)
            tw, th = x2 - x1, y2 - y1
            if tw > 0 and th > 0:
                region = canvas[y1:y2, x1:x2]
                thumb_crop = thumb[:th, :tw]
                if region.shape == thumb_crop.shape:
                    blended = cv2.addWeighted(region, 0.4, thumb_crop, 0.6, 0)
                    canvas[y1:y2, x1:x2] = blended

    # Draw walking path
    path_color = (79, 195, 247)
    for i in range(len(normalized) - 1):
        cv2.line(canvas, normalized[i], normalized[i + 1], path_color, 2,
                 cv2.LINE_AA)

    # Draw hotspot dots
    for i, (nx, ny) in enumerate(normalized):
        cv2.circle(canvas, (nx, ny), hotspot_radius + 4, path_color, 1,
                   cv2.LINE_AA)
        cv2.circle(canvas, (nx, ny), hotspot_radius, path_color, -1,
                   cv2.LINE_AA)
        cv2.circle(canvas, (nx, ny), hotspot_radius, (255, 255, 255), 2,
                   cv2.LINE_AA)
        text = str(i + 1)
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(canvas, text, (nx - ts[0] // 2, ny + ts[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                    cv2.LINE_AA)

    # Title
    cv2.putText(canvas, "Floor Plan", (canvas_size // 2 - 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                cv2.LINE_AA)

    return canvas, normalized


def generate_tour_config(positions, panorama_files, canvas_size=800):
    """Generate JSON configuration for the virtual tour viewer."""
    config = {
        "panoramas": [],
        "canvasSize": canvas_size,
        "floorPlanImage": "floorplan.jpg",
    }
    for i, fname in enumerate(panorama_files):
        pano = {
            "id": f"pano_{i+1:03d}",
            "src": f"panoramas/{fname}",
            "label": f"Viewpoint {i+1}",
            "x": positions[i][0] if i < len(positions) else 0,
            "y": positions[i][1] if i < len(positions) else 0,
        }
        config["panoramas"].append(pano)
    return config


def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_floorplan.py <panoramas_dir> <output_dir>")
        sys.exit(1)

    panorama_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    panorama_files = sorted([
        f for f in os.listdir(panorama_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Found {len(panorama_files)} panorama frames")

    # Step 1: Load MiDaS
    print("\nStep 1: Loading depth estimation model...")
    midas, transform = load_midas()

    # Step 2: Extract nadir views + estimate depth + get wall points
    print("\nStep 2: Depth estimation + wall detection...")
    nadir_views = []
    all_wall_points = []

    for i, fname in enumerate(panorama_files):
        img_path = os.path.join(panorama_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            nadir_views.append(None)
            all_wall_points.append(np.zeros((360, 2)))
            continue

        # Nadir view
        nadir = extract_nadir_view(img, size=256)
        nadir_views.append(nadir)

        # Depth estimation (resize for speed)
        small = cv2.resize(img, (1024, 512))
        inv_depth = estimate_depth(small, midas, transform)

        # Wall points from horizon depth
        wall_pts = horizon_depth_to_wall_points(inv_depth, n_samples=360)
        all_wall_points.append(wall_pts)

        print(f"  Frame {i+1}/{len(panorama_files)}: "
              f"depth range {inv_depth.min():.1f}-{inv_depth.max():.1f}")

    # Step 3: Build camera trajectory
    print("\nStep 3: Estimating camera trajectory...")
    positions, headings = build_trajectory(panorama_files, panorama_dir)

    # Rotate wall points according to camera heading at each position
    rotated_wall_points = []
    for pts, heading in zip(all_wall_points, headings):
        angle = math.radians(heading)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = pts @ rot.T
        rotated_wall_points.append(rotated)

    # Step 4: Accumulate wall map
    print("\nStep 4: Building wall map...")
    wall_grid, transform_params = accumulate_wall_map(
        rotated_wall_points, positions, canvas_size=800
    )

    # Step 5: Create floor plan image
    print("\nStep 5: Rendering floor plan...")
    floorplan, normalized_pos = create_floorplan_image(
        wall_grid, positions, nadir_views, transform_params, canvas_size=800
    )
    floorplan_path = os.path.join(output_dir, "floorplan.jpg")
    cv2.imwrite(floorplan_path, floorplan, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Floor plan saved: {floorplan_path}")

    # Step 6: Generate tour config
    print("\nStep 6: Generating tour config...")
    config = generate_tour_config(normalized_pos, panorama_files)
    config_path = os.path.join(output_dir, "tour_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {config_path}")

    print(f"\nDone! Files saved to {output_dir}")


if __name__ == "__main__":
    main()
