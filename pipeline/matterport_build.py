"""
Matterport-style 3D reconstruction from multi-panorama 360° video.

Pipeline:
1. Load all panorama frames
2. Estimate depth for each using Depth Anything V2
3. Estimate camera trajectory using ORB feature matching
4. Backproject each panorama to 3D mesh using depth + spherical projection
5. Export: combined OBJ mesh, per-viewpoint depth images, camera positions JSON

Usage:
    python matterport_build.py <panoramas_dir> <output_dir>
"""

import sys
import os
import json
import glob
import cv2
import numpy as np
import math


def load_depth_model():
    """Load Depth Anything V2."""
    import torch
    from transformers import pipeline as hf_pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Depth Anything V2 (device: {device})...")
    pipe = hf_pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    return pipe


def estimate_depth(img_bgr, pipe):
    """Get depth map from Depth Anything V2."""
    from PIL import Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    result = pipe(pil)
    depth = np.array(result["depth"]).astype(np.float32)
    if "predicted_depth" in result:
        raw = result["predicted_depth"]
        if hasattr(raw, 'numpy'):
            depth = raw.squeeze().numpy().astype(np.float32)
    if depth.shape[:2] != img_bgr.shape[:2]:
        depth = cv2.resize(depth, (img_bgr.shape[1], img_bgr.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    return depth


def depth_to_distance(depth_map):
    """Convert disparity-like depth to metric-like distance."""
    d = depth_map.astype(np.float64)
    d_clipped = np.clip(d, 1.0, None)
    distance = 1.0 / d_clipped
    median_d = np.median(distance)
    if median_d > 0:
        distance = distance / median_d * 3.5
    return np.clip(distance, 0.2, 12.0)


def estimate_trajectory(frames, process_size=(512, 256)):
    """Estimate camera positions from sequential panorama frames using ORB."""
    print("Estimating camera trajectory...")
    positions = [(0.0, 0.0)]  # Start at origin
    headings = [0.0]
    cumulative_yaw = 0.0

    for i in range(1, len(frames)):
        img1 = cv2.resize(frames[i - 1], process_size)
        img2 = cv2.resize(frames[i], process_size)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        yaw_deg = 0.0
        if des1 is not None and des2 is not None and len(kp1) >= 10 and len(kp2) >= 10:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
            good = matches[:min(50, len(matches))]
            if len(good) >= 5:
                w = process_size[0]
                dx_sum = 0
                for m in good:
                    dx = (kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0]) / w * 360
                    if dx > 180: dx -= 360
                    if dx < -180: dx += 360
                    dx_sum += dx
                yaw_deg = dx_sum / len(good)

        cumulative_yaw += yaw_deg
        headings.append(cumulative_yaw)

        # Estimate forward motion (assume ~0.5m per frame)
        step = 0.5
        rad = math.radians(cumulative_yaw)
        x = positions[-1][0] + step * math.sin(rad)
        z = positions[-1][1] + step * math.cos(rad)
        positions.append((x, z))

    print(f"  {len(positions)} viewpoints, total path length: "
          f"{sum(math.sqrt((positions[i][0]-positions[i-1][0])**2 + (positions[i][1]-positions[i-1][1])**2) for i in range(1, len(positions))):.1f}m")
    return positions, headings


def pano_to_mesh_vertices(img_bgr, distance, subsample=4):
    """Convert equirectangular panorama + distance to 3D vertices + colors.
    Returns vertices (N,3), colors (N,3), faces (M,3).
    """
    h, w = img_bgr.shape[:2]

    rows = np.arange(0, h, subsample)
    cols = np.arange(0, w, subsample)
    nr, nc = len(rows), len(cols)
    col_grid, row_grid = np.meshgrid(cols, rows)

    theta = (col_grid.astype(np.float64) / w) * 2 * np.pi
    phi = (0.5 - row_grid.astype(np.float64) / h) * np.pi

    d = distance[row_grid, col_grid]

    x = d * np.cos(phi) * np.sin(theta)
    y = d * np.sin(phi)
    z = d * np.cos(phi) * np.cos(theta)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = img_rgb[row_grid, col_grid, 0]
    g = img_rgb[row_grid, col_grid, 1]
    b = img_rgb[row_grid, col_grid, 2]

    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    colors = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)

    # Generate triangle faces (quad grid → 2 triangles per quad)
    faces = []
    for ri in range(nr - 1):
        for ci in range(nc - 1):
            v0 = ri * nc + ci
            v1 = ri * nc + ci + 1
            v2 = (ri + 1) * nc + ci
            v3 = (ri + 1) * nc + ci + 1
            faces.append((v0, v2, v1))
            faces.append((v1, v2, v3))
    faces = np.array(faces, dtype=np.int32)

    return vertices, colors, faces


def write_obj_with_colors(filepath, all_vertices, all_colors, all_faces):
    """Write OBJ file with vertex colors."""
    with open(filepath, 'w') as f:
        f.write("# Matterport-style 3D mesh from panoramas\n")
        for i in range(len(all_vertices)):
            v = all_vertices[i]
            c = all_colors[i] / 255.0
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
        for face in all_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"  Wrote OBJ: {len(all_vertices):,} vertices, {len(all_faces):,} faces")


def write_ply_mesh(filepath, vertices, colors, faces):
    """Write PLY with mesh faces."""
    n_verts = len(vertices)
    n_faces = len(faces)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {n_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    vdata = np.empty(n_verts, dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
    ])
    vdata['x'] = vertices[:, 0].astype(np.float32)
    vdata['y'] = vertices[:, 1].astype(np.float32)
    vdata['z'] = vertices[:, 2].astype(np.float32)
    vdata['r'] = colors[:, 0].astype(np.uint8)
    vdata['g'] = colors[:, 1].astype(np.uint8)
    vdata['b'] = colors[:, 2].astype(np.uint8)

    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vdata.tobytes())
        for face in faces:
            f.write(np.uint8(3).tobytes())
            f.write(face.astype(np.int32).tobytes())
    print(f"  Wrote PLY mesh: {n_verts:,} verts, {n_faces:,} faces")


def generate_dollhouse_image(vertices, colors, positions, headings, output_path, width=1200, height=800):
    """Render top-down dollhouse view as an image."""
    # Filter to points near ground level (y close to 0 or below median)
    y_vals = vertices[:, 1]
    y_median = np.median(y_vals)
    # Take points in the lower 60% (floor + furniture, not ceiling)
    y_threshold = y_median
    mask = y_vals <= y_threshold

    pts = vertices[mask]
    cols = colors[mask]

    if len(pts) == 0:
        print("  Warning: no points for dollhouse")
        return

    # Project to XZ plane
    x = pts[:, 0]
    z = pts[:, 2]

    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()

    # Add padding
    pad = max(x_max - x_min, z_max - z_min) * 0.1
    x_min -= pad; x_max += pad
    z_min -= pad; z_max += pad

    x_range = x_max - x_min
    z_range = z_max - z_min
    scale = min(width / x_range, height / z_range) * 0.9

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = 20  # Dark background

    # Rasterize points
    px = ((x - x_min) * scale + (width - x_range * scale) / 2).astype(int)
    py = ((z - z_min) * scale + (height - z_range * scale) / 2).astype(int)

    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px = px[valid]
    py = py[valid]
    pt_colors = cols[valid]

    for i in range(len(px)):
        img[py[i], px[i]] = pt_colors[i]

    # Draw viewpoint positions
    for idx, (vx, vz) in enumerate(positions):
        sx = int((vx - x_min) * scale + (width - x_range * scale) / 2)
        sy = int((vz - z_min) * scale + (height - z_range * scale) / 2)
        if 0 <= sx < width and 0 <= sy < height:
            cv2.circle(img, (sx, sy), 8, (0, 200, 255), -1)
            cv2.circle(img, (sx, sy), 9, (255, 255, 255), 1)
            # Draw heading arrow
            heading_rad = math.radians(headings[idx])
            ax = int(sx + 15 * math.sin(heading_rad))
            ay = int(sy + 15 * math.cos(heading_rad))
            cv2.arrowedLine(img, (sx, sy), (ax, ay), (255, 255, 255), 2, tipLength=0.4)

    # Draw path
    for i in range(1, len(positions)):
        sx1 = int((positions[i-1][0] - x_min) * scale + (width - x_range * scale) / 2)
        sy1 = int((positions[i-1][1] - z_min) * scale + (height - z_range * scale) / 2)
        sx2 = int((positions[i][0] - x_min) * scale + (width - x_range * scale) / 2)
        sy2 = int((positions[i][1] - z_min) * scale + (height - z_range * scale) / 2)
        cv2.line(img, (sx1, sy1), (sx2, sy2), (100, 100, 100), 1)

    cv2.imwrite(output_path, img)
    print(f"  Dollhouse image: {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python matterport_build.py <panoramas_dir> <output_dir>")
        sys.exit(1)

    pano_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # Find panorama files
    patterns = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(os.path.join(pano_dir, p))))
    if not files:
        print(f"No panorama images found in {pano_dir}")
        sys.exit(1)
    print(f"Found {len(files)} panoramas")

    # Load all frames (resize for processing)
    process_w, process_h = 1024, 512
    frames_full = []
    frames_proc = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        frames_full.append(img)
        frames_proc.append(cv2.resize(img, (process_w, process_h), interpolation=cv2.INTER_AREA))
    print(f"Loaded {len(frames_proc)} frames at {process_w}x{process_h}")

    # Step 1: Estimate trajectory
    positions, headings = estimate_trajectory(frames_proc)

    # Step 2: Load depth model
    depth_pipe = load_depth_model()

    # Step 3: Process each panorama
    all_vertices = []
    all_colors = []
    all_faces = []
    vertex_offset = 0

    viewpoints_data = []

    for i, (img, pos, heading) in enumerate(zip(frames_proc, positions, headings)):
        print(f"\nProcessing panorama {i+1}/{len(frames_proc)}...")

        # Depth estimation
        depth_raw = estimate_depth(img, depth_pipe)
        distance = depth_to_distance(depth_raw)

        # Save depth map visualization
        depth_vis = ((depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min()) * 255).astype(np.uint8)
        depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(output_dir, f"depth_{i:03d}.jpg"), depth_vis_color)

        # Generate mesh vertices
        vertices, colors, faces = pano_to_mesh_vertices(img, distance, subsample=5)

        # Transform to world coordinates
        heading_rad = math.radians(heading)
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)

        # Rotate around Y axis by heading, then translate
        world_verts = vertices.copy()
        world_verts[:, 0] = vertices[:, 0] * cos_h - vertices[:, 2] * sin_h + pos[0]
        world_verts[:, 2] = vertices[:, 0] * sin_h + vertices[:, 2] * cos_h + pos[1]

        # Add offset to faces
        world_faces = faces + vertex_offset
        vertex_offset += len(world_verts)

        all_vertices.append(world_verts)
        all_colors.append(colors)
        all_faces.append(world_faces)

        # Save viewpoint info
        viewpoints_data.append({
            "id": i,
            "image": os.path.basename(files[i]),
            "position": {"x": pos[0], "z": pos[1], "y": 0.0},
            "heading": heading,
            "num_vertices": len(world_verts),
        })

        print(f"  Depth range: {distance.min():.1f}-{distance.max():.1f}m, "
              f"vertices: {len(world_verts):,}, faces: {len(faces):,}")

    # Combine all meshes
    print(f"\nCombining {len(all_vertices)} meshes...")
    combined_verts = np.vstack(all_vertices)
    combined_colors = np.vstack(all_colors)
    combined_faces = np.vstack(all_faces)

    # Write PLY mesh
    ply_path = os.path.join(output_dir, "scene.ply")
    write_ply_mesh(ply_path, combined_verts, combined_colors, combined_faces)

    # Write OBJ
    obj_path = os.path.join(output_dir, "scene.obj")
    write_obj_with_colors(obj_path, combined_verts, combined_colors, combined_faces)

    # Generate dollhouse image
    dh_path = os.path.join(output_dir, "dollhouse.jpg")
    generate_dollhouse_image(combined_verts, combined_colors, positions, headings, dh_path)

    # Write viewpoints JSON
    scene_data = {
        "viewpoints": viewpoints_data,
        "positions": [{"x": p[0], "z": p[1]} for p in positions],
        "headings": headings,
        "total_vertices": len(combined_verts),
        "total_faces": len(combined_faces),
    }
    json_path = os.path.join(output_dir, "scene.json")
    with open(json_path, 'w') as f:
        json.dump(scene_data, f, indent=2)
    print(f"\nScene data: {json_path}")

    # Summary
    ply_size = os.path.getsize(ply_path) / (1024 * 1024)
    print(f"\n{'='*50}")
    print(f"Matterport-style build complete!")
    print(f"  Viewpoints: {len(viewpoints_data)}")
    print(f"  Total vertices: {len(combined_verts):,}")
    print(f"  Total faces: {len(combined_faces):,}")
    print(f"  PLY mesh: {ply_path} ({ply_size:.1f} MB)")
    print(f"  Dollhouse: {dh_path}")
    print(f"  Scene JSON: {json_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
