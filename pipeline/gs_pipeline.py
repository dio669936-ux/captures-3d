#!/usr/bin/env python3
"""
Unified Gaussian Splatting Pipeline
Input: video (regular/360°), images (regular/360°)
Output: .ply Gaussian Splatting file

Handles:
1. Regular video → ffmpeg frames → COLMAP → OpenSplat → .ply
2. 360° video → ffmpeg frames → perspective crops → COLMAP → OpenSplat → .ply
3. Multiple regular images → COLMAP → OpenSplat → .ply
4. 360° panorama (1 or more) → tangent crops + depth estimation → .ply point cloud
"""

import sys
import os
import argparse
import subprocess
import shutil
import glob
from pathlib import Path

OPENSPLAT_PATH = "/home/ubuntu/OpenSplat/build/opensplat"


def detect_input_type(input_path):
    """Detect whether input is video, directory of images, or single image."""
    p = Path(input_path)
    
    if p.is_dir():
        images = list(p.glob("*.jpg")) + list(p.glob("*.JPG")) + list(p.glob("*.png")) + list(p.glob("*.PNG"))
        if not images:
            print(f"Error: No images found in {input_path}")
            sys.exit(1)
        return "images", images
    
    if p.is_file():
        ext = p.suffix.lower()
        if ext in ('.mp4', '.mov', '.avi', '.mkv', '.webm'):
            return "video", [p]
        elif ext in ('.jpg', '.jpeg', '.png', '.tiff', '.bmp'):
            return "single_image", [p]
    
    print(f"Error: Cannot detect input type for {input_path}")
    sys.exit(1)


def is_panorama(image_path):
    """Check if image is equirectangular panorama (2:1 aspect ratio)."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    ratio = w / h
    return 1.8 < ratio < 2.2


def extract_video_frames(video_path, output_dir, fps=2, max_frames=100):
    """Extract frames from video using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        "-frames:v", str(max_frames),
        f"{output_dir}/frame_%04d.jpg"
    ]
    
    print(f"Extracting frames from {video_path} (fps={fps}, max={max_frames})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        sys.exit(1)
    
    frames = sorted(glob.glob(f"{output_dir}/frame_*.jpg"))
    print(f"  Extracted {len(frames)} frames")
    return frames


def extract_perspective_crops(image_paths, output_dir, crop_size=800, fov=80, is_360=True):
    """Extract perspective crops from equirectangular panoramas."""
    import cv2
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent))
    from pano360_to_ply import equirect_to_perspective
    
    os.makedirs(output_dir, exist_ok=True)
    total = 0
    
    for pi, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        eq_w, eq_h = 4096, 2048
        img_eq = cv2.resize(img, (eq_w, eq_h))
        print(f"  Processing {Path(img_path).name}...")
        
        # Generate overlapping perspective crops
        for i in range(8):
            theta = i * 45.0
            crop = equirect_to_perspective(img_eq, fov, theta, 0, crop_size, crop_size)
            fname = f"{output_dir}/p{pi:02d}_h{i:02d}_t{theta:.0f}.jpg"
            cv2.imwrite(fname, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            total += 1
        
        # Add slightly elevated views
        for theta in [0, 90, 180, 270]:
            crop = equirect_to_perspective(img_eq, fov, theta, 20, crop_size, crop_size)
            fname = f"{output_dir}/p{pi:02d}_up_t{theta:.0f}.jpg"
            cv2.imwrite(fname, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            total += 1
    
    print(f"  Total perspective crops: {total}")
    return total


def run_colmap(project_dir, image_dir):
    """Run COLMAP SfM pipeline."""
    db_path = f"{project_dir}/database.db"
    sparse_dir = f"{project_dir}/sparse"
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Feature extraction
    print("  COLMAP: Feature extraction...")
    result = subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_PINHOLE",
        "--SiftExtraction.max_image_size", "800",
        "--SiftExtraction.max_num_features", "8192",
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  COLMAP feature extraction failed: {result.stderr[:200]}")
        return False
    
    # Exhaustive matching
    print("  COLMAP: Feature matching...")
    result = subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", db_path,
        "--SiftMatching.guided_matching", "1",
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  COLMAP matching failed: {result.stderr[:200]}")
        return False
    
    # Mapper
    print("  COLMAP: Reconstruction...")
    result = subprocess.run([
        "colmap", "mapper",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir,
        "--Mapper.ba_global_max_num_iterations", "50",
    ], capture_output=True, text=True)
    
    # Check if reconstruction succeeded
    if os.path.exists(f"{sparse_dir}/0"):
        n_images = len(glob.glob(f"{sparse_dir}/0/images.*"))
        print(f"  COLMAP: Reconstruction successful")
        return True
    else:
        print("  COLMAP: No good reconstruction found")
        return False


def run_opensplat(project_dir, output_path, num_iters=5000, downscales=2):
    """Run OpenSplat Gaussian Splatting training."""
    print(f"  OpenSplat: Training ({num_iters} iterations, CPU)...")
    
    result = subprocess.run([
        OPENSPLAT_PATH,
        project_dir,
        "--cpu",
        "-n", str(num_iters),
        "-o", output_path,
        "--num-downscales", str(downscales),
        "--sh-degree", "1",
    ], capture_output=True, text=True, timeout=7200)
    
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  OpenSplat: Output saved ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  OpenSplat: Training failed")
        if result.stderr:
            print(f"  Error: {result.stderr[:300]}")
        return False


def run_depth_pipeline(image_paths, output_path, num_points=500000):
    """Run depth-based pipeline for 360° panoramas (no COLMAP needed)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from pano360_to_ply import (
        equirect_to_perspective, run_depth_estimation,
        perspective_depth_to_equirect, backproject_equirect_to_3d, save_ply
    )
    import cv2
    import numpy as np
    
    all_points = []
    all_colors = []
    
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        eq_w, eq_h = 2048, 1024
        img_eq = cv2.resize(img, (eq_w, eq_h))
        print(f"  Processing {Path(img_path).name}...")
        
        # Generate perspective crops
        fov = 90
        crop_size = 512
        crop_configs = []
        for i in range(14):
            theta = i * 360.0 / 10 - 180 if i < 10 else [0, 180, 0, 180][i-10]
            phi = 0 if i < 10 else [60, 60, -60, -60][i-10]
            crop_configs.append((theta, phi))
        
        crops = []
        for theta, phi in crop_configs:
            crop = equirect_to_perspective(img_eq, fov, theta, phi, crop_size, crop_size)
            crops.append(crop)
        
        # Run depth estimation
        crop_depths = run_depth_estimation(crops)
        
        # Blend back to equirectangular
        eq_depth_sum = np.zeros((eq_h, eq_w), dtype=np.float64)
        eq_weight_sum = np.zeros((eq_h, eq_w), dtype=np.float64)
        
        for (theta, phi), depth in zip(crop_configs, crop_depths):
            if depth.shape != (crop_size, crop_size):
                depth = cv2.resize(depth, (crop_size, crop_size))
            d_contrib, w_contrib = perspective_depth_to_equirect(
                depth, fov, theta, phi, crop_size, crop_size, eq_h, eq_w
            )
            eq_depth_sum += d_contrib * w_contrib
            eq_weight_sum += w_contrib
        
        eq_weight_sum = np.maximum(eq_weight_sum, 1e-8)
        aligned_depth = (eq_depth_sum / eq_weight_sum).astype(np.float32)
        
        gaps = eq_weight_sum < 0.01
        if gaps.any():
            aligned_depth = cv2.inpaint(aligned_depth, (gaps.astype(np.uint8) * 255), 5, cv2.INPAINT_TELEA)
        aligned_depth = cv2.GaussianBlur(aligned_depth, (7, 7), 2.0)
        
        pts_per_pano = num_points // len(image_paths)
        points, colors = backproject_equirect_to_3d(img_eq, aligned_depth, pts_per_pano, 6.0)
        all_points.append(points)
        all_colors.append(colors)
    
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    save_ply(output_path, points, colors)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Unified Gaussian Splatting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From regular video (phone camera walking around)
  python gs_pipeline.py video.mp4 -o output.ply
  
  # From 360° panorama images
  python gs_pipeline.py ./panoramas/ -o output.ply --panorama
  
  # From regular photos
  python gs_pipeline.py ./photos/ -o output.ply
  
  # From single 360° panorama
  python gs_pipeline.py panorama.jpg -o output.ply --panorama
        """
    )
    parser.add_argument("input", help="Input video, image, or directory of images")
    parser.add_argument("-o", "--output", default="output.ply", help="Output .ply file")
    parser.add_argument("--panorama", action="store_true", help="Force treat input as 360° panorama")
    parser.add_argument("--video-fps", type=float, default=2, help="Frame extraction rate for video")
    parser.add_argument("--max-frames", type=int, default=60, help="Max frames to extract from video")
    parser.add_argument("--num-iters", type=int, default=5000, help="OpenSplat training iterations")
    parser.add_argument("--num-points", type=int, default=500000, help="Points for depth-based pipeline")
    parser.add_argument("--work-dir", default=None, help="Working directory (auto-created if not set)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Unified Gaussian Splatting Pipeline")
    print("=" * 60)
    
    # Detect input type
    input_type, files = detect_input_type(args.input)
    print(f"\nInput: {args.input}")
    print(f"Type: {input_type} ({len(files)} file(s))")
    
    # Check if panorama
    is_pano = args.panorama
    if not is_pano and input_type in ("images", "single_image"):
        is_pano = is_panorama(files[0])
        if is_pano:
            print("  Auto-detected: 360° panorama (2:1 aspect ratio)")
    
    # Setup work directory
    work_dir = args.work_dir or f"/tmp/gs_pipeline_{os.getpid()}"
    os.makedirs(work_dir, exist_ok=True)
    print(f"Work dir: {work_dir}")
    
    # Pipeline routing
    if input_type == "video":
        # Extract frames
        frame_dir = f"{work_dir}/frames"
        frames = extract_video_frames(files[0], frame_dir, args.video_fps, args.max_frames)
        
        if is_pano:
            # 360° video → extract perspective crops
            print("\n360° video detected → extracting perspective crops...")
            crop_dir = f"{work_dir}/images"
            extract_perspective_crops(
                [Path(f) for f in frames], crop_dir
            )
            # Try COLMAP on crops
            print("\nRunning COLMAP on perspective crops...")
            colmap_ok = run_colmap(work_dir, crop_dir)
        else:
            # Regular video → COLMAP directly
            print("\nRunning COLMAP on video frames...")
            # Copy frames to images dir
            img_dir = f"{work_dir}/images"
            os.makedirs(img_dir, exist_ok=True)
            for f in frames:
                shutil.copy2(f, img_dir)
            colmap_ok = run_colmap(work_dir, img_dir)
        
        if colmap_ok:
            print("\nTraining Gaussian Splatting...")
            run_opensplat(work_dir, args.output, args.num_iters)
        else:
            print("\nCOLMAP failed → falling back to depth-based pipeline...")
            # Get the panorama frames for depth pipeline
            if is_pano:
                run_depth_pipeline([Path(f) for f in frames[:5]], args.output, args.num_points)
            else:
                print("Error: COLMAP failed and depth fallback only works for 360° content.")
                print("Tips: Ensure video has camera movement (not just rotation)")
                sys.exit(1)
    
    elif input_type == "images":
        if is_pano:
            # 360° panoramas → depth-based pipeline (best for 360°)
            print("\n360° panorama images → depth-based 3D pipeline...")
            print("(Tangent crop depth estimation — no GPU needed)")
            run_depth_pipeline(files, args.output, args.num_points)
        else:
            # Regular images → COLMAP + OpenSplat
            img_dir = f"{work_dir}/images"
            os.makedirs(img_dir, exist_ok=True)
            for f in files:
                shutil.copy2(str(f), img_dir)
            
            print("\nRunning COLMAP on images...")
            colmap_ok = run_colmap(work_dir, img_dir)
            
            if colmap_ok:
                print("\nTraining Gaussian Splatting...")
                run_opensplat(work_dir, args.output, args.num_iters)
            else:
                print("Error: COLMAP reconstruction failed.")
                print("Tips: Ensure images have >50% overlap and different viewpoints")
                sys.exit(1)
    
    elif input_type == "single_image":
        if is_pano:
            print("\nSingle 360° panorama → depth-based 3D pipeline...")
            run_depth_pipeline(files, args.output, args.num_points)
        else:
            print("\nSingle regular image → depth-based point cloud...")
            print("Note: Single image cannot produce proper Gaussian Splatting.")
            print("      Output is depth-estimated point cloud.")
            # Simple depth-based output for single regular image
            from pano360_to_ply import run_depth_estimation, save_ply
            import cv2
            import numpy as np
            
            img = cv2.imread(str(files[0]))
            h, w = img.shape[:2]
            img_small = cv2.resize(img, (min(w, 1024), min(h, 1024)))
            
            from PIL import Image
            from transformers import pipeline as hf_pipeline
            import torch
            
            pipe = hf_pipeline("depth-estimation", 
                             model="depth-anything/Depth-Anything-V2-Small-hf",
                             device="cpu", torch_dtype=torch.float32)
            pil_img = Image.fromarray(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
            result = pipe(pil_img)
            depth = np.array(result["depth"]).astype(np.float32)
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth = (depth - d_min) / (d_max - d_min)
            
            sh, sw = img_small.shape[:2]
            u, v = np.meshgrid(np.arange(sw), np.arange(sh))
            z = (1.0 - depth) * 5.0 + 0.5
            x = (u - sw/2) / sw * z * 2
            y = -(v - sh/2) / sh * z * 2
            
            points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
            colors = img_small.reshape(-1, 3)[:, ::-1] / 255.0
            
            if len(points) > args.num_points:
                idx = np.random.choice(len(points), args.num_points, replace=False)
                points, colors = points[idx], colors[idx]
            
            save_ply(args.output, points, colors)
    
    print(f"\nOutput: {args.output}")
    if os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f"Size: {size_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
