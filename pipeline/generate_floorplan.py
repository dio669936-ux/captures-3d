"""
Generate a floor plan + camera trajectory from equirectangular 360° panorama frames.

This script:
1. Extracts nadir (downward-looking) views from each equirectangular frame
2. Uses feature matching to estimate relative camera movement between frames
3. Builds a camera trajectory (walking path)
4. Creates a floor plan image with the trajectory and hotspot positions
5. Outputs a JSON config for the virtual tour viewer

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
    """Extract a perspective view from an equirectangular image.
    
    Args:
        equirect_img: Equirectangular image (H x W x 3)
        fov: Field of view in degrees
        theta: Horizontal angle (yaw) in degrees, 0 = center
        phi: Vertical angle (pitch) in degrees, -90 = nadir, +90 = zenith
        out_h, out_w: Output image dimensions
    
    Returns:
        Perspective image (out_h x out_w x 3)
    """
    h, w = equirect_img.shape[:2]
    
    f = out_w / (2 * math.tan(math.radians(fov / 2)))
    
    # Create pixel grid
    u = np.arange(out_w) - out_w / 2
    v = np.arange(out_h) - out_h / 2
    u, v = np.meshgrid(u, v)
    
    # Ray direction in camera space
    x = u
    y = -v  
    z = np.full_like(u, f, dtype=np.float64)
    
    # Normalize
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/norm, y/norm, z/norm
    
    # Rotation matrices
    theta_r = math.radians(theta)
    phi_r = math.radians(phi)
    
    # Rotation around X (pitch)
    cos_p, sin_p = math.cos(phi_r), math.sin(phi_r)
    x1 = x
    y1 = y * cos_p - z * sin_p
    z1 = y * sin_p + z * cos_p
    
    # Rotation around Y (yaw)
    cos_t, sin_t = math.cos(theta_r), math.sin(theta_r)
    x2 = x1 * cos_t + z1 * sin_t
    y2 = y1
    z2 = -x1 * sin_t + z1 * cos_t
    
    # Convert to spherical coordinates
    lon = np.arctan2(x2, z2)
    lat = np.arcsin(np.clip(y2, -1, 1))
    
    # Map to equirectangular pixel coordinates
    px = (lon / math.pi + 1) * w / 2
    py = (-lat / (math.pi/2) + 1) * h / 2
    
    px = np.clip(px, 0, w - 1).astype(np.float32)
    py = np.clip(py, 0, h - 1).astype(np.float32)
    
    result = cv2.remap(equirect_img, px, py, cv2.INTER_LINEAR)
    return result


def extract_nadir_view(equirect_img, size=512):
    """Extract the nadir (floor/downward) view from equirectangular image."""
    return equirect_to_perspective(equirect_img, fov=120, theta=0, phi=-90, 
                                   out_h=size, out_w=size)


def extract_forward_views(equirect_img, n_views=4, size=512):
    """Extract forward-facing perspective views at different yaw angles."""
    views = []
    for i in range(n_views):
        theta = i * (360 / n_views)
        view = equirect_to_perspective(equirect_img, fov=90, theta=theta, phi=0,
                                       out_h=size, out_w=size)
        views.append((theta, view))
    return views


def estimate_movement(img1, img2):
    """Estimate camera movement between two equirectangular frames using feature matching."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Use ORB features
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0, 0.0  # No movement detected
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 10:
        return 0.0, 0.0
    
    # Use top matches to estimate displacement
    good_matches = matches[:min(50, len(matches))]
    
    dx_sum = 0
    dy_sum = 0
    h, w = gray1.shape
    
    for m in good_matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        
        # Convert pixel displacement to angular displacement
        # In equirectangular: x maps to longitude, y maps to latitude
        dx = (pt2[0] - pt1[0]) / w * 360  # degrees longitude
        dy = (pt2[1] - pt1[1]) / h * 180  # degrees latitude
        
        # Handle wrap-around
        if dx > 180: dx -= 360
        if dx < -180: dx += 360
        
        dx_sum += dx
        dy_sum += dy
    
    avg_dx = dx_sum / len(good_matches)
    avg_dy = dy_sum / len(good_matches)
    
    return avg_dx, avg_dy


def estimate_heading(equirect_img):
    """Estimate the forward heading direction from optical flow patterns.
    For a 360 camera moving forward, the expansion point in the equirect image
    indicates the heading direction.
    Returns heading angle in degrees (0-360).
    """
    # The heading is roughly where the most "expansion" happens in optical flow
    # For simplicity, return 0 (front of equirectangular image)
    return 0


def build_trajectory(panorama_files, panorama_dir):
    """Build camera trajectory from sequential panorama frames."""
    positions = [(0, 0)]  # Start at origin
    headings = [0]
    
    prev_img = None
    
    for i, fname in enumerate(panorama_files):
        img_path = os.path.join(panorama_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Resize for faster processing
        small = cv2.resize(img, (960, 480))
        
        if prev_img is not None:
            # Estimate angular displacement between frames
            dx, dy = estimate_movement(prev_img, small)
            
            # Camera rotation translates to apparent motion in equirect
            # dx (yaw change) tells us rotation
            # The actual walking direction is roughly the heading
            
            # Assume constant speed walking, direction changes based on rotation
            heading = headings[-1] - dx  # Yaw change
            headings.append(heading)
            
            # Step forward in heading direction
            step = 30  # pixels per step on floor plan
            x = positions[-1][0] + step * math.sin(math.radians(heading))
            y = positions[-1][1] - step * math.cos(math.radians(heading))
            positions.append((x, y))
        
        prev_img = small
        print(f"  Processed frame {i+1}/{len(panorama_files)}")
    
    return positions, headings


def create_floorplan_image(nadir_views, positions, canvas_size=800, 
                           hotspot_radius=12, path_color=(79, 195, 247)):
    """Create a floor plan image with nadir views, trajectory, and hotspot markers."""
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[:] = (30, 26, 26)  # Dark background (#1a1a1e)
    
    if not positions:
        return canvas
    
    # Normalize positions to fit canvas
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    range_x = max(max_x - min_x, 1)
    range_y = max(max_y - min_y, 1)
    scale = min((canvas_size - 160) / range_x, (canvas_size - 160) / range_y)
    
    cx, cy = canvas_size // 2, canvas_size // 2
    
    normalized = []
    for x, y in positions:
        nx = int(cx + (x - (min_x + max_x) / 2) * scale)
        ny = int(cy + (y - (min_y + max_y) / 2) * scale)
        normalized.append((nx, ny))
    
    # Draw nadir thumbnails at positions (small, semi-transparent)
    thumb_size = 40
    for i, (nx, ny) in enumerate(normalized):
        if i < len(nadir_views) and nadir_views[i] is not None:
            thumb = cv2.resize(nadir_views[i], (thumb_size, thumb_size))
            x1 = max(0, nx - thumb_size // 2)
            y1 = max(0, ny - thumb_size // 2)
            x2 = min(canvas_size, x1 + thumb_size)
            y2 = min(canvas_size, y1 + thumb_size)
            tw = x2 - x1
            th = y2 - y1
            if tw > 0 and th > 0:
                region = canvas[y1:y2, x1:x2]
                thumb_crop = thumb[:th, :tw]
                if region.shape == thumb_crop.shape:
                    blended = cv2.addWeighted(region, 0.5, thumb_crop, 0.5, 0)
                    canvas[y1:y2, x1:x2] = blended
    
    # Draw path
    for i in range(len(normalized) - 1):
        pt1 = normalized[i]
        pt2 = normalized[i + 1]
        cv2.line(canvas, pt1, pt2, path_color, 2, cv2.LINE_AA)
    
    # Draw hotspot dots
    for i, (nx, ny) in enumerate(normalized):
        # Outer glow
        cv2.circle(canvas, (nx, ny), hotspot_radius + 4, path_color, 1, cv2.LINE_AA)
        # Main dot
        cv2.circle(canvas, (nx, ny), hotspot_radius, path_color, -1, cv2.LINE_AA)
        # White border
        cv2.circle(canvas, (nx, ny), hotspot_radius, (255, 255, 255), 2, cv2.LINE_AA)
        # Number
        text = str(i + 1)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        tx = nx - text_size[0] // 2
        ty = ny + text_size[1] // 2
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   (255, 255, 255), 1, cv2.LINE_AA)
    
    # Title
    cv2.putText(canvas, "Floor Plan", (canvas_size // 2 - 50, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    return canvas, normalized


def generate_tour_config(positions, panorama_files, canvas_size=800):
    """Generate JSON configuration for the virtual tour viewer."""
    config = {
        "panoramas": [],
        "canvasSize": canvas_size,
        "floorPlanImage": "floorplan.jpg"
    }
    
    for i, fname in enumerate(panorama_files):
        pano = {
            "id": f"pano_{i+1:03d}",
            "src": f"panoramas/{fname}",
            "label": f"Viewpoint {i+1}",
            "x": positions[i][0] if i < len(positions) else 0,
            "y": positions[i][1] if i < len(positions) else 0
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
    
    # Get sorted panorama files
    panorama_files = sorted([f for f in os.listdir(panorama_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Found {len(panorama_files)} panorama frames")
    
    # Step 1: Extract nadir views
    print("\nStep 1: Extracting nadir views...")
    nadir_views = []
    for fname in panorama_files:
        img = cv2.imread(os.path.join(panorama_dir, fname))
        if img is not None:
            nadir = extract_nadir_view(img, size=256)
            nadir_views.append(nadir)
            nadir_path = os.path.join(output_dir, f"nadir_{fname}")
            cv2.imwrite(nadir_path, nadir)
        else:
            nadir_views.append(None)
    
    # Step 2: Build camera trajectory
    print("\nStep 2: Estimating camera trajectory...")
    positions, headings = build_trajectory(panorama_files, panorama_dir)
    
    # Step 3: Create floor plan image
    print("\nStep 3: Creating floor plan...")
    floorplan, normalized_pos = create_floorplan_image(
        nadir_views, positions, canvas_size=800
    )
    floorplan_path = os.path.join(output_dir, "floorplan.jpg")
    cv2.imwrite(floorplan_path, floorplan, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Floor plan saved: {floorplan_path}")
    
    # Step 4: Generate tour config
    print("\nStep 4: Generating tour config...")
    config = generate_tour_config(normalized_pos, panorama_files)
    config_path = os.path.join(output_dir, "tour_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {config_path}")
    
    print(f"\nDone! Files saved to {output_dir}")
    print(f"  - floorplan.jpg (floor plan with trajectory)")
    print(f"  - tour_config.json (tour configuration)")
    print(f"  - nadir_*.jpg (nadir views)")


if __name__ == '__main__':
    main()
