"""Cut equirectangular panoramas into perspective crops for COLMAP."""
import os, sys, math, numpy as np
from PIL import Image

def equirect_to_perspective(equirect, fov_deg, yaw_deg, pitch_deg, out_size=800):
    """Extract a perspective view from an equirectangular image."""
    h, w = equirect.shape[:2]
    fov = math.radians(fov_deg)
    f = out_size / (2 * math.tan(fov / 2))
    
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    
    # Rotation matrices
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    
    R = np.array([
        [cy, -sy*sp, sy*cp],
        [0,   cp,    sp],
        [-sy, -cy*sp, cy*cp]
    ])
    
    # Create pixel grid
    u = np.arange(out_size) - out_size/2
    v = np.arange(out_size) - out_size/2
    uu, vv = np.meshgrid(u, v)
    
    # Direction vectors
    dirs = np.stack([uu, vv, np.full_like(uu, f)], axis=-1).astype(np.float64)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = dirs @ R.T
    
    # To spherical
    lon = np.arctan2(dirs[...,0], dirs[...,2])
    lat = np.arcsin(np.clip(dirs[...,1], -1, 1))
    
    # To equirect pixel coords
    px = (lon / (2*math.pi) + 0.5) * w
    py = (lat / math.pi + 0.5) * h
    
    px = np.clip(px, 0, w-1).astype(np.int32)
    py = np.clip(py, 0, h-1).astype(np.int32)
    
    return equirect[py, px]

def process_panoramas(input_dir, output_dir, crop_size=800):
    os.makedirs(output_dir, exist_ok=True)
    
    # 8 horizontal views (45° apart) + 2 vertical views
    views = [
        ("front", 0, 0),
        ("front_right", 45, 0),
        ("right", 90, 0),
        ("back_right", 135, 0),
        ("back", 180, 0),
        ("back_left", -135, 0),
        ("left", -90, 0),
        ("front_left", -45, 0),
    ]
    fov = 90  # degrees
    
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    print(f"Processing {len(files)} panoramas...")
    
    count = 0
    for fi, fname in enumerate(files):
        print(f"  [{fi+1}/{len(files)}] {fname}")
        img = np.array(Image.open(os.path.join(input_dir, fname)))
        
        # Resize for faster processing
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((4096, 2048), Image.LANCZOS)
        img = np.array(pil_img)
        
        base = os.path.splitext(fname)[0]
        for vname, yaw, pitch in views:
            crop = equirect_to_perspective(img, fov, yaw, pitch, crop_size)
            out_name = f"{base}_{vname}.jpg"
            Image.fromarray(crop).save(os.path.join(output_dir, out_name), quality=90)
            count += 1
    
    print(f"Done! Created {count} perspective crops in {output_dir}")

if __name__ == "__main__":
    input_dir = "/home/ubuntu/new_house/Thư mục không có tiêu đề/"
    output_dir = "/home/ubuntu/gs_pipeline/images"
    process_panoramas(input_dir, output_dir, crop_size=800)
