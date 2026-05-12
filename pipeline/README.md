# Captures 3D — Training Pipeline

This pipeline converts images or video into 3D Gaussian Splat scenes using Nerfstudio.

## Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3060+ or better)
- **CUDA**: 11.8 or 12.x
- **Python**: 3.10+
- **COLMAP**: Required for camera pose estimation

## Quick Setup

```bash
# 1. Install Nerfstudio (includes COLMAP integration)
pip install nerfstudio

# 2. Install this pipeline's dependencies
pip install -r requirements.txt

# 3. Verify installation
ns-train --help
```

## Usage

### From Video

```bash
# Step 1: Extract frames from video
python process_video.py --input room.mp4 --output data/room --fps 2

# Step 2: Process with COLMAP (camera poses)
ns-process-data images --data data/room/frames --output-dir data/room/processed

# Step 3: Train Gaussian Splatting model
python train.py --data data/room/processed --method splatfacto --output output/room

# Step 4: Export to .splat format for web viewer
python export.py --model output/room --output output/room/scene.splat
```

### From Images

```bash
# Step 1: Put images in a folder (20-120 images recommended)
# data/room/images/img_001.jpg, img_002.jpg, ...

# Step 2: Process with COLMAP
ns-process-data images --data data/room/images --output-dir data/room/processed

# Step 3: Train
python train.py --data data/room/processed --method splatfacto --output output/room

# Step 4: Export
python export.py --model output/room --output output/room/scene.splat
```

## Capture Tips

### Video Recording
- Duration: 30-60 seconds per room
- Move slowly, avoid fast motion
- Keep consistent lighting
- Avoid motion blur
- Cover the scene from multiple angles and heights

### Photo Collection
- Minimum: 20 images
- Recommended: 30-50 images
- Best: 50-120 images for complex scenes
- Overlap between consecutive images: ~70%
- Include different heights and angles
- Consistent lighting throughout

## Training Parameters

| Parameter | Quick Demo | Quality | High Quality |
|-----------|-----------|---------|-------------|
| `--max-num-iterations` | 7000 | 15000 | 30000 |
| `--steps-per-eval-image` | 500 | 250 | 100 |
| Training time (RTX 3060) | ~5 min | ~15 min | ~30 min |
| Training time (RTX 4090) | ~2 min | ~6 min | ~12 min |

## Output Files

After training, you'll find:
- `scene.splat` — Ready for web viewer (GaussianSplats3D / Spark)
- `scene.ply` — Full precision point cloud
- `transforms.json` — Camera poses and scene metadata

## Loading in Web Viewer

1. Open the Captures 3D web viewer
2. Click the Upload button
3. Choose "Local File" tab
4. Drop your `.splat` file
5. Navigate the 3D scene with mouse/keyboard
