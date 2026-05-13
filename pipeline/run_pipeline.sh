#!/bin/bash
# Captures 3D — Video to Gaussian Splat Pipeline (100% open-source)
# Uses: ffmpeg + COLMAP + OpenSplat
# Works on CPU (no GPU required)

set -e

# === Configuration ===
VIDEO_PATH="${1:?Usage: ./run_pipeline.sh <video_path> [output.ply] [fps] [iterations]}"
OUTPUT="${2:-splat.ply}"
FPS="${3:-1}"
ITERATIONS="${4:-2000}"
DOWNSCALE="${5:-4}"

WORK_DIR="/tmp/captures3d_work"
FRAMES_DIR="$WORK_DIR/frames"
COLMAP_DIR="$WORK_DIR/colmap"
COLMAP_DB="$COLMAP_DIR/database.db"

export QT_QPA_PLATFORM=offscreen

echo "============================================"
echo "  Captures 3D — Video to 3D Pipeline"
echo "============================================"
echo "Video:      $VIDEO_PATH"
echo "Output:     $OUTPUT"
echo "FPS:        $FPS"
echo "Iterations: $ITERATIONS"
echo "Downscale:  $DOWNSCALE"
echo "============================================"

# Clean previous run
rm -rf "$WORK_DIR"
mkdir -p "$FRAMES_DIR" "$COLMAP_DIR/sparse" "$COLMAP_DIR/images"

# === Step 1: Extract Frames ===
echo ""
echo "📽️  Step 1/4: Extracting frames at ${FPS} fps..."
START=$(date +%s)
ffmpeg -i "$VIDEO_PATH" -vf "fps=$FPS" -q:v 2 "$FRAMES_DIR/frame_%05d.jpg" -y 2>&1 | tail -3
NUM_FRAMES=$(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l)
END=$(date +%s)
echo "✅ Extracted $NUM_FRAMES frames in $((END-START))s"

if [ "$NUM_FRAMES" -lt 3 ]; then
    echo "❌ Too few frames ($NUM_FRAMES). Try higher FPS or longer video."
    exit 1
fi

# Copy frames to colmap images dir
cp "$FRAMES_DIR"/*.jpg "$COLMAP_DIR/images/"

# === Step 2: COLMAP Feature Extraction ===
echo ""
echo "📐 Step 2/4: COLMAP feature extraction..."
START=$(date +%s)
colmap feature_extractor \
    --database_path "$COLMAP_DB" \
    --image_path "$COLMAP_DIR/images" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu 0 \
    2>&1 | grep -E "Processed|Timer|image" | tail -5
END=$(date +%s)
echo "✅ Feature extraction done in $((END-START))s"

# === Step 3: COLMAP Matching + Reconstruction ===
echo ""
echo "🔗 Step 3/4: COLMAP matching & reconstruction..."
START=$(date +%s)

# Sequential matching (fast for video)
colmap sequential_matcher \
    --database_path "$COLMAP_DB" \
    --SiftMatching.use_gpu 0 \
    2>&1 | grep -E "Processed|Timer|match" | tail -5

# Sparse reconstruction
mkdir -p "$COLMAP_DIR/sparse/0"
colmap mapper \
    --database_path "$COLMAP_DB" \
    --image_path "$COLMAP_DIR/images" \
    --output_path "$COLMAP_DIR/sparse" \
    2>&1 | grep -E "Registering|Bundle|Timer|image" | tail -10

# Check if reconstruction succeeded
if [ ! -f "$COLMAP_DIR/sparse/0/cameras.bin" ]; then
    echo "❌ COLMAP reconstruction failed. Try with more frames or better video."
    exit 1
fi

# Undistort images
colmap image_undistorter \
    --image_path "$COLMAP_DIR/images" \
    --input_path "$COLMAP_DIR/sparse/0" \
    --output_path "$COLMAP_DIR/undistorted" \
    2>&1 | tail -3

END=$(date +%s)
echo "✅ COLMAP reconstruction done in $((END-START))s"

# Count registered cameras
NUM_CAMERAS=$(colmap model_analyzer --path "$COLMAP_DIR/sparse/0" 2>&1 | grep "Registered images" | awk '{print $3}' || echo "?")
echo "   Registered cameras: $NUM_CAMERAS"

# === Step 4: OpenSplat Training ===
echo ""
echo "🧠 Step 4/4: Training Gaussian Splatting ($ITERATIONS iterations)..."
echo "   This may take a while on CPU..."
START=$(date +%s)

OPENSPLAT="/home/ubuntu/OpenSplat/build/opensplat"
$OPENSPLAT "$COLMAP_DIR/undistorted/sparse" \
    --output "$OUTPUT" \
    --num-iters "$ITERATIONS" \
    --downscale-factor "$DOWNSCALE" \
    --num-downscales 0 \
    --cpu \
    2>&1 | grep -E "Step|Loss|Timer|Saving" | tail -20

END=$(date +%s)
echo "✅ Training done in $((END-START))s"

# === Done ===
echo ""
echo "============================================"
echo "  ✅ Pipeline Complete!"
echo "============================================"
if [ -f "$OUTPUT" ]; then
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo "Output: $OUTPUT ($SIZE)"
    echo ""
    echo "View it at: https://captures-3d-ognvwkqc.devinapps.com"
    echo "  → Click upload icon → Splat File tab → drag & drop $OUTPUT"
else
    echo "❌ Output file not found"
    exit 1
fi
