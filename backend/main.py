"""Captures 3D Backend — Video to Gaussian Splat pipeline API."""

import asyncio
import os
import shutil
import subprocess
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DATA_DIR = Path(os.environ.get("CAPTURES_DATA_DIR", "/tmp/captures-3d-data"))
UPLOAD_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"
OUTPUT_DIR = DATA_DIR / "output"

for d in [UPLOAD_DIR, JOBS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Captures 3D API",
    description="Video/images to 3D Gaussian Splat pipeline",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING_FRAMES = "extracting_frames"
    PROCESSING_POSES = "processing_poses"
    TRAINING = "training"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = 0
    message: str = ""
    created_at: str = ""
    input_filename: str = ""
    output_url: str | None = None
    error: str | None = None


# In-memory job store (replace with DB for production)
jobs: dict[str, JobInfo] = {}


@app.get("/api/health")
async def health():
    """Health check and GPU availability."""
    gpu_available = _check_gpu()
    nerfstudio_available = _check_nerfstudio()
    colmap_available = _check_colmap()
    ffmpeg_available = _check_ffmpeg()
    return {
        "status": "ok",
        "gpu": gpu_available,
        "nerfstudio": nerfstudio_available,
        "colmap": colmap_available,
        "ffmpeg": ffmpeg_available,
    }


@app.post("/api/upload/video", response_model=JobInfo)
async def upload_video(
    file: UploadFile = File(...),
    max_iterations: int = 7000,
    target_fps: int = 2,
):
    """Upload a video and start the full pipeline: extract → COLMAP → train → export."""
    if not file.filename:
        raise HTTPException(400, "No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(400, f"Unsupported video format: {ext}")

    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    video_path = job_dir / f"input{ext}"
    async with aiofiles.open(video_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    job = JobInfo(
        job_id=job_id,
        status=JobStatus.QUEUED,
        progress=0,
        message="Video uploaded, starting pipeline...",
        created_at=datetime.utcnow().isoformat(),
        input_filename=file.filename,
    )
    jobs[job_id] = job

    asyncio.create_task(_run_pipeline(job_id, video_path, target_fps, max_iterations))

    return job


@app.post("/api/upload/images", response_model=JobInfo)
async def upload_images(
    files: list[UploadFile] = File(...),
    max_iterations: int = 7000,
):
    """Upload images and start the pipeline: COLMAP → train → export."""
    if not files:
        raise HTTPException(400, "No files provided")

    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    images_dir = job_dir / "images"
    images_dir.mkdir(parents=True)

    for i, f in enumerate(files):
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        img_path = images_dir / f"img_{i:04d}{ext}"
        async with aiofiles.open(img_path, "wb") as out:
            content = await f.read()
            await out.write(content)

    image_count = len(list(images_dir.iterdir()))
    if image_count == 0:
        shutil.rmtree(job_dir)
        raise HTTPException(400, "No valid image files uploaded")

    job = JobInfo(
        job_id=job_id,
        status=JobStatus.QUEUED,
        progress=0,
        message=f"{image_count} images uploaded, starting pipeline...",
        created_at=datetime.utcnow().isoformat(),
        input_filename=f"{image_count} images",
    )
    jobs[job_id] = job

    asyncio.create_task(
        _run_pipeline_from_images(job_id, images_dir, max_iterations)
    )

    return job


@app.get("/api/jobs/{job_id}", response_model=JobInfo)
async def get_job(job_id: str):
    """Get job status and progress."""
    if job_id not in jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return jobs[job_id]


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    return list(jobs.values())


@app.get("/api/scenes/{job_id}/scene.splat")
async def get_scene(job_id: str):
    """Download the generated .splat file."""
    splat_path = OUTPUT_DIR / job_id / "scene.splat"
    if not splat_path.exists():
        ply_path = OUTPUT_DIR / job_id / "splat.ply"
        if ply_path.exists():
            return FileResponse(ply_path, media_type="application/octet-stream")
        raise HTTPException(404, "Scene not ready yet")
    return FileResponse(splat_path, media_type="application/octet-stream")


# Mount output directory for static file serving
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


async def _run_pipeline(
    job_id: str, video_path: Path, target_fps: int, max_iterations: int
):
    """Full pipeline: video → frames → COLMAP → train → export."""
    job = jobs[job_id]
    job_dir = JOBS_DIR / job_id

    try:
        # Step 1: Extract frames
        job.status = JobStatus.EXTRACTING_FRAMES
        job.progress = 5
        job.message = "Extracting frames from video..."

        frames_dir = job_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        await _run_cmd([
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps={target_fps}",
            "-q:v", "2",
            str(frames_dir / "frame_%05d.jpg"),
        ])

        frame_count = len(list(frames_dir.glob("*.jpg")))
        job.progress = 20
        job.message = f"Extracted {frame_count} frames"

        # Step 2: Process with COLMAP (via Nerfstudio)
        await _run_colmap_and_train(job_id, frames_dir, max_iterations)

    except Exception as e:
        job.status = JobStatus.FAILED
        job.progress = 0
        job.error = str(e)
        job.message = f"Pipeline failed: {e}"


async def _run_pipeline_from_images(
    job_id: str, images_dir: Path, max_iterations: int
):
    """Pipeline from images: COLMAP → train → export."""
    try:
        await _run_colmap_and_train(job_id, images_dir, max_iterations)
    except Exception as e:
        job = jobs[job_id]
        job.status = JobStatus.FAILED
        job.progress = 0
        job.error = str(e)
        job.message = f"Pipeline failed: {e}"


async def _run_colmap_and_train(
    job_id: str, images_dir: Path, max_iterations: int
):
    """Shared pipeline: COLMAP poses → splatfacto train → export .splat."""
    job = jobs[job_id]
    job_dir = JOBS_DIR / job_id
    processed_dir = job_dir / "processed"

    # Step 2: COLMAP camera poses
    job.status = JobStatus.PROCESSING_POSES
    job.progress = 25
    job.message = "Computing camera poses with COLMAP..."

    await _run_cmd([
        "ns-process-data", "images",
        "--data", str(images_dir),
        "--output-dir", str(processed_dir),
    ])

    job.progress = 40
    job.message = "Camera poses computed"

    # Step 3: Train splatfacto
    job.status = JobStatus.TRAINING
    job.progress = 45
    job.message = "Training Gaussian Splatting model..."

    model_dir = job_dir / "model"
    await _run_cmd([
        "ns-train", "splatfacto",
        "--data", str(processed_dir),
        "--output-dir", str(model_dir),
        "--max-num-iterations", str(max_iterations),
        "--steps-per-eval-image", "500",
        "--pipeline.model.num-downscales", "2",
    ])

    job.progress = 80
    job.message = "Training complete, exporting..."

    # Step 4: Export to .splat
    job.status = JobStatus.EXPORTING
    job.progress = 85
    job.message = "Exporting to .splat format..."

    output_job_dir = OUTPUT_DIR / job_id
    output_job_dir.mkdir(parents=True, exist_ok=True)

    # Find the config file
    config_files = list(model_dir.rglob("config.yml"))
    if not config_files:
        raise RuntimeError("No config.yml found after training")

    config_path = sorted(config_files, key=lambda p: p.stat().st_mtime)[-1]

    await _run_cmd([
        "ns-export", "gaussian-splat",
        "--load-config", str(config_path),
        "--output-dir", str(output_job_dir),
    ])

    job.progress = 100
    job.status = JobStatus.COMPLETED
    job.message = "Scene ready!"
    job.output_url = f"/api/scenes/{job_id}/scene.splat"


async def _run_cmd(cmd: list[str], timeout: int = 3600):
    """Run a subprocess command asynchronously."""
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        process.kill()
        raise RuntimeError(f"Command timed out: {' '.join(cmd)}")

    if process.returncode != 0:
        error_msg = stderr.decode()[-500:] if stderr else "Unknown error"
        raise RuntimeError(f"Command failed ({process.returncode}): {error_msg}")

    return stdout.decode() if stdout else ""


def _check_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_nerfstudio() -> bool:
    try:
        result = subprocess.run(
            ["ns-train", "--help"], capture_output=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_colmap() -> bool:
    try:
        result = subprocess.run(
            ["colmap", "--help"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_ffmpeg() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
