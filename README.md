# Captures 3D

Immersive 3D experience viewer powered by Gaussian Splatting. Upload a video and get a navigable 3D scene automatically.

Inspired by [Captures Studio](https://www.captures.studio/).

## Features

- **Video to 3D** — Upload a video, get a navigable Gaussian Splat scene (requires GPU backend)
- **3D Gaussian Splat Viewer** — Real-time rendering via [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)
- **Free-Move Navigation** — Orbit and fly modes with smooth camera transitions
- **Viewpoint System** — Predefined viewpoints for guided tour navigation
- **Multi-Format Support** — Load `.ply`, `.splat`, `.ksplat` files
- **Upload & URL Loading** — Drag-and-drop files or load from URL
- **Demo Scenes** — Built-in sample Gaussian Splat scenes to explore
- **Responsive UI** — Dark theme, glassmorphism, works on desktop and mobile

## Quick Start

### Frontend Only (view existing .splat files)

```bash
cd captures-3d
python -m http.server 8080
```

Open [http://localhost:8080](http://localhost:8080) — use Demo Scenes, upload .splat files, or load from URL.

### Full Pipeline (video → 3D scene)

Requires a GPU machine with CUDA, COLMAP, and Nerfstudio installed.

```bash
# 1. Install backend
cd backend
pip install -e .
pip install -e ".[gpu]"  # includes nerfstudio, opencv, etc.

# 2. Start API server
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Serve frontend (separate terminal)
cd ..
python -m http.server 8080
```

Open http://localhost:8080, click "Load Scene" → "Video" tab, drop your video. The backend will:
1. Extract frames (ffmpeg)
2. Compute camera poses (COLMAP via ns-process-data)
3. Train Gaussian Splatting (Nerfstudio splatfacto)
4. Export .splat and load in viewer

### Manual Pipeline (CLI)

```bash
# From video
python pipeline/process_video.py -i room.mp4 -o data/room --fps 2
ns-process-data images --data data/room/frames --output-dir data/room/processed
python pipeline/train.py -d data/room/processed -o output/room --quick
python pipeline/export.py -m output/room -o output/room/scene.splat
```

Then load `scene.splat` in the web viewer via drag-and-drop or URL.

## Architecture

```
captures-3d/
├── index.html           # Main app entry
├── css/style.css        # UI styling (dark theme, glassmorphism)
├── js/
│   ├── app.js           # App logic, UI, video upload, pipeline status
│   └── viewer.js        # GaussianSplats3D viewer wrapper
├── backend/
│   ├── main.py          # FastAPI server (video → 3D pipeline)
│   └── pyproject.toml   # Backend dependencies
├── pipeline/            # CLI pipeline scripts
│   ├── process_video.py # Video → frames extraction
│   ├── train.py         # Nerfstudio splatfacto training
│   ├── export.py        # Export to .splat for web viewer
│   └── README.md        # Pipeline setup guide
└── README.md
```

## Controls

### Orbit Mode (Default)
- **Left click + drag**: Rotate around scene
- **Right click + drag**: Pan camera
- **Scroll**: Zoom in/out
- **Left click**: Set focal point

### Fly Mode (WASD)
- **W/S**: Move forward/backward
- **A/D**: Strafe left/right
- **Q/E**: Move down/up
- **Mouse**: Look around

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Backend status, GPU/COLMAP/Nerfstudio checks |
| `/api/upload/video` | POST | Upload video → start full pipeline |
| `/api/upload/images` | POST | Upload images → start pipeline |
| `/api/jobs/{id}` | GET | Get job status and progress |
| `/api/jobs` | GET | List all jobs |
| `/api/scenes/{id}/scene.splat` | GET | Download generated .splat file |

## Tech Stack

- **Viewer**: [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) (Three.js-based renderer)
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Python async API)
- **Training**: [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) (splatfacto method)
- **SfM**: [COLMAP](https://github.com/colmap/colmap) (camera pose estimation)
- **Video**: ffmpeg (frame extraction)

## GPU Requirements

For the training pipeline (video → 3D):
- NVIDIA GPU with CUDA 11.8+ and 8GB+ VRAM
- COLMAP installed (`apt install colmap`)
- ffmpeg installed
- Nerfstudio (`pip install nerfstudio`)

The web viewer itself runs on any modern browser with WebGL2 — no GPU needed for viewing.

## Roadmap

- [x] Phase 1: Web viewer with free-move 3D navigation
- [x] Phase 1: Upload/URL scene loading + demo scenes
- [x] Phase 1: Viewpoint navigation system
- [x] Phase 1: Video upload → backend pipeline API
- [ ] Phase 2: Room-to-room transition system
- [ ] Phase 2: Minimap with room overview
- [ ] Phase 2: Scene editor (viewpoint placement)
- [ ] Phase 3: Navigation graph (edges between viewpoints)
- [ ] Phase 3: Guided tour mode with transitions
- [ ] Phase 3: Labels and annotations

## License

MIT
