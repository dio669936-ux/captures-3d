# Captures 3D

Immersive 3D experience viewer powered by Gaussian Splatting. Create virtual tours from photos or video with free-move 3D navigation.

Inspired by [Captures Studio](https://www.captures.studio/).

## Features

- **3D Gaussian Splat Viewer** — Real-time rendering via [Spark](https://github.com/sparkjsdev/spark) (Three.js)
- **Free-Move Navigation** — Orbit and fly modes with smooth camera transitions
- **Viewpoint System** — Predefined viewpoints for guided tour navigation
- **Multi-Format Support** — Load `.ply`, `.splat`, `.ksplat`, `.spz` files
- **Upload & URL Loading** — Drag-and-drop files or load from URL
- **Demo Scenes** — Built-in sample scenes to explore
- **Responsive UI** — Works on desktop and mobile

## Quick Start

The viewer is a static web app — just serve the files:

```bash
# Option 1: Python
python -m http.server 8080

# Option 2: Node
npx serve .

# Option 3: Any static file server
```

Open [http://localhost:8080](http://localhost:8080) in your browser.

## Architecture

```
captures-3d/
├── index.html           # Main app entry
├── css/style.css        # UI styling
├── js/
│   ├── app.js           # App logic, UI, demo scenes
│   └── viewer.js        # Spark/Three.js viewer wrapper
├── pipeline/            # GPU training pipeline (Nerfstudio)
│   ├── process_video.py # Video → frames extraction
│   ├── train.py         # Nerfstudio splatfacto training
│   ├── export.py        # Export to .splat for web viewer
│   └── README.md        # Pipeline setup guide
└── README.md
```

## Creating Your Own 3D Scenes

### End-to-End Pipeline

1. **Capture** — Record 30-60s video or take 20-120 photos of your space
2. **Process** — Extract frames and compute camera poses (COLMAP)
3. **Train** — Run Gaussian Splatting training (Nerfstudio splatfacto)
4. **View** — Load the exported `.splat` file in the web viewer

See [pipeline/README.md](pipeline/README.md) for detailed instructions.

### Quick Commands

```bash
# From video
python pipeline/process_video.py -i room.mp4 -o data/room --fps 2
ns-process-data images --data data/room/frames --output-dir data/room/processed
python pipeline/train.py -d data/room/processed -o output/room --quick
python pipeline/export.py -m output/room -o output/room/scene.splat
```

## Controls

### Orbit Mode (Default)
- **Left click + drag**: Rotate around scene
- **Right click + drag**: Pan camera
- **Scroll**: Zoom in/out

### Fly Mode (WASD)
- **W/S**: Move forward/backward
- **A/D**: Strafe left/right
- **Q/E**: Move down/up
- **Mouse**: Look around

## Tech Stack

- **Viewer**: [Spark](https://github.com/sparkjsdev/spark) (Three.js Gaussian Splatting renderer)
- **3D Engine**: [Three.js](https://threejs.org/) r180
- **Training**: [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) (splatfacto)
- **SfM**: [COLMAP](https://github.com/colmap/colmap) (camera pose estimation)

## Roadmap

- [x] Phase 1: Web viewer with free-move 3D navigation
- [x] Phase 1: Upload/URL scene loading
- [x] Phase 1: Viewpoint navigation system
- [ ] Phase 2: Room-to-room transition system
- [ ] Phase 2: Minimap with room overview
- [ ] Phase 2: Scene editor (viewpoint placement)
- [ ] Phase 3: Navigation graph (edges between viewpoints)
- [ ] Phase 3: Guided tour mode with transitions
- [ ] Phase 3: Labels and annotations

## License

MIT
