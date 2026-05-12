import { SplatViewer } from './viewer.js';

const DEMO_SCENES = [
  {
    name: 'Nike Shoe',
    desc: 'Product scan (~8MB)',
    icon: '\u{1F45F}',
    url: 'https://huggingface.co/cakewalk/splat-data/resolve/main/nike.splat',
    cameraPosition: [0, -1, 3],
    cameraTarget: [0, 0, 0],
    cameraUp: [0, -1, -0.54],
  },
  {
    name: 'Plush Toy',
    desc: 'Object scan (~9MB)',
    icon: '\u{1F9F8}',
    url: 'https://huggingface.co/cakewalk/splat-data/resolve/main/plush.splat',
    cameraPosition: [0, -1, 3],
    cameraTarget: [0, 0, 0],
    cameraUp: [0, -1, -0.54],
  },
  {
    name: 'Train',
    desc: 'Outdoor train (~33MB)',
    icon: '\u{1F682}',
    url: 'https://huggingface.co/cakewalk/splat-data/resolve/main/train.splat',
    cameraPosition: [-3, -2, 2],
    cameraTarget: [0, 0, 0],
    cameraUp: [0, -1, -0.54],
  },
  {
    name: 'Room',
    desc: 'Indoor room (~51MB)',
    icon: '\u{1F3E0}',
    url: 'https://huggingface.co/cakewalk/splat-data/resolve/main/room.splat',
    cameraPosition: [-3, -1, -1],
    cameraTarget: [0, 0, 0],
    cameraUp: [0, -1, -0.54],
  },
];

const PIPELINE_STEPS = [
  { key: 'extracting_frames', label: 'Extracting frames from video' },
  { key: 'processing_poses', label: 'Computing camera poses (COLMAP)' },
  { key: 'training', label: 'Training Gaussian Splatting model' },
  { key: 'exporting', label: 'Exporting .splat file' },
  { key: 'completed', label: 'Scene ready' },
];

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? `${window.location.protocol}//${window.location.hostname}:8000`
  : '/api';

class App {
  constructor() {
    this.viewer = null;
    this.currentScene = null;
    this.pollingInterval = null;
    this.init();
  }

  init() {
    const container = document.getElementById('viewer-container');
    this.viewer = new SplatViewer(container);

    this.viewer.onLoadProgress = (percent, message) => {
      this._updateLoading(percent, message);
    };
    this.viewer.onLoadComplete = () => {
      this._hideLoading();
    };
    this.viewer.onFpsUpdate = (fps) => {
      this._updateFps(fps);
    };

    this.viewer.init();
    this._setupUI();
    this._loadDefaultScene();
  }

  _setupUI() {
    this._setupUploadModal();
    this._setupVideoUpload();
    this._setupControls();
    this._setupInfoPanel();
    this._setupViewpointsPanel();
    this._setupDemoScenes();
  }

  _setupUploadModal() {
    const modal = document.getElementById('upload-modal');
    const btnUpload = document.getElementById('btn-upload');
    const btnClose = document.getElementById('btn-close-upload');
    const backdrop = modal.querySelector('.modal-backdrop');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const urlInput = document.getElementById('url-input');
    const btnLoadUrl = document.getElementById('btn-load-url');

    btnUpload.addEventListener('click', () => {
      modal.classList.remove('hidden');
      this._checkBackendHealth();
    });
    btnClose.addEventListener('click', () => modal.classList.add('hidden'));
    backdrop.addEventListener('click', () => modal.classList.add('hidden'));

    const tabs = modal.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        modal.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
      });
    });

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('drag-over');
    });
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file) this._handleFileUpload(file);
    });

    fileInput.addEventListener('change', () => {
      const file = fileInput.files[0];
      if (file) this._handleFileUpload(file);
    });

    btnLoadUrl.addEventListener('click', () => {
      const url = urlInput.value.trim();
      if (url) this._loadSceneFromUrl(url, 'Custom Scene');
    });

    urlInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const url = urlInput.value.trim();
        if (url) this._loadSceneFromUrl(url, 'Custom Scene');
      }
    });
  }

  _setupVideoUpload() {
    const videoDropZone = document.getElementById('video-drop-zone');
    const videoInput = document.getElementById('video-input');

    videoDropZone.addEventListener('click', () => videoInput.click());
    videoDropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      videoDropZone.classList.add('drag-over');
    });
    videoDropZone.addEventListener('dragleave', () => {
      videoDropZone.classList.remove('drag-over');
    });
    videoDropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      videoDropZone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file) this._handleVideoUpload(file);
    });

    videoInput.addEventListener('change', () => {
      const file = videoInput.files[0];
      if (file) this._handleVideoUpload(file);
    });
  }

  async _handleVideoUpload(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['mp4', 'mov', 'avi', 'mkv', 'webm'].includes(ext)) {
      alert('Unsupported video format. Use .mp4, .mov, .avi, .mkv, or .webm');
      return;
    }

    const quality = document.getElementById('train-quality').value;
    const fps = document.getElementById('train-fps').value;

    const formData = new FormData();
    formData.append('file', file);

    const params = new URLSearchParams({
      max_iterations: quality,
      target_fps: fps,
    });

    this._showPipelineStatus('Uploading video...');

    try {
      const response = await fetch(`${API_BASE}/api/upload/video?${params}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(err.detail || 'Upload failed');
      }

      const job = await response.json();
      this._updateSceneTitle(file.name.replace(/\.[^.]+$/, ''));
      this._pollJobStatus(job.job_id);
    } catch (error) {
      this._showPipelineError(error.message);
    }
  }

  _pollJobStatus(jobId) {
    if (this.pollingInterval) clearInterval(this.pollingInterval);

    this.pollingInterval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
        if (!response.ok) throw new Error('Failed to get job status');

        const job = await response.json();
        this._updatePipelineUI(job);

        if (job.status === 'completed') {
          clearInterval(this.pollingInterval);
          this.pollingInterval = null;
          this._hidePipelineStatus();
          const splatUrl = `${API_BASE}${job.output_url}`;
          await this.viewer.loadScene(splatUrl);
          document.getElementById('upload-modal').classList.add('hidden');
        } else if (job.status === 'failed') {
          clearInterval(this.pollingInterval);
          this.pollingInterval = null;
          this._showPipelineError(job.error || 'Pipeline failed');
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 2000);
  }

  _showPipelineStatus(message) {
    const status = document.getElementById('pipeline-status');
    const progress = document.getElementById('pipeline-progress');
    const msg = document.getElementById('pipeline-message');
    const steps = document.getElementById('pipeline-steps');

    status.classList.remove('hidden');
    progress.style.width = '5%';
    msg.textContent = message;

    steps.innerHTML = PIPELINE_STEPS.map(step => `
      <div class="pipeline-step" data-step="${step.key}">
        <div class="pipeline-step-dot"></div>
        <span>${step.label}</span>
      </div>
    `).join('');
  }

  _updatePipelineUI(job) {
    const progress = document.getElementById('pipeline-progress');
    const msg = document.getElementById('pipeline-message');

    progress.style.width = `${job.progress}%`;
    msg.textContent = job.message;

    const currentStatus = job.status;
    let passedCurrent = false;
    PIPELINE_STEPS.forEach(step => {
      const el = document.querySelector(`.pipeline-step[data-step="${step.key}"]`);
      if (!el) return;

      el.classList.remove('active', 'done', 'failed');
      if (step.key === currentStatus) {
        el.classList.add('active');
        passedCurrent = true;
      } else if (!passedCurrent) {
        el.classList.add('done');
      }
    });
  }

  _hidePipelineStatus() {
    document.getElementById('pipeline-status').classList.add('hidden');
  }

  _showPipelineError(message) {
    const msg = document.getElementById('pipeline-message');
    msg.textContent = `Error: ${message}`;
    msg.style.color = '#ef4444';

    const steps = document.querySelectorAll('.pipeline-step.active');
    steps.forEach(s => {
      s.classList.remove('active');
      s.classList.add('failed');
    });
  }

  async _checkBackendHealth() {
    const existing = document.querySelector('.backend-status');
    if (existing) existing.remove();

    try {
      const response = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) });
      if (!response.ok) throw new Error('Backend not responding');
      const data = await response.json();

      const statusDiv = document.createElement('div');
      statusDiv.className = `backend-status ${data.gpu ? 'ok' : 'error'}`;
      if (data.gpu) {
        statusDiv.textContent = 'GPU backend connected — video processing available';
      } else if (data.nerfstudio) {
        statusDiv.textContent = 'Backend connected (no GPU) — processing will be slow';
      } else {
        statusDiv.textContent = 'Backend connected — Nerfstudio not installed';
      }
      document.getElementById('tab-video').appendChild(statusDiv);
    } catch {
      const statusDiv = document.createElement('div');
      statusDiv.className = 'backend-status error';
      statusDiv.textContent = 'Backend not connected — start the API server to process videos';
      document.getElementById('tab-video').appendChild(statusDiv);
    }
  }

  _setupControls() {
    const btnOrbit = document.getElementById('btn-orbit');
    const btnFly = document.getElementById('btn-fly');
    const btnReset = document.getElementById('btn-reset-camera');
    const btnFullscreen = document.getElementById('btn-fullscreen');

    btnOrbit.addEventListener('click', () => {
      this.viewer.setFlyMode(false);
      btnOrbit.classList.add('active');
      btnFly.classList.remove('active');
    });

    btnFly.addEventListener('click', () => {
      this.viewer.setFlyMode(true);
      btnFly.classList.add('active');
      btnOrbit.classList.remove('active');
    });

    btnReset.addEventListener('click', () => {
      this.viewer.resetCamera();
    });

    btnFullscreen.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
      } else {
        document.exitFullscreen();
      }
    });
  }

  _setupInfoPanel() {
    const panel = document.getElementById('info-panel');
    const btnInfo = document.getElementById('btn-info');
    const btnClose = document.getElementById('btn-close-info');

    btnInfo.addEventListener('click', () => {
      panel.classList.toggle('hidden');
    });
    btnClose.addEventListener('click', () => {
      panel.classList.add('hidden');
    });
  }

  _setupViewpointsPanel() {
    const panel = document.getElementById('viewpoints-panel');
    const btnViewpoints = document.getElementById('btn-viewpoints');
    const btnClose = document.getElementById('btn-close-viewpoints');

    btnViewpoints.addEventListener('click', () => {
      panel.classList.toggle('visible');
    });
    btnClose.addEventListener('click', () => {
      panel.classList.remove('visible');
    });
  }

  _setupDemoScenes() {
    const grid = document.getElementById('demo-scenes');
    DEMO_SCENES.forEach((scene, index) => {
      const card = document.createElement('div');
      card.className = 'demo-scene-card';
      card.innerHTML = `
        <span class="scene-icon">${scene.icon}</span>
        <div class="scene-name">${scene.name}</div>
        <div class="scene-desc">${scene.desc}</div>
      `;
      card.addEventListener('click', () => {
        this._loadDemoScene(index);
      });
      grid.appendChild(card);
    });
  }

  async _loadDefaultScene() {
    this._updateLoading(5, 'Loading default scene...');
    await this._loadDemoScene(0);
  }

  async _loadDemoScene(index) {
    const scene = DEMO_SCENES[index];
    if (!scene) return;

    document.getElementById('upload-modal').classList.add('hidden');
    this._showLoading();
    this._updateSceneTitle(scene.name);
    this.currentScene = scene;

    const options = {
      cameraPosition: scene.cameraPosition,
      cameraTarget: scene.cameraTarget,
      cameraUp: scene.cameraUp,
      position: scene.position,
      rotation: scene.rotation,
      scale: scene.scale,
    };

    await this.viewer.loadScene(scene.url, options);
    this._updateInfoPanel(scene.name, this._getExtension(scene.url));
    this._setupDefaultViewpoints(scene);
  }

  async _loadSceneFromUrl(url, name) {
    document.getElementById('upload-modal').classList.add('hidden');
    this._showLoading();
    this._updateSceneTitle(name);

    await this.viewer.loadScene(url, {
      cameraPosition: [1, -1, 2],
      cameraTarget: [0, 0, 0],
    });
    this._updateInfoPanel(name, this._getExtension(url));
  }

  async _handleFileUpload(file) {
    document.getElementById('upload-modal').classList.add('hidden');
    this._showLoading();
    const name = file.name.replace(/\.[^.]+$/, '');
    this._updateSceneTitle(name);

    await this.viewer.loadLocalFile(file);
    this._updateInfoPanel(name, this._getExtension(file.name));
  }

  _setupDefaultViewpoints(scene) {
    this.viewer.viewpoints = [];
    const list = document.getElementById('viewpoints-list');
    list.innerHTML = '';

    const defaultPos = scene.cameraPosition || [1, -1, 2];
    const defaultTarget = scene.cameraTarget || [0, 0, 0];

    const viewpoints = [
      { name: 'Front View', room: 'Main', pos: defaultPos, target: defaultTarget },
      { name: 'Top View', room: 'Main', pos: [0, -5, 0.1], target: [0, 0, 0] },
      { name: 'Side View', room: 'Main', pos: [5, -1, 0], target: [0, 0, 0] },
      { name: 'Close Up', room: 'Main', pos: [0, -0.5, 1.5], target: [0, 0, 0] },
    ];

    viewpoints.forEach((vp, i) => {
      this.viewer.addViewpoint(
        vp.name,
        vp.room,
        { x: vp.pos[0], y: vp.pos[1], z: vp.pos[2] },
        { x: vp.target[0], y: vp.target[1], z: vp.target[2] },
      );

      const item = document.createElement('div');
      item.className = `viewpoint-item${i === 0 ? ' active' : ''}`;
      item.innerHTML = `
        <div class="viewpoint-dot"></div>
        <div>
          <div class="viewpoint-name">${vp.name}</div>
          <div class="viewpoint-room">${vp.room}</div>
        </div>
      `;
      item.addEventListener('click', () => {
        list.querySelectorAll('.viewpoint-item').forEach(el => el.classList.remove('active'));
        item.classList.add('active');
        this.viewer.goToViewpoint(i);
      });
      list.appendChild(item);
    });
  }

  _showLoading() {
    const screen = document.getElementById('loading-screen');
    screen.classList.remove('fade-out');
    screen.style.display = 'flex';
  }

  _hideLoading() {
    const screen = document.getElementById('loading-screen');
    screen.classList.add('fade-out');
    setTimeout(() => {
      screen.style.display = 'none';
    }, 500);
  }

  _updateLoading(percent, message) {
    const bar = document.getElementById('loading-bar');
    const status = document.getElementById('loading-status');
    if (bar) bar.style.width = `${percent}%`;
    if (status) status.textContent = message;
  }

  _updateSceneTitle(name) {
    document.getElementById('scene-title').textContent = name;
  }

  _updateFps(fps) {
    document.getElementById('fps-counter').textContent = `${fps} FPS`;
    const infoFps = document.getElementById('info-fps');
    if (infoFps) infoFps.textContent = `${fps}`;
  }

  _updateInfoPanel(name, format) {
    document.getElementById('info-scene-name').textContent = name;
    document.getElementById('info-format').textContent = format.toUpperCase();
  }

  _getExtension(path) {
    return path.split('.').pop().split('?')[0] || 'unknown';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.app = new App();
});
