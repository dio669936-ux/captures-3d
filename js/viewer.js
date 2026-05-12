import * as GaussianSplats3D from 'https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.min.js';

export class SplatViewer {
  constructor(container) {
    this.container = container;
    this.viewer = null;
    this.flyMode = false;
    this.flySpeed = 2.0;
    this.flyKeys = {};
    this.viewpoints = [];
    this.animating = false;
    this.onLoadProgress = null;
    this.onLoadComplete = null;
    this.onFpsUpdate = null;
    this._fpsFrames = 0;
    this._fpsLastTime = 0;
    this._animLoopId = null;
  }

  init() {
    this._setupKeyboardControls();
  }

  _setupKeyboardControls() {
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      this.flyKeys[e.key.toLowerCase()] = true;
    });
    window.addEventListener('keyup', (e) => {
      this.flyKeys[e.key.toLowerCase()] = false;
    });
  }

  _startFpsCounter() {
    if (this._animLoopId) cancelAnimationFrame(this._animLoopId);
    const tick = (time) => {
      this._fpsFrames++;
      if (time - this._fpsLastTime >= 1000) {
        if (this.onFpsUpdate) this.onFpsUpdate(this._fpsFrames);
        this._fpsFrames = 0;
        this._fpsLastTime = time;
      }
      this._animLoopId = requestAnimationFrame(tick);
    };
    this._animLoopId = requestAnimationFrame(tick);
  }

  async loadScene(url, options = {}) {
    if (this.viewer) {
      try {
        this.viewer.dispose();
      } catch (e) {
        // ignore
      }
      this.container.innerHTML = '';
      this.viewer = null;
    }

    if (this.onLoadProgress) this.onLoadProgress(10, 'Initializing viewer...');

    try {
      const cameraUp = options.cameraUp || [0, -1, -0.54];
      const cameraPos = options.cameraPosition || [1, -1, 2];
      const cameraLookAt = options.cameraTarget || [0, 0, 0];

      this.viewer = new GaussianSplats3D.Viewer({
        cameraUp,
        initialCameraPosition: cameraPos,
        initialCameraLookAt: cameraLookAt,
        rootElement: this.container,
        selfDrivenMode: true,
        useBuiltInControls: true,
        dynamicScene: false,
        sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
        sharedMemoryForWorkers: false,
        integerBasedSort: true,
        gpuAcceleratedSort: true,
      });

      if (this.onLoadProgress) this.onLoadProgress(30, 'Loading splat data...');

      const sceneOptions = {
        splatAlphaRemovalThreshold: 5,
        showLoadingUI: false,
        progressiveLoad: true,
      };

      if (options.position) sceneOptions.position = options.position;
      if (options.rotation) sceneOptions.rotation = options.rotation;
      if (options.scale) {
        const s = typeof options.scale === 'number'
          ? [options.scale, options.scale, options.scale]
          : options.scale;
        sceneOptions.scale = s;
      }

      await this.viewer.addSplatScene(url, sceneOptions);

      if (this.onLoadProgress) this.onLoadProgress(90, 'Rendering scene...');

      this.viewer.start();
      this._startFpsCounter();

      if (this.onLoadProgress) this.onLoadProgress(100, 'Scene loaded');

      await new Promise(resolve => setTimeout(resolve, 800));
      if (this.onLoadComplete) this.onLoadComplete();

      return true;
    } catch (error) {
      console.error('Failed to load scene:', error);
      if (this.onLoadProgress) this.onLoadProgress(0, `Error: ${error.message}`);
      return false;
    }
  }

  async loadLocalFile(file) {
    const url = URL.createObjectURL(file);
    return this.loadScene(url, {
      cameraPosition: [1, -1, 2],
      cameraTarget: [0, 0, 0],
    });
  }

  setFlyMode(enabled) {
    this.flyMode = enabled;
  }

  resetCamera() {
    if (this.viewer) {
      // GaussianSplats3D viewer uses its own controls
      // Reset by reinitializing camera
      const cam = this.viewer.camera;
      if (cam) {
        cam.position.set(1, -1, 2);
        cam.lookAt(0, 0, 0);
      }
    }
  }

  addViewpoint(name, room, position, target) {
    const vp = { name, room, position, target };
    this.viewpoints.push(vp);
    return vp;
  }

  goToViewpoint(index) {
    const vp = this.viewpoints[index];
    if (vp && this.viewer && this.viewer.camera) {
      const cam = this.viewer.camera;
      cam.position.set(vp.position.x, vp.position.y, vp.position.z);
      cam.lookAt(vp.target.x, vp.target.y, vp.target.z);
    }
  }

  dispose() {
    if (this._animLoopId) cancelAnimationFrame(this._animLoopId);
    if (this.viewer) {
      try {
        this.viewer.dispose();
      } catch (e) {
        // ignore
      }
    }
  }
}
