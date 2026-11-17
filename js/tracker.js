import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm';
import { ColorTracker } from './color-tracker.js';

export class InputTracker {
  constructor(videoEl, stageEl, callbacks = {}) {
    this.videoEl = videoEl;
    this.stageEl = stageEl;
    this.callbacks = callbacks;

    this.mode = 'HAND'; // 'HAND' | 'COLOR' | 'TOUCH'
    this.mirrored = true; // 固定左右鏡像

    this.handLandmarker = null;
    this.visionFileset = null;
    this.modelInitPromise = null;
    this.activeDelegate = 'GPU';
    this.useCanvasInput = false; // 若 S25/Android 硬體視訊貼圖靜默回傳空結果，自動切換離屏 Canvas 輸入

    this.workCanvas = document.createElement('canvas');
    this.workCanvas.width = 320;
    this.workCanvas.height = 240;
    this.workCtx = this.workCanvas.getContext('2d', { willReadFrequently: true });

    this.colorTracker = new ColorTracker();

    this.stream = null;
    this.isRunning = false;
    this.isInferencing = false;
    this.inferTimer = null;
    this.lostFrames = 0;
    this.emptyHandFrames = 0;
    this.consecutiveErrors = 0;
    this.lastInferTs = 0;
    this.lastVideoTime = -1;
    this.hasHandNow = false;

    // 即時診斷數據
    this.diag = {
      engine: 'TOUCH',
      sentCount: 0,
      resultCount: 0,
      foundCount: 0,
      cursorUpdates: 0,
      errorCount: 0,
      lastError: '無',
      lastCoord: '-',
      lastReturnTime: '-',
    };

    this._initPointerEvents();
  }

  _emitDiag() {
    this.callbacks.onDiagnostics?.({ ...this.diag });
  }

  _recordError(msg) {
    this.diag.errorCount++;
    this.diag.lastError = String(msg || '未知錯誤').slice(0, 90);
    this._emitDiag();
  }

  // 共用 object-fit: cover 座標轉換（固定左右鏡像）
  mapVideoToStage(rawX, rawY) {
    const vw = this.videoEl?.videoWidth || 640;
    const vh = this.videoEl?.videoHeight || 480;
    const sw = this.stageEl?.clientWidth || window.innerWidth || 640;
    const sh = this.stageEl?.clientHeight || window.innerHeight || 480;

    const scale = Math.max(sw / vw, sh / vh);
    const renderedW = vw * scale;
    const renderedH = vh * scale;
    const offsetX = (renderedW - sw) / 2;
    const offsetY = (renderedH - sh) / 2;

    const mirroredX = this.mirrored ? 1.0 - rawX : rawX;
    const stageX = (mirroredX * renderedW - offsetX) / sw;
    const stageY = (rawY * renderedH - offsetY) / sh;

    return {
      x: Math.min(1, Math.max(0, stageX)),
      y: Math.min(1, Math.max(0, stageY)),
    };
  }

  // 反向轉換：點擊螢幕座標 (clientX, clientY) → 原始攝影機座標 (rawX, rawY in 0..1) 供顏色取樣
  mapScreenToVideo(clientX, clientY) {
    const rect = this.stageEl.getBoundingClientRect();
    const vw = this.videoEl?.videoWidth || 640;
    const vh = this.videoEl?.videoHeight || 480;
    const sw = rect.width || window.innerWidth || 640;
    const sh = rect.height || window.innerHeight || 480;

    const sx = (clientX - rect.left) / sw;
    const sy = (clientY - rect.top) / sh;

    const scale = Math.max(sw / vw, sh / vh);
    const renderedW = vw * scale;
    const renderedH = vh * scale;
    const offsetX = (renderedW - sw) / 2;
    const offsetY = (renderedH - sh) / 2;

    const mirroredX = (sx * sw + offsetX) / renderedW;
    const rawX = this.mirrored ? 1.0 - mirroredX : mirroredX;
    const rawY = (sy * sh + offsetY) / renderedH;

    return {
      rawX: Math.min(1, Math.max(0, rawX)),
      rawY: Math.min(1, Math.max(0, rawY)),
    };
  }

  sampleColorAtScreenPoint(clientX, clientY) {
    const { rawX, rawY } = this.mapScreenToVideo(clientX, clientY);
    const sampled = this.colorTracker.sampleAtVideoNorm(this.videoEl, rawX, rawY);
    if (sampled) {
      const mapped = this.mapVideoToStage(rawX, rawY);
      this.diag.foundCount++;
      this.diag.cursorUpdates++;
      this.diag.lastCoord = `${mapped.x.toFixed(2)}, ${mapped.y.toFixed(2)}`;
      this.diag.lastReturnTime = new Date().toLocaleTimeString('zh-TW', { hour12: false });
      this._emitDiag();
    }
    return sampled;
  }

  hasColorTarget() {
    return !!this.colorTracker.target;
  }

  clearColorTarget() {
    this.colorTracker.clearTarget();
  }

  async _createLandmarker(delegate) {
    if (!this.visionFileset) {
      this.visionFileset = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );
    }
    this.activeDelegate = delegate;
    this.diag.engine = `HAND (${delegate}${this.useCanvasInput ? '+Canvas' : ''})`;
    this._emitDiag();

    return HandLandmarker.createFromOptions(this.visionFileset, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate,
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.25,
      minHandPresenceConfidence: 0.25,
      minTrackingConfidence: 0.25,
    });
  }

  async initHandLandmarker() {
    if (this.handLandmarker) return true;
    if (this.modelInitPromise) return this.modelInitPromise;

    this.callbacks.onModelLoading?.();
    this.modelInitPromise = (async () => {
      try {
        try {
          this.handLandmarker = await this._createLandmarker('GPU');
        } catch (gpuErr) {
          this._recordError(`GPU切換CPU: ${gpuErr?.message || gpuErr}`);
          this.handLandmarker = await this._createLandmarker('CPU');
        }
        this.callbacks.onModelReady?.();
        return true;
      } catch (err) {
        this.modelInitPromise = null;
        this._recordError(`模型載入失敗: ${err?.message || err}`);
        this.callbacks.onModelError?.(err?.message || String(err));
        return false;
      }
    })();

    return this.modelInitPromise;
  }

  async _openWebcamStream() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('此瀏覽器不支援前鏡頭 API');
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
    } catch {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
    }

    this.videoEl.setAttribute('playsinline', '');
    this.videoEl.setAttribute('webkit-playsinline', '');
    this.videoEl.muted = true;
    this.videoEl.srcObject = this.stream;
    this.videoEl.classList.add('active');

    try {
      await this.videoEl.play();
    } catch {
      this.videoEl.onloadedmetadata = () => {
        this.videoEl.play().catch(() => {});
      };
    }
  }

  async start(mode) {
    this.stop();
    this.mode = mode;
    this.isRunning = true;
    this.lastInferTs = 0;
    this.lastVideoTime = -1;
    this.consecutiveErrors = 0;
    this.emptyHandFrames = 0;
    this.hasHandNow = false;
    this.lostFrames = 0;

    this.diag.engine = mode === 'HAND' ? `HAND (${this.activeDelegate})` : mode;
    this.diag.lastError = '無';
    this._emitDiag();

    if (mode === 'HAND') {
      await this._openWebcamStream();
      const ready = await this.initHandLandmarker();
      if (!this.isRunning || this.mode !== 'HAND') return;
      if (ready) {
        this.inferTimer = setInterval(() => this._runCameraInference(), 66);
      } else {
        this.callbacks.onModelFallback?.();
      }
    } else if (mode === 'COLOR') {
      await this._openWebcamStream();
      this.colorTracker.resetHistory();
      this.inferTimer = setInterval(() => this._runColorInference(), 50);
    }
  }

  stop() {
    this.isRunning = false;
    this.hasHandNow = false;
    if (this.inferTimer) {
      clearInterval(this.inferTimer);
      this.inferTimer = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.videoEl) {
      this.videoEl.srcObject = null;
      this.videoEl.classList.remove('active');
    }
  }

  _runColorInference() {
    if (!this.isRunning || this.mode !== 'COLOR' || !this.videoEl) return;
    if (this.videoEl.paused && this.stream) {
      this.videoEl.play().catch(() => {});
    }
    if (this.videoEl.readyState < 2 || this.videoEl.videoWidth === 0 || this.videoEl.videoHeight === 0) return;
    if (!this.colorTracker.target) return;

    this.diag.sentCount++;
    const res = this.colorTracker.detect(this.videoEl);
    this.diag.resultCount++;
    const nowStr = new Date().toLocaleTimeString('zh-TW', { hour12: false });
    this.diag.lastReturnTime = nowStr;

    if (res) {
      this.lostFrames = 0;
      this.diag.foundCount++;
      this.diag.cursorUpdates++;
      if (!this.hasHandNow) {
        this.hasHandNow = true;
        this.callbacks.onTrackingState?.(true);
      }
      const mapped = this.mapVideoToStage(res.rawX, res.rawY);
      this.diag.lastCoord = `${mapped.x.toFixed(2)}, ${mapped.y.toFixed(2)}`;
      this.callbacks.onPoint?.(mapped.x, mapped.y, performance.now());
    } else {
      this.lostFrames++;
      if (this.lostFrames >= 6) {
        if (this.hasHandNow) {
          this.hasHandNow = false;
          this.callbacks.onTrackingState?.(false);
        }
        this.callbacks.onLost?.();
      }
    }
    this._emitDiag();
  }

  async _runCameraInference() {
    if (!this.isRunning || this.isInferencing || !this.handLandmarker || !this.videoEl) return;
    if (this.videoEl.paused && this.stream) {
      this.videoEl.play().catch(() => {});
    }
    if (this.videoEl.readyState < 2 || this.videoEl.videoWidth === 0 || this.videoEl.videoHeight === 0) return;

    if (this.videoEl.currentTime === this.lastVideoTime) return;
    this.lastVideoTime = this.videoEl.currentTime;

    this.isInferencing = true;
    this.diag.sentCount++;

    try {
      const t0 = performance.now();
      const ts = Math.max(this.lastInferTs + 1, Math.round(t0));
      this.lastInferTs = ts;

      let sourceInput = this.videoEl;
      if (this.useCanvasInput) {
        const vw = this.videoEl.videoWidth || 320;
        const vh = this.videoEl.videoHeight || 240;
        if (this.workCanvas.width !== vw || this.workCanvas.height !== vh) {
          this.workCanvas.width = vw;
          this.workCanvas.height = vh;
        }
        this.workCtx.drawImage(this.videoEl, 0, 0, vw, vh);
        sourceInput = this.workCanvas;
      }

      const result = this.handLandmarker.detectForVideo(sourceInput, ts);
      this.diag.resultCount++;
      this.diag.lastReturnTime = new Date().toLocaleTimeString('zh-TW', { hour12: false });
      this.consecutiveErrors = 0;

      if (result.landmarks && result.landmarks.length > 0) {
        this.lostFrames = 0;
        this.emptyHandFrames = 0;
        this.diag.foundCount++;
        this.diag.cursorUpdates++;

        if (!this.hasHandNow) {
          this.hasHandNow = true;
          this.callbacks.onTrackingState?.(true);
        }

        // 自動辨識任一隻手：取第一隻偵測到的手掌中心（手腕 0 + 食指根 5 + 小指根 17）
        const lm = result.landmarks[0];
        const rawX = (lm[0].x + lm[5].x + lm[17].x) / 3;
        const rawY = (lm[0].y + lm[5].y + lm[17].y) / 3;

        const mapped = this.mapVideoToStage(rawX, rawY);
        this.diag.lastCoord = `${mapped.x.toFixed(2)}, ${mapped.y.toFixed(2)}`;
        this.callbacks.onPoint?.(mapped.x, mapped.y, performance.now());
      } else {
        this.lostFrames++;
        this.emptyHandFrames++;

        // 若 Galaxy S25 / Android GPU delegate 或硬體視訊貼圖靜默回傳空結果，自動啟用離屏 Canvas 與 CPU delegate 自癒
        if (this.emptyHandFrames === 25 && !this.useCanvasInput) {
          this.useCanvasInput = true;
          this.diag.engine = `HAND (${this.activeDelegate}+Canvas)`;
        } else if (this.emptyHandFrames >= 55 && this.activeDelegate === 'GPU') {
          this.emptyHandFrames = 0;
          try { this.handLandmarker.close?.(); } catch {}
          this.handLandmarker = null;
          this._createLandmarker('CPU')
            .then((lm) => { this.handLandmarker = lm; })
            .catch((e) => this._recordError(e?.message || e));
        }

        if (this.lostFrames >= 5) {
          if (this.hasHandNow) {
            this.hasHandNow = false;
            this.callbacks.onTrackingState?.(false);
          }
          this.callbacks.onLost?.();
        }
      }
      this._emitDiag();
    } catch (err) {
      this.consecutiveErrors++;
      this._recordError(err?.message || err);
      if (!this.useCanvasInput) {
        this.useCanvasInput = true;
        this.diag.engine = `HAND (${this.activeDelegate}+Canvas)`;
      }
      if (this.consecutiveErrors >= 3 && this.activeDelegate === 'GPU') {
        this.consecutiveErrors = 0;
        try { this.handLandmarker.close?.(); } catch {}
        this.handLandmarker = null;
        this._createLandmarker('CPU')
          .then((lm) => { this.handLandmarker = lm; })
          .catch((e) => this._recordError(e?.message || e));
      }
    } finally {
      this.isInferencing = false;
    }
  }

  _initPointerEvents() {
    const updateFromPointer = (e) => {
      if (!this.isRunning && this.mode !== 'TOUCH') return;
      if (this.mode !== 'TOUCH') return;
      const rect = this.stageEl.getBoundingClientRect();
      const nx = Math.min(1, Math.max(0, (e.clientX - rect.left) / Math.max(1, rect.width)));
      const ny = Math.min(1, Math.max(0, (e.clientY - rect.top) / Math.max(1, rect.height)));
      this.diag.cursorUpdates++;
      this.diag.lastCoord = `${nx.toFixed(2)}, ${ny.toFixed(2)}`;
      this.callbacks.onPoint?.(nx, ny, performance.now());
    };

    window.addEventListener('pointermove', updateFromPointer);
    window.addEventListener('pointerdown', (e) => {
      if (e.target.closest('#diag-widget') || e.target.closest('.modal-overlay')) return;
      updateFromPointer(e);
    });
  }
}
