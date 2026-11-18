// ============================================================================
// WebXR + Hand-Tracking Swimming Engine + WebGPU->XRWebGLLayer Bridge (XREAL Aura)
// ============================================================================
class AbyssalXR {
  constructor(gpuCanvas, cb) {
    this.gpuCanvas = gpuCanvas;
    this.cb = cb; // { onStart, onEnd, onResize, onFrame, onCyclePose }
    this.session = null;
    this.refSpace = null;
    this.gl = null;
    this.glTex = null;
    this.glFbo = null;
    this.baseLayer = null;
    this.texW = 0;
    this.texH = 0;

    // World-space swim rig (matches main.js Y-up coordinates)
    this.rigPos = new Float64Array(3);
    this.rigVel = new Float64Array(3);
    this.rigYaw = 0;

    // Head-relative hand tracking state
    this.hands = {
      left:  { valid: false, prevRel: [0, 0, 0], vel: [0, 0, 0], pinching: false, pinchTime: 0 },
      right: { valid: false, prevRel: [0, 0, 0], vel: [0, 0, 0], pinching: false, pinchTime: 0 },
    };

    // Dynamic bioluminescent hand + water-trail particles (max 64 instances * 8 floats)
    this.MAX_ORBS = 64;
    this.handVerts = new Float32Array(this.MAX_ORBS * 8);
    this.nOrbs = 0;
    this.trails = []; // { x, y, z, vx, vy, vz, life, r, g, b }
    this.lastTime = 0;
    this.strokeIntensity = 0;
  }

  async isSupported(mode = 'immersive-vr') {
    if (!navigator.xr) return false;
    try { return await navigator.xr.isSessionSupported(mode); }
    catch { return false; }
  }

  async start(startPos, startYaw = 0, mode = 'immersive-vr') {
    if (this.session) return;

    this.rigPos[0] = startPos[0];
    this.rigPos[1] = startPos[1];
    this.rigPos[2] = startPos[2];
    this.rigVel.fill(0);
    this.rigYaw = startYaw;
    this.hands.left.valid = false;
    this.hands.right.valid = false;
    this.trails.length = 0;
    this.nOrbs = 0;

    if (!this.gl) {
      const glCanvas = document.createElement('canvas');
      this.gl = glCanvas.getContext('webgl2', {
        xrCompatible: true,
        alpha: false,
        antialias: false,
        depth: false,
        stencil: false,
        powerPreference: 'high-performance',
      });
      if (!this.gl) throw new Error('WebGL2 unavailable for WebXR bridge');
      this.glTex = this.gl.createTexture();
      this.glFbo = this.gl.createFramebuffer();
    }
    await this.gl.makeXRCompatible();

    const session = await navigator.xr.requestSession(mode, {
      requiredFeatures: ['local'],
      optionalFeatures: ['hand-tracking'],
    });
    this.session = session;
    this.texW = 0;
    this.texH = 0;

    // 0.85 resolution scale keeps XREAL Aura 1200p Micro-OLED crisp while hitting frame pacing
    this.baseLayer = new XRWebGLLayer(session, this.gl, {
      alpha: false,
      depth: false,
      stencil: false,
      antialias: false,
      framebufferScaleFactor: 0.85,
    });
    await session.updateRenderState({ baseLayer: this.baseLayer });
    this.refSpace = await session.requestReferenceSpace('local');

    session.addEventListener('end', () => {
      this.session = null;
      this.lastTime = 0;
      this.nOrbs = 0;
      this.cb.onEnd?.();
    });

    this.lastTime = performance.now();
    this.cb.onStart?.(this);
    session.requestAnimationFrame(this._onFrame);
  }

  async stop() {
    if (this.session) await this.session.end();
  }

  _onFrame = (now, frame) => {
    const session = this.session;
    if (!session) return;
    session.requestAnimationFrame(this._onFrame);

    const dt = Math.min(0.05, Math.max(0.001, (now - this.lastTime) / 1000));
    this.lastTime = now;

    const pose = frame.getViewerPose(this.refSpace);
    if (!pose || pose.views.length === 0) return;

    // 1. Update hand-tracking swimming physics & bioluminescent hand/trail orbs
    this._updateSwimming(frame, pose, dt, now);

    // 2. Match WebGPU render target dimensions to XR eye viewport
    const views = pose.views;
    const v0 = this.baseLayer.getViewport(views[0]);
    const eyeW = Math.max(2, v0.width & ~1);
    const eyeH = Math.max(2, v0.height & ~1);
    this.cb.onResize(eyeW, eyeH);

    // 3. Build per-eye camera basis + asymmetric projection parameters
    const eyes = new Array(views.length);
    for (let i = 0; i < views.length; i++) {
      const view = views[i];
      const t = view.transform;
      const P = view.projectionMatrix;
      const q = t.orientation;

      const right = this._rotY(this._qRot(q, 1, 0, 0), this.rigYaw);
      const up    = this._rotY(this._qRot(q, 0, 1, 0), this.rigYaw);
      const fwd   = this._rotY(this._qRot(q, 0, 0, -1), this.rigYaw);
      const off   = this._rotY([t.position.x, t.position.y, t.position.z], this.rigYaw);

      eyes[i] = {
        view,
        xrViewport: this.baseLayer.getViewport(view),
        pos: [this.rigPos[0] + off[0], this.rigPos[1] + off[1], this.rigPos[2] + off[2]],
        right,
        up,
        fwd,
        proj: [P[0], P[5], P[8], P[9]], // scaleX, scaleY, asymX, asymY
      };
    }

    // 4. Hand off to main.js for collision + once-per-frame water + per-eye WebGPU render & blit
    this.cb.onFrame(dt, eyes, this);
  };

  blitEye(xrViewport) {
    const gl = this.gl;
    const w = this.gpuCanvas.width;
    const h = this.gpuCanvas.height;

    gl.bindTexture(gl.TEXTURE_2D, this.glTex);
    if (this.texW !== w || this.texH !== h) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.gpuCanvas);
      this.texW = w;
      this.texH = h;
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, this.gpuCanvas);
    }

    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.glFbo);
    gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.glTex, 0);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.baseLayer.framebuffer);
    gl.blitFramebuffer(
      0, h, w, 0,
      xrViewport.x, xrViewport.y, xrViewport.x + xrViewport.width, xrViewport.y + xrViewport.height,
      gl.COLOR_BUFFER_BIT, gl.LINEAR
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  _updateSwimming(frame, viewerPose, dt, now) {
    const headPos = viewerPose.transform.position;
    const headOri = viewerPose.transform.orientation;
    const headFwd = this._qRot(headOri, 0, 0, -1);

    let frameImpulse = 0;
    let pinchCount = 0;
    let orbIdx = 0;

    const pushOrb = (wx, wy, wz, rad, r, g, b, a) => {
      if (orbIdx >= this.MAX_ORBS) return;
      const o = orbIdx * 8;
      this.handVerts[o] = wx; this.handVerts[o + 1] = wy; this.handVerts[o + 2] = wz; this.handVerts[o + 3] = rad;
      this.handVerts[o + 4] = r; this.handVerts[o + 5] = g; this.handVerts[o + 6] = b; this.handVerts[o + 7] = a;
      orbIdx++;
    };

    const toWorld = (p) => {
      const rw = this._rotY([p.x, p.y, p.z], this.rigYaw);
      return [this.rigPos[0] + rw[0], this.rigPos[1] + rw[1], this.rigPos[2] + rw[2]];
    };

    for (const source of frame.session.inputSources) {
      // Fallback: Android XR Compute Puck / Gamepad thumbstick
      if (source.gamepad && !source.hand) {
        const axes = source.gamepad.axes;
        if (axes.length >= 4) {
          const sx = Math.abs(axes[2]) > 0.15 ? axes[2] : 0;
          const sy = Math.abs(axes[3]) > 0.15 ? -axes[3] : 0;
          const rightV = this._rotY(this._qRot(headOri, 1, 0, 0), this.rigYaw);
          const fwdV = this._rotY(headFwd, this.rigYaw);
          this.rigVel[0] += (fwdV[0] * sy + rightV[0] * sx) * 9.0 * dt;
          this.rigVel[1] += (fwdV[1] * sy) * 9.0 * dt;
          this.rigVel[2] += (fwdV[2] * sy + rightV[2] * sx) * 9.0 * dt;
        }
        continue;
      }

      if (!source.hand || (source.handedness !== 'left' && source.handedness !== 'right')) continue;
      const side = source.handedness;
      const state = this.hands[side];

      const wrist = frame.getJointPose(source.hand.get('wrist'), this.refSpace);
      const palm  = frame.getJointPose(source.hand.get('middle-finger-metacarpal'), this.refSpace);
      const idxM  = frame.getJointPose(source.hand.get('index-finger-metacarpal'), this.refSpace);
      const pnkM  = frame.getJointPose(source.hand.get('pinky-finger-metacarpal'), this.refSpace);
      const thmT  = frame.getJointPose(source.hand.get('thumb-tip'), this.refSpace);
      const idxT  = frame.getJointPose(source.hand.get('index-finger-tip'), this.refSpace);
      const midT  = frame.getJointPose(source.hand.get('middle-finger-tip'), this.refSpace);
      const rngT  = frame.getJointPose(source.hand.get('ring-finger-tip'), this.refSpace);
      const pnkT  = frame.getJointPose(source.hand.get('pinky-finger-tip'), this.refSpace);

      if (!wrist || !palm || !idxM || !pnkM || !idxT || !thmT) {
        state.valid = false;
        continue;
      }

      // 1. Head-relative palm position prevents turning your head from causing phantom drift
      const pp = palm.transform.position;
      const rel = [pp.x - headPos.x, pp.y - headPos.y, pp.z - headPos.z];

      // Draw bioluminescent palm + fingertip orbs in world space
      const wp = toWorld(pp);
      const pinchDist = this._dist(idxT.transform.position, thmT.transform.position);
      const isPinching = pinchDist < 0.026;
      if (isPinching && !state.pinching) {
        if (side === 'right' && now - state.pinchTime < 380) this.cb.onCyclePose?.();
        state.pinchTime = now;
      }
      state.pinching = isPinching;
      if (isPinching) pinchCount++;

      const pr = isPinching ? 0.9 : 0.15;
      const pg = isPinching ? 0.85 : 0.85;
      const pb = isPinching ? 0.25 : 1.0;
      pushOrb(wp[0], wp[1], wp[2], 0.028, pr * 0.4, pg * 0.5, pb * 0.7, 0.85);
      for (const tip of [thmT, idxT, midT, rngT,