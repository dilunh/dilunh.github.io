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
    this.glProg = null;
    this.glVao = null;
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
    this.trails = [];
    this.lastTime = 0;
    this.strokeIntensity = 0;
  }

  async isSupported(mode = 'immersive-ar') {
    if (!navigator.xr) return false;
    try { return await navigator.xr.isSessionSupported(mode); }
    catch { return false; }
  }

  _initGL() {
    if (this.gl) return;
    const glCanvas = document.createElement('canvas');
    const gl = glCanvas.getContext('webgl2', {
      xrCompatible: true,
      alpha: true,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 unavailable for WebXR bridge');

    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, `#version 300 es
      out vec2 vUV;
      void main() {
        float x = float((gl_VertexID & 1) * 4 - 1);
        float y = float((gl_VertexID >> 1) * 4 - 1);
        vUV = vec2(x * 0.5 + 0.5, 0.5 - y * 0.5);
        gl_Position = vec4(x, y, 0.0, 1.0);
      }`);
    gl.compileShader(vs);

    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, `#version 300 es
      precision highp float;
      uniform sampler2D uTex;
      in vec2 vUV;
      out vec4 oCol;
      void main() { oCol = texture(uTex, vUV); }`);
    gl.compileShader(fs);

    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('XR blit shader failed: ' + gl.getProgramInfoLog(prog));
    }

    this.gl = gl;
    this.glProg = prog;
    this.glVao = gl.createVertexArray();
    this.glTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.glTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  async start(startPos, startYaw = 0, mode = 'immersive-ar') {
    if (this.session) return;
    if (!navigator.xr) throw new Error('WebXR not available (requires HTTPS)');

    this.rigPos[0] = startPos[0];
    this.rigPos[1] = startPos[1];
    this.rigPos[2] = startPos[2];
    this.rigVel.fill(0);
    this.rigYaw = startYaw;
    this.hands.left.valid = false;
    this.hands.right.valid = false;
    this.trails.length = 0;
    this.nOrbs = 0;

    // IMPORTANT: Call requestSession FIRST inside the click handler before any other await
    // so Chrome does not drop transient user activation.
    const session = await navigator.xr.requestSession(mode, {
      optionalFeatures: ['local-floor', 'local', 'hand-tracking'],
    });
    this.session = session;
    this.texW = 0;
    this.texH = 0;

    try {
      this._initGL();
      await this.gl.makeXRCompatible();

      this.baseLayer = new XRWebGLLayer(session, this.gl, {
        alpha: mode === 'immersive-ar',
        depth: false,
        stencil: false,
        antialias: false,
        framebufferScaleFactor: 0.85,
      });
      await session.updateRenderState({ baseLayer: this.baseLayer });

      try {
        this.refSpace = await session.requestReferenceSpace('local');
      } catch {
        this.refSpace = await session.requestReferenceSpace('local-floor');
      }
    } catch (err) {
      this.session = null;
      await session.end().catch(() => {});
      throw err;
    }

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

    this._updateSwimming(frame, pose, dt, now);

    const views = pose.views;
    const v0 = this.baseLayer.getViewport(views[0]);
    const eyeW = Math.max(2, v0.width & ~1);
    const eyeH = Math.max(2, v0.height & ~1);
    this.cb.onResize(eyeW, eyeH);

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
        proj: [P[0], P[5], P[8], P[9]],
      };
    }

    this.cb.onFrame(dt, eyes, this);
  };

  blitEye(xrViewport) {
    const gl = this.gl;
    const w = this.gpuCanvas.width;
    const h = this.gpuCanvas.height;

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.glTex);
    if (this.texW !== w || this.texH !== h) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.gpuCanvas);
      this.texW = w;
      this.texH = h;
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, this.gpuCanvas);
    }

    // Draw fullscreen textured triangle directly into XRWebGLLayer framebuffer
    // (avoids blitFramebuffer SRGB/MSAA INVALID_OPERATION errors on Android GPUs)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.baseLayer.framebuffer);
    gl.viewport(xrViewport.x, xrViewport.y, xrViewport.width, xrViewport.height);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.CULL_FACE);
    gl.useProgram(this.glProg);
    gl.bindVertexArray(this.glVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
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

      const pp = palm.transform.position;
      const rel = [pp.x - headPos.x, pp.y - headPos.y, pp.z - headPos.z];

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
      for (const tip of [thmT, idxT, midT, rngT, pnkT]) {
        if (!tip) continue;
        const wt = toWorld(tip.transform.position);
        pushOrb(wt[0], wt[1], wt[2], 0.011, pr, pg, pb, 0.9);
      }

      if (!state.valid) {
        state.prevRel = rel;
        state.vel[0] = state.vel[1] = state.vel[2] = 0;
        state.valid = true;
        continue;
      }

      const rawVx = (rel[0] - state.prevRel[0]) / dt;
      const rawVy = (rel[1] - state.prevRel[1]) / dt;
      const rawVz = (rel[2] - state.prevRel[2]) / dt;
      state.prevRel = rel;

      const rawSpeed = Math.hypot(rawVx, rawVy, rawVz);
      if (rawSpeed > 4.5) {
        state.vel[0] *= 0.5; state.vel[1] *= 0.5; state.vel[2] *= 0.5;
        continue;
      }

      const smooth = 1.0 - Math.exp(-18.0 * dt);
      state.vel[0] += (rawVx - state.vel[0]) * smooth;
      state.vel[1] += (rawVy - state.vel[1]) * smooth;
      state.vel[2] += (rawVz - state.vel[2]) * smooth;

      const u = this._sub(idxM.transform.position, pnkM.transform.position);
      const v = this._sub(pp, wrist.transform.position);
      let palmNormal = side === 'right' ? this._cross(u, v) : this._cross(v, u);
      const pnLen = Math.hypot(palmNormal[0], palmNormal[1], palmNormal[2]) || 1;
      palmNormal[0] /= pnLen; palmNormal[1] /= pnLen; palmNormal[2] /= pnLen;

      const openDist = midT ? this._dist(midT.transform.position, pp) : 0.08;
      const openness = isPinching ? 0.15 : Math.min(1.0, Math.max(0.2, (openDist - 0.03) / 0.06));

      const speed = Math.hypot(state.vel[0], state.vel[1], state.vel[2]);
      if (speed > 0.08) {
        const dir = [state.vel[0] / speed, state.vel[1] / speed, state.vel[2] / speed];
        const palmFacing = dir[0] * palmNormal[0] + dir[1] * palmNormal[1] + dir[2] * palmNormal[2];
        const dragCoeff = palmFacing > 0.05 ? (0.15 + 0.85 * palmFacing) * openness : 0.12;

        const worldDir = this._rotY(dir, this.rigYaw);
        const accel = speed * speed * dragCoeff * 12.0;

        this.rigVel[0] -= worldDir[0] * accel * dt;
        this.rigVel[1] -= worldDir[1] * accel * dt;
        this.rigVel[2] -= worldDir[2] * accel * dt;
        frameImpulse += accel * dragCoeff;

        if (palmFacing > 0.25 && speed > 0.35 && this.trails.length < 36) {
          this.trails.push({
            x: wp[0] + (Math.random() - 0.5) * 0.04,
            y: wp[1] + (Math.random() - 0.5) * 0.04,
            z: wp[2] + (Math.random() - 0.5) * 0.04,
            vx: worldDir[0] * 0.25,
            vy: worldDir[1] * 0.25 + 0.08,
            vz: worldDir[2] * 0.25,
            life: 1.0,
          });
        }
      }
    }

    if (pinchCount === 2) {
      const gazeWorld = this._rotY(headFwd, this.rigYaw);
      this.rigVel[0] += gazeWorld[0] * 4.5 * dt;
      this.rigVel[1] += gazeWorld[1] * 4.5 * dt;
      this.rigVel[2] += gazeWorld[2] * 4.5 * dt;
    }

    for (let i = this.trails.length - 1; i >= 0; i--) {
      const tr = this.trails[i];
      tr.life -= dt * 1.4;
      if (tr.life <= 0) { this.trails.splice(i, 1); continue; }
      tr.x += tr.vx * dt; tr.y += tr.vy * dt; tr.z += tr.vz * dt;
      pushOrb(tr.x, tr.y, tr.z, 0.009 * tr.life, 0.25, 0.95, 1.0, tr.life * 0.7);
    }
    this.nOrbs = orbIdx;

    const damp = Math.exp(-1.65 * dt);
    this.rigVel[0] *= damp;
    this.rigVel[1] *= damp;
    this.rigVel[2] *= damp;

    const curSpeed = Math.hypot(this.rigVel[0], this.rigVel[1], this.rigVel[2]);
    if (curSpeed > 9.0) {
      const s = 9.0 / curSpeed;
      this.rigVel[0] *= s; this.rigVel[1] *= s; this.rigVel[2] *= s;
    }

    this.rigPos[0] += this.rigVel[0] * dt;
    this.rigPos[1] += this.rigVel[1] * dt;
    this.rigPos[2] += this.rigVel[2] * dt;
    this.strokeIntensity = Math.min(1, frameImpulse * 0.15);
  }

  _rotY(v, yaw) {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    return [v[0] * c - v[2] * s, v[1], v[0] * s + v[2] * c];
  }

  _qRot(q, x, y, z) {
    const ix =  q.w * x + q.y * z - q.z * y;
    const iy =  q.w * y + q.z * x - q.x * z;
    const iz =  q.w * z + q.x * y - q.y * x;
    const iw = -q.x * x - q.y * y - q.z * z;
    return [
      ix * q.w + iw * -q.x + iy * -q.z - iz * -q.y,
      iy * q.w + iw * -q.y + iz * -q.x - ix * -q.z,
      iz * q.w + iw * -q.z + ix * -q.y - iy * -q.x,
    ];
  }

  _sub(a, b) { return [a.x - b.x, a.y - b.y, a.z - b.z]; }
  _dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z); }
  _cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
}