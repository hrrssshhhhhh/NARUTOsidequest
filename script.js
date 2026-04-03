// =====================================================
//  NARUTOsidequest — script.js
//
//  HOW IT WORKS (high level):
//  1. Webcam → MediaPipe SelfieSegmentation → body mask
//  2. Webcam → MediaPipe Holistic           → hand landmarks
//  3. Hand landmarks → normalise → TF.js model → confidence score
//  4. confidence > threshold → trigger clone animation
//  5. Clone animation = draw segmented-you at 16 offsets
//     + spawn animated smoke sprites at each clone position
//
//  CHANGES FROM ORIGINAL:
//  - Status dot + message system
//  - Orange joint dots on skeleton (was red)
//  - Reset button — no page refresh needed
//  - Sound effect on trigger
//  - waitingForReset guard — prevents freeze after reset
//  - tf.tidy() — prevents GPU memory leak / freeze
//  - Reused offscreen canvas — fixes freeze at 100% confidence
// =====================================================

// --- DOM References ---
const video  = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

// --- Sound Effect ---
const cloneSound = new Audio('assets/clone-sound.mp3');
cloneSound.volume = 0.8;

// --- App State ---
let clonesTriggered = false;
let cloneStartTime  = null;
let mask            = null;
let gestureModel    = null;
let waitingForReset = false;

// =====================================================
//  STATUS HELPER
// =====================================================
function setStatus(state, msg) {
  const dot = document.getElementById('dot');
  const txt = document.getElementById('statusMsg');
  dot.className = 'dot ' + state;
  txt.className = 'status-msg ' + state;
  txt.textContent = msg;
}

// =====================================================
//  LOAD GESTURE MODEL
// =====================================================
async function loadGestureModel() {
  try {
    gestureModel = await tf.loadLayersModel('gesture-model.json');
    setStatus('ready', 'model ready · perform your sign');
    console.log('✅ Gesture model loaded');
  } catch (e) {
    setStatus('', 'no model found · open trainer first');
    console.warn('⚠️ No gesture model — train one via trainer.html');
  }
}

loadGestureModel();

// =====================================================
//  NORMALISE HAND LANDMARKS
//  Wrist-relative + palm-scale-divided
//  Makes features position & size invariant
// =====================================================
function normaliseHand(lm) {
  const wrist = lm[0];
  const mcp   = lm[9];

  const scale = Math.sqrt(
    (mcp.x - wrist.x) ** 2 +
    (mcp.y - wrist.y) ** 2 +
    (mcp.z - wrist.z) ** 2
  ) || 1;

  const out = [];
  for (let i = 0; i < 21; i++) {
    out.push((lm[i].x - wrist.x) / scale);
    out.push((lm[i].y - wrist.y) / scale);
    out.push((lm[i].z - wrist.z) / scale);
  }
  return out;
}

// =====================================================
//  PREDICT GESTURE
//  Uses tf.tidy() to auto-clean all tensors every frame
//  This was the main cause of the freeze at 100%
// =====================================================
function predictGesture(rightLm, leftLm, threshold = 0.99999) {
  if (!gestureModel || !rightLm || !leftLm) return false;

  const prob = tf.tidy(() => {
    const inputData   = [...normaliseHand(rightLm), ...normaliseHand(leftLm)];
    const inputTensor = tf.tensor2d([inputData]);
    return gestureModel.predict(inputTensor).dataSync()[0];
  });

  const pct = (prob * 100).toFixed(1);
  document.getElementById('confVal').textContent  = pct + '%';
  document.getElementById('confFill').style.width = pct + '%';

  return prob > threshold;
}

// =====================================================
//  CLONE CONFIGURATION
// =====================================================
const clones = [
  // Front row — tight on both sides, same level as you
  { x: -170, y:  0,  scale: 0.88, delay:  700, smokeSpawned: false },
  { x:  170, y:  0,  scale: 0.88, delay:  700, smokeSpawned: false },

  // Second row — slightly behind, peeking over shoulders
  { x: -300, y:  10, scale: 0.75, delay: 1000, smokeSpawned: false },
  { x:  300, y:  10, scale: 0.75, delay: 1000, smokeSpawned: false },
  { x:  -80, y:  10, scale: 0.72, delay: 1100, smokeSpawned: false },
  { x:   80, y:  10, scale: 0.72, delay: 1100, smokeSpawned: false },

  // Third row — filling the background wall to wall
  { x: -390, y:  20, scale: 0.60, delay: 1400, smokeSpawned: false },
  { x:  390, y:  20, scale: 0.60, delay: 1400, smokeSpawned: false },
  { x: -200, y:  20, scale: 0.58, delay: 1500, smokeSpawned: false },
  { x:  200, y:  20, scale: 0.58, delay: 1500, smokeSpawned: false },

  // Deep background — tiny heads poking up top
  { x: -130, y:  35, scale: 0.42, delay: 1900, smokeSpawned: false },
  { x:  130, y:  35, scale: 0.42, delay: 1900, smokeSpawned: false },
  { x: -320, y:  35, scale: 0.38, delay: 2100, smokeSpawned: false },
  { x:  320, y:  35, scale: 0.38, delay: 2100, smokeSpawned: false },
  { x:    0, y:  40, scale: 0.32, delay: 2400, smokeSpawned: false },
];

// =====================================================
//  SMOKE PARTICLE SYSTEM
// =====================================================
const SMOKE_SETS   = ['smoke_1', 'smoke_2', 'smoke_3'];
const SMOKE_FRAMES = 5;
const SMOKE_DUR    = 600;
const activeSmokes = [];

function spawnSmoke(cx, cy, scale) {
  const folder = SMOKE_SETS[Math.floor(Math.random() * SMOKE_SETS.length)];
  const frames = [];
  for (let i = 1; i <= SMOKE_FRAMES; i++) {
    const img = new Image();
    img.src = `assets/${folder}/${i}.png`;
    frames.push(img);
  }
  activeSmokes.push({ x: cx, y: cy, scale: scale * 1.2, start: performance.now(), frames });
}

function drawSmokes() {
  const now = performance.now();
  for (let i = activeSmokes.length - 1; i >= 0; i--) {
    const s        = activeSmokes[i];
    const frameIdx = Math.floor((now - s.start) / (SMOKE_DUR / SMOKE_FRAMES));
    if (frameIdx >= s.frames.length) { activeSmokes.splice(i, 1); continue; }
    const img = s.frames[frameIdx];
    ctx.save();
    ctx.translate(s.x, s.y);
    ctx.scale(s.scale, s.scale);
    ctx.drawImage(img, -img.width / 2, -img.height / 2);
    ctx.restore();
  }
}

// =====================================================
//  EXTRACT PERSON
//  KEY FIX: reuse ONE offscreen canvas instead of
//  creating a new one every frame — this was causing
//  the memory buildup and freeze at 100% confidence
// =====================================================
const offscreen    = document.createElement('canvas');
const offscreenCtx = offscreen.getContext('2d');

function extractPerson() {
  offscreen.width  = canvas.width;
  offscreen.height = canvas.height;
  offscreenCtx.clearRect(0, 0, offscreen.width, offscreen.height);
  offscreenCtx.drawImage(mask, 0, 0, canvas.width, canvas.height);
  offscreenCtx.globalCompositeOperation = 'source-in';
  offscreenCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
  offscreenCtx.globalCompositeOperation = 'source-over';
  return offscreen;
}

// =====================================================
//  DRAW CLONES
// =====================================================
function drawClones(personCanvas) {
  const now    = performance.now();
  const sorted = [...clones].sort((a, b) => b.delay - a.delay);
  sorted.forEach(cl => {
    if (now - cloneStartTime < cl.delay) return;
    ctx.save();
    ctx.translate(cl.x + canvas.width * (1 - cl.scale) / 2, cl.y);
    ctx.scale(cl.scale, cl.scale);
    ctx.drawImage(personCanvas, 0, 0);
    ctx.restore();
  });
  ctx.drawImage(personCanvas, 0, 0);
}

// =====================================================
//  DRAW HAND SKELETON
// =====================================================
const FINGER_CHAINS = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8],
  [0, 9, 10, 11, 12],
  [0, 13, 14, 15, 16],
  [0, 17, 18, 19, 20],
];

function drawSkeleton(lm) {
  ctx.strokeStyle = '#39d98a';
  ctx.lineWidth   = 2;
  for (const chain of FINGER_CHAINS) {
    ctx.beginPath();
    chain.forEach((idx, pos) => {
      const x = lm[idx].x * canvas.width;
      const y = lm[idx].y * canvas.height;
      pos === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  lm.forEach(pt => {
    ctx.beginPath();
    ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#ff6b1a';
    ctx.fill();
  });
}

// =====================================================
//  OVERLAY IMAGE SWAP
// =====================================================
function activateOverlay() {
  const img = document.getElementById('overlayImg');
  if (img.dataset.state === '2') return;
  img.src           = 'assets/state-2.png';
  img.dataset.state = '2';
  const btn = document.getElementById('overlayBtn');
  btn.classList.add('pop');
  setTimeout(() => btn.classList.remove('pop'), 300);
}

// =====================================================
//  RESET BUTTON
// =====================================================
document.getElementById('resetBtn').addEventListener('click', () => {
  clonesTriggered = false;
  cloneStartTime  = null;
  waitingForReset = true;
  clones.forEach(cl => { cl.smokeSpawned = false; });
  activeSmokes.length = 0;

  const img = document.getElementById('overlayImg');
  img.src           = 'assets/state-1.png';
  img.dataset.state = '1';

  document.getElementById('confFill').style.width = '0%';
  document.getElementById('confVal').textContent  = '0.0%';

  setStatus('ready', 'reset · lower your hands first, then sign again');
});

// =====================================================
//  MEDIAPIPE — SELFIE SEGMENTATION
// =====================================================
const selfie = new SelfieSegmentation({
  locateFile: f =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${f}`,
});
selfie.setOptions({ modelSelection: 1 });
selfie.onResults(r => { mask = r.segmentationMask; });

// =====================================================
//  MEDIAPIPE — HOLISTIC
// =====================================================
const holistic = new Holistic({
  locateFile: f =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`,
});
holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true });

// =====================================================
//  CAMERA LOOP
// =====================================================
const camera = new Camera(video, {
  width: 640,
  height: 480,
  onFrame: async () => {
    await selfie.send({ image: video });
    await holistic.send({ image: video });
  },
});
camera.start();

// =====================================================
//  MAIN RESULTS CALLBACK
// =====================================================
holistic.onResults(results => {
  if (!mask) return;
  if (!video.videoWidth || !video.videoHeight) return;

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const personCanvas = extractPerson();

  // Gesture detection — stops completely once clones trigger
  // and also blocked while waitingForReset is true
  if (!clonesTriggered && !waitingForReset && gestureModel) {
    const detected = predictGesture(
      results.rightHandLandmarks,
      results.leftHandLandmarks
    );

    if (detected) {
      clonesTriggered = true;
      cloneStartTime  = performance.now();
      setStatus('active', 'shadow clone jutsu! ✦');
      console.log('🔥 Clone jutsu triggered!');

      cloneSound.currentTime = 0;
      cloneSound.play().catch(e => console.warn('Sound blocked:', e));
    }
  }

  // Auto-unlock waitingForReset once confidence drops below 50%
  // (means user lowered their hands)
  if (waitingForReset) {
    const confPct = parseFloat(document.getElementById('confVal').textContent);
    if (isNaN(confPct) || confPct < 50) {
      waitingForReset = false;
      setStatus('ready', 'ready · perform your sign');
    }
  }

  // Clone + smoke rendering
  if (clonesTriggered) {
    const now = performance.now();
    clones.forEach(cl => {
      if (!cl.smokeSpawned && now - cloneStartTime >= cl.delay) {
        cl.smokeSpawned = true;
        const cx = cl.x + canvas.width / 2;
        const cy = cl.y + canvas.height / 2 - 40;
        spawnSmoke(cx - 15, cy, cl.scale);
        spawnSmoke(cx + 15, cy, cl.scale);
      }
    });
    activateOverlay();
    drawClones(personCanvas);
    drawSmokes();
  } else {
    ctx.drawImage(personCanvas, 0, 0);
  }

  if (results.rightHandLandmarks) drawSkeleton(results.rightHandLandmarks);
  if (results.leftHandLandmarks)  drawSkeleton(results.leftHandLandmarks);
});