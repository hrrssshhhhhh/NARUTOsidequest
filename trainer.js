// =====================================================
//  NARUTOsidequest — trainer.js
//
//  PURPOSE:
//  Collect hand landmark data for two classes,
//  train a binary neural net in the browser,
//  and export the model for use in the main app.
//
//  ML PIPELINE:
//  Webcam → MediaPipe Holistic → hand landmarks (21 pts/hand)
//  → normalise to 126 floats → TF.js NN → probability [0,1]
//
//  MODEL ARCHITECTURE:
//  Input [126] → Dense(64, ReLU) → Dropout(0.3)
//             → Dense(32, ReLU) → Dense(1, Sigmoid) → Output
//
//  WORKFLOW:
//  1. Press 1 → record your clone sign (100+ samples)
//  2. Press 2 → record other poses  (100+ samples)
//  3. Click Train Model
//  4. Click Save Model → gesture-model.json + .weights.bin
//  5. Move both files to project root
//  6. Open main app and perform your sign!
// =====================================================

// --- DOM References ---
const video  = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

// =====================================================
//  MEDIAPIPE HOLISTIC SETUP
// =====================================================
const holistic = new Holistic({
  locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`,
});
holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true });

const cam = new Camera(video, {
  width: 640,
  height: 480,
  onFrame: async () => await holistic.send({ image: video }),
});
cam.start();

holistic.onResults(res => {
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Mirror the feed — feels more natural for a selfie camera
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  if (res.rightHandLandmarks) drawHand(res.rightHandLandmarks);
  if (res.leftHandLandmarks)  drawHand(res.leftHandLandmarks);

  // Capture a data sample if currently recording
  captureFrame(res.rightHandLandmarks, res.leftHandLandmarks);

  // Live prediction — only runs after model is trained
  if (model && res.rightHandLandmarks && res.leftHandLandmarks) {
    const input = tf.tensor2d([extract(res.rightHandLandmarks, res.leftHandLandmarks)]);
    const prob  = model.predict(input).dataSync()[0];
    input.dispose();
    updateConfBar(prob);
  }
});

// =====================================================
//  HAND SKELETON DRAWING
//  Note: x is mirrored (canvas.width - x)
//  because we flipped the canvas with scale(-1, 1)
// =====================================================
const CHAINS = [
  [0, 1, 2, 3, 4],     // thumb
  [0, 5, 6, 7, 8],     // index
  [0, 9, 10, 11, 12],  // middle
  [0, 13, 14, 15, 16], // ring
  [0, 17, 18, 19, 20], // pinky
];

function drawHand(lm) {
  ctx.strokeStyle = '#39d98a';
  ctx.lineWidth   = 2;

  for (const chain of CHAINS) {
    ctx.beginPath();
    chain.forEach((idx, pos) => {
      // Mirror x coordinate to match the flipped canvas
      const x = canvas.width - lm[idx].x * canvas.width;
      const y = lm[idx].y * canvas.height;
      pos === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  lm.forEach(pt => {
    ctx.beginPath();
    ctx.arc(
      canvas.width - pt.x * canvas.width,
      pt.y * canvas.height,
      3, 0, Math.PI * 2
    );
    ctx.fillStyle = '#ff6b1a'; // orange — matches our theme
    ctx.fill();
  });
}

// =====================================================
//  FEATURE EXTRACTION
//
//  Converts raw MediaPipe landmarks into a
//  normalised 126-float feature vector.
//
//  normalise(): wrist-relative + palm-scale-divided
//  → model works regardless of hand position / size
//
//  extract(): combines both hands → [126] input vector
// =====================================================
function normalise(lm) {
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
  return out; // 63 floats
}

function extract(right, left) {
  return [...normalise(right), ...normalise(left)]; // 126 floats
}

// =====================================================
//  DATA COLLECTION STATE
//  samples object stores labelled feature vectors:
//  { clone_sign: [[126 floats], ...],
//    not_sign:   [[126 floats], ...] }
// =====================================================
let samples   = { clone_sign: [], not_sign: [] };
let recording = null; // null | 'clone_sign' | 'not_sign'
let model     = null; // TF.js model (null until trained)

const statusEl = document.getElementById('trainStatus');

// Called every frame — only stores if recording is active
function captureFrame(right, left) {
  if (!recording || !right || !left) return;

  samples[recording].push(extract(right, left));

  document.getElementById('countClone').textContent = samples.clone_sign.length;
  document.getElementById('countOther').textContent  = samples.not_sign.length;
}

// =====================================================
//  RECORDING FLOW
//  countdown (3s) → recording (4s) → stop
//  Countdown gives you time to get into position
// =====================================================
const COUNTDOWN   = 3; // seconds before recording starts
const RECORD_TIME = 4; // seconds of actual data capture

let countdownTimer = null;
let recordTimer    = null;

function startCountdown(label) {
  cancelRecording(); // cancel any in-progress session

  const badge = document.getElementById('recBadge');
  let rem = COUNTDOWN;

  badge.classList.add('active');
  badge.textContent    = `GET READY ${rem}`;
  statusEl.textContent = `Starting in ${rem}s — get into position`;

  countdownTimer = setInterval(() => {
    rem--;
    if (rem > 0) {
      badge.textContent    = `GET READY ${rem}`;
      statusEl.textContent = `Starting in ${rem}s`;
    } else {
      clearInterval(countdownTimer);
      countdownTimer = null;
      startRecording(label);
    }
  }, 1000);
}

function startRecording(label) {
  recording = label;

  const badge = document.getElementById('recBadge');
  let rem = RECORD_TIME;

  badge.textContent    = `● REC ${rem}s`;
  statusEl.textContent = `Recording — hold your pose!`;

  recordTimer = setInterval(() => {
    rem--;
    if (rem > 0) {
      badge.textContent = `● REC ${rem}s`;
    } else {
      stopRecording();
      statusEl.textContent = `Done! Record more or click Train.`;
    }
  }, 1000);
}

function stopRecording() {
  recording = null;
  clearInterval(recordTimer);
  recordTimer = null;
  document.getElementById('recBadge').classList.remove('active');
}

function cancelRecording() {
  clearInterval(countdownTimer);
  clearInterval(recordTimer);
  countdownTimer = null;
  recordTimer    = null;
  recording      = null;
  document.getElementById('recBadge').classList.remove('active');
}

// Button listeners
document.getElementById('btnClone').addEventListener('click', () =>
  startCountdown('clone_sign'));

document.getElementById('btnOther').addEventListener('click', () =>
  startCountdown('not_sign'));

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.repeat) return;
  if (e.key === '1') startCountdown('clone_sign');
  if (e.key === '2') startCountdown('not_sign');
});

// =====================================================
//  MODEL TRAINING
//
//  Architecture:
//    Dense(64, ReLU)   — extract patterns from landmarks
//    Dropout(0.3)      — prevent overfitting
//    Dense(32, ReLU)   — refine features
//    Dense(1, Sigmoid) — output probability [0, 1]
//
//  Loss:      Binary cross-entropy (binary classification)
//  Optimiser: Adam
//  Epochs:    50
//
//  TIPS FOR BETTER ACCURACY:
//  - Collect 100+ samples per class
//  - Vary lighting, hand distance, and slight angle
//  - If accuracy is low: add more data before adding layers
//  - If overfitting: increase Dropout rate (e.g. 0.4)
//  - If underfitting: add another Dense layer
// =====================================================
document.getElementById('btnTrain').addEventListener('click', async () => {
  const nPos = samples.clone_sign.length;
  const nNeg = samples.not_sign.length;

  if (nPos < 5 || nNeg < 5) {
    statusEl.textContent = 'Need at least 5 samples per class.';
    return;
  }

  // Build flat feature + label arrays
  const xs = [];
  const ys = [];
  samples.clone_sign.forEach(s => { xs.push(s); ys.push(1); });
  samples.not_sign.forEach(s   => { xs.push(s); ys.push(0); });

  // Fisher-Yates shuffle — prevents model learning sample order
  for (let i = xs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [xs[i], xs[j]] = [xs[j], xs[i]];
    [ys[i], ys[j]] = [ys[j], ys[i]];
  }

  const xTensor = tf.tensor2d(xs);
  const yTensor = tf.tensor1d(ys);

  // Free previous model's GPU memory before creating new one
  if (model) model.dispose();

  // Build model
  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [126], units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1,  activation: 'sigmoid' }));

  model.compile({
    optimizer: 'adam',
    loss:      'binaryCrossentropy',
    metrics:   ['accuracy'],
  });

  document.getElementById('btnTrain').disabled = true;
  statusEl.textContent = 'Training...';

  await model.fit(xTensor, yTensor, {
    epochs:    50,
    batchSize: 16,
    shuffle:   true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        statusEl.textContent =
          `Epoch ${epoch + 1}/50 — acc: ${(logs.acc * 100).toFixed(1)}%`;
      },
    },
  });

  // Always dispose tensors after use
  xTensor.dispose();
  yTensor.dispose();

  document.getElementById('btnTrain').disabled = false;
  statusEl.textContent =
    `✓ Done! ${nPos + nNeg} samples. Test your sign above.`;
});

// =====================================================
//  LIVE CONFIDENCE BAR UPDATE
//  Called every frame after model is trained
// =====================================================
function updateConfBar(prob) {
  const pct = (prob * 100).toFixed(0);
  document.getElementById('confFill').style.width = pct + '%';
  document.getElementById('confLabel').textContent = pct + '%';
}

// =====================================================
//  EXPORT / IMPORT / SAVE / CLEAR
// =====================================================

// Export raw sample data as JSON (so you can re-train later)
document.getElementById('btnExport').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(samples)], { type: 'application/json' });
  const a = Object.assign(document.createElement('a'), {
    href:     URL.createObjectURL(blob),
    download: 'gesture-data.json',
  });
  a.click();
});

// Import previously exported sample data
document.getElementById('btnImport').addEventListener('click', () =>
  document.getElementById('importInput').click());

document.getElementById('importInput').addEventListener('change', e => {
  const reader = new FileReader();
  reader.onload = ev => {
    const data = JSON.parse(ev.target.result);
    samples.clone_sign.push(...(data.clone_sign || []));
    samples.not_sign.push(...(data.not_sign   || []));
    document.getElementById('countClone').textContent = samples.clone_sign.length;
    document.getElementById('countOther').textContent  = samples.not_sign.length;
    statusEl.textContent = 'Data imported.';
  };
  reader.readAsText(e.target.files[0]);
});

// Save trained model as gesture-model.json + gesture-model.weights.bin
// Move both files to project root after downloading
document.getElementById('btnSave').addEventListener('click', async () => {
  if (!model) {
    statusEl.textContent = 'Train a model first!';
    return;
  }
  await model.save('downloads://gesture-model');
  statusEl.textContent = '✓ Saved! Move both files to your project root.';
});

// Clear all collected sample data
document.getElementById('btnClear').addEventListener('click', () => {
  samples = { clone_sign: [], not_sign: [] };
  document.getElementById('countClone').textContent = '0';
  document.getElementById('countOther').textContent  = '0';
  statusEl.textContent = 'Data cleared.';
});