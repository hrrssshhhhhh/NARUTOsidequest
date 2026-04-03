# NARUTOsidequest — Shadow Clone Jutsu

> Real-time AR web app that detects a hand gesture via webcam and spawns Naruto shadow clones of you — with smoke effects and sound, running entirely in the browser.

![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=flat&logo=google&logoColor=white)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![No Backend](https://img.shields.io/badge/Backend-None-success?style=flat)

---

## Demo

> Perform your trained hand sign → confidence hits 100% → clones spawn with smoke + sound 🔥

---

## How It Works

```
Webcam feed
    │
    ├──▶ MediaPipe SelfieSegmentation ──▶ body mask (cuts you from background)
    │
    └──▶ MediaPipe Holistic ────────────▶ 21 hand landmarks × 2 hands
                                                │
                                         normalise landmarks
                                         (wrist-relative, palm-scale invariant)
                                                │
                                         TF.js Neural Network
                                         [126] → Dense(64) → Dropout(0.3)
                                                → Dense(32) → Dense(1, sigmoid)
                                                │
                                         confidence > 0.99999?
                                                │
                                         YES → spawn clones + smoke + sound 🔥
```

---

## Features

- **Real-time hand gesture recognition** using a custom-trained TF.js neural network
- **Body segmentation** via MediaPipe — clones are composited copies of your actual silhouette
- **16 shadow clones** with staggered delays, depth scaling, and smoke animations
- **Sound effect** plays on gesture trigger
- **Reset button** — re-trigger without refreshing the page
- **Gesture trainer UI** — record samples, train, and export your model in the browser
- **Zero backend** — everything runs client-side via CDN

---

## Tech Stack

| Technology | Role |
|---|---|
| MediaPipe Holistic | 21-point hand landmark detection (both hands) |
| MediaPipe Selfie Segmentation | Body isolation for clone compositing |
| TensorFlow.js | In-browser neural network training + inference |
| HTML5 Canvas | All rendering — video, clones, smoke, hand skeleton |
| Web Audio API | Sound effect playback on gesture trigger |
| jsDelivr CDN | All dependencies — zero npm install |

---

## Project Structure

```
NARUTOsidequest/
├── index.html          ← main app
├── script.js           ← clone rendering, gesture detection, smoke system
├── styles.css          ← main app styles (dark + orange theme)
├── trainer.html        ← gesture training UI
├── trainer.js          ← TF.js model definition, training loop, export
├── trainer.css         ← trainer styles
├── README.md
└── assets/
    ├── state-1.png         ← hand sign overlay (ready state)
    ├── state-2.png         ← hand sign overlay (triggered state)
    ├── clone-sound.mp3     ← sound effect on trigger
    ├── smoke_1/            ← smoke sprite set 1 (5 frames)
    ├── smoke_2/            ← smoke sprite set 2 (5 frames)
    ├── smoke_3/            ← smoke sprite set 3 (5 frames)
    └── smoke_small_1/      ← small smoke sprites
```

---

## Getting Started

### Prerequisites
- Node.js (for `npx serve`)
- Google Chrome
- A webcam

### Run Locally

```bash
# Clone the repo
git clone https://github.com/hrrssshhhhhh/NARUTOsidequest.git
cd NARUTOsidequest

# Start local server (webcam requires localhost or https)
npx serve -p 3000
```

Open Chrome:
- **Trainer:** `http://localhost:3000/trainer.html`
- **Main App:** `http://localhost:3000`

---

## Training Your Own Gesture

The model files are not included — you train your own sign in the browser.

### Step 1 — Record Samples
1. Open `http://localhost:3000/trainer.html` in Chrome
2. Press `1` → hold your two-hand sign for 4 seconds → repeat until **100+ samples**
3. Press `2` → move hands randomly (negative samples) → repeat until **100+ samples**

### Step 2 — Train
- Click **Train Model** — trains for 50 epochs in the browser
- Watch accuracy climb live
- Test your sign — the confidence bar updates in real time

### Step 3 — Export & Use
- Click **Save Model** → downloads `gesture-model.json` + `gesture-model.weights.bin`
- Move both files into the project root
- Open `http://localhost:3000` → perform your sign → clones appear 🔥

---

## ML Model

**Architecture (binary classifier):**
```
Input: [126 floats]
  21 landmarks × 3 coords (x,y,z) × 2 hands
  normalised: wrist-relative + palm-scale-invariant

→ Dense(64, ReLU)
→ Dropout(0.3)
→ Dense(32, ReLU)
→ Dense(1, Sigmoid)

Output: probability in [0, 1]
Threshold: 0.99999 (very strict to avoid false triggers)
```

**Training:**
- Loss: Binary cross-entropy
- Optimiser: Adam
- Epochs: 50
- Batch size: 16

---

## Customisation

**Clone positions** — edit `clones` array in `script.js`:
```js
{ x: -170, y: 0, scale: 0.88, delay: 700 }
// x,y = offset from canvas centre
// scale = size relative to you (0–1)
// delay = ms after trigger
```

**Confidence threshold:**
```js
function predictGesture(rightLm, leftLm, threshold = 0.99999)
// lower it (e.g. 0.999) if sign rarely triggers
```

**Sound volume:**
```js
cloneSound.volume = 0.8; // 0.0 to 1.0
```