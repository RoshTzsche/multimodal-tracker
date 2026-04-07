# Multimodal Tracker

A real-time computer vision system that combines **hand gesture recognition** and **facial expression detection** to trigger visual overlays. Built with MediaPipe Tasks API and OpenCV, runs on Linux, Windows, and macOS.

---

## How it works

Each video frame is processed by two independent MediaPipe models running in parallel:

1. **Hand Landmarker** — extracts 21 3D landmarks from the detected hand.
2. **Face Landmarker** — extracts 468 3D landmarks from the detected face.

Both outputs are fed into analytic geometry classifiers (no secondary neural network needed). When a registered **(Hand Gesture, Face Expression)** pair is matched, a PNG overlay is composited onto the frame in real-time.

```
Webcam → Hand Landmarker ─→ classify_hand() ──┐
       → Face Landmarker ─→ classify_face() ──┴→ Combo Map → Overlay
```

---

## Recognized states

**Hand gestures**

| Label | Description |
|---|---|
| `THUMB_UP` | Thumb extended upward, all other fingers closed |
| `THUMB_DOWN` | Thumb extended downward |
| `FIST` | All fingers closed |
| `OPEN_PALM` | 4 or more fingers extended |
| `PEACE` | Index + middle finger up |
| `POINT` | Only index finger up |
| `UNKNOWN` | No matching pattern |

**Facial expressions**

| Label | Description |
|---|---|
| `NEUTRAL` | Default resting face |
| `SURPRISED` | Mouth open (MAR > 0.45) |
| `SMILE` | Wide mouth, lips closed |
| `WINK_LEFT` | Left eye closed, right eye open |
| `WINK_RIGHT` | Right eye closed, left eye open |

---

## Combo map

| Hand | Face | Overlay file |
|---|---|---|
| `THUMB_UP` | `NEUTRAL` | `like.png` |
| `THUMB_DOWN` | `NEUTRAL` | `dislike.png` |
| `FIST` | `NEUTRAL` | `rock.png` |
| `PEACE` | `NEUTRAL` | `peace.png` |
| `OPEN_PALM` | `SURPRISED` | `shocked.png` |
| `POINT` | `SURPRISED` | `look_there.png` |
| `PEACE` | `SURPRISED` | `party.png` |
| `THUMB_UP` | `SMILE` | `super_like.png` |
| `OPEN_PALM` | `SMILE` | `hello.png` |
| `PEACE` | `SMILE` | `happy_vibes.png` |
| `POINT` | `SMILE` | `idea.png` |
| `POINT` | `WINK_LEFT` | `secret.png` |
| `POINT` | `WINK_RIGHT` | `target_locked.png` |
| `FIST` | `WINK_LEFT` | `bro_fist.png` |
| `OPEN_PALM` | `WINK_RIGHT` | `high_five.png` |

---

## Requirements

- **OS:** Windows 10+, macOS 12+, or any modern Linux distro
- **Python:** 3.8 – 3.11 *(MediaPipe does not yet support 3.12+)*
- **Hardware:** Any functional USB or integrated webcam

---

## Installation

### 1 · Clone the repository

```bash
git clone https://github.com/your-user/multimodal-tracker.git
cd multimodal-tracker
```

### 2 · Create and activate a virtual environment

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3 · Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

### 4 · Create the images folder

The repository does not ship overlay images. Create the folder and add your own PNGs:

```bash
mkdir images
```

Place the following files inside `images/`. PNG files with transparent backgrounds (RGBA) work best, but solid JPEGs are also accepted — the system normalizes everything to BGRA automatically.

```
images/
├── like.png
├── dislike.png
├── rock.png
├── peace.png
├── shocked.png
├── look_there.png
├── party.png
├── super_like.png
├── hello.png
├── happy_vibes.png
├── idea.png
├── secret.png
├── target_locked.png
├── bro_fist.png
└── high_five.png
```

> **Tip:** Any PNG works. Images are automatically resized to 200 px wide. Missing images are skipped silently — only their combo won't fire.

### 5 · Run

```bash
python tracker.py
```

MediaPipe model files (~40 MB total) are downloaded automatically on first run and cached in `./models/`.

**Controls:** press `ESC` to exit.

---

## Project structure

```
multimodal-tracker/
├── tracker.py      # Main entry point — camera loop, detection, drawing
├── actions.py      # Combo map, image loading and caching
├── images/         # [create manually] PNG overlay assets
├── models/         # [auto-created] MediaPipe .task model files
├── .gitignore
└── README.md
```

---

## Changing the active camera

The system defaults to camera index `0`. If you have multiple cameras (e.g. a laptop webcam + external USB), change the index inside `tracker.py`:

```python
# tracker.py — inside the run() method
cap = self._initialize_camera(camera_index=1)  # 0, 1, 2 …
```

On Linux you can also list available devices with:
```bash
ls /dev/video*
v4l2-ctl --list-devices
```

---

## Adding new combos

1. Add your PNG file to `images/`.
2. Open `actions.py` and add an entry to `self.combo_map`:

```python
("FIST", "SMILE"): os.path.join(self.images_dir, "power_up.png"),
```

3. That's it — no other code changes needed.

---

## Technical details

### Finger extension (hand classification)

Squared Euclidean distance is used to avoid `sqrt()` on every frame:

```
d²(wrist, tip) > d²(wrist, pip)  →  finger is extended
```

### Surprise detection — Mouth Aspect Ratio (MAR)

```
MAR = ||P_top - P_bot|| / ||P_left - P_right||

MAR > 0.45  →  SURPRISED
```

### Wink detection — Eye Aspect Ratio (EAR)

Six landmarks per eye (p1…p6):

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 · ||p1-p4||)

EAR_left < 0.2  AND  EAR_right > 0.2  →  WINK_LEFT
EAR_right < 0.2 AND  EAR_left  > 0.2  →  WINK_RIGHT
```

### Alpha compositing

Overlay pixels are blended using the PNG alpha channel:

```
output = α · overlay + (1 - α) · background
```

---

*by Rosh.*
