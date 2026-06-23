# SuperSight

**SuperSight** is a Python application that estimates a person's heart rate from a laptop webcam using **remote photoplethysmography (rPPG)** — detecting tiny color changes in the face caused by blood flow.

## Project goal

Build a real-time heart-rate monitor that:

1. Captures live video from your webcam
2. Detects and tracks your face
3. Extracts a pulse signal from subtle skin-color variations
4. Estimates beats per minute (BPM) and displays it on screen

This is a learning project: code is commented for beginners, and work is split into small milestones.

## Requirements

- Ubuntu Linux (tested on 24.04 LTS)
- Python 3.12+
- A working webcam and graphical display (the preview window needs a real desktop session)

## Setup

```bash
cd supersight
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| M1 | Environment setup + live webcam preview | **Complete** |
| M2 | Face detection and region of interest | **Complete** |
| M3 | Extract raw color signal from face | **In progress** |
| M4 | Filter and process the pulse signal | Not started |
| M5 | Estimate heart rate (BPM) | Not started |
| M6 | On-screen display and smoothing | Not started |
| M7 | Polish, tuning, and documentation | Not started |

## Milestone 1 — Run the webcam test

```bash
source venv/bin/activate
python m1_webcam.py
```

You should see a window titled **SuperSight M1** with live video, and the console should print frame width, height, and FPS. Press **q** to quit.

## Milestone 2 — Face detection + forehead ROI

The Face Landmarker model lives in `models/face_landmarker.task` (downloaded once during setup).

```bash
source venv/bin/activate
python m2_face_roi.py
```

You should see a window titled **SuperSight M2** with a gray face outline, a **green rectangle on your forehead**, and **"No face detected"** if you leave the frame. Press **q** to quit.

## Milestone 3 — Raw color signal capture

```bash
source venv/bin/activate
python m3_signal.py
```

You should see **SuperSight M3** with the forehead ROI, plus a HUD showing the **raw green average**, **sample count**, and **measured sampling rate (Hz)**. If you leave the frame, sampling pauses and **"No face — signal paused"** appears. Press **q** to quit.
