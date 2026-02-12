# Semantic Video Tracking Studio (SAM-like, Text-Driven)

A full-stack application for **video-based semantic tracking** using text prompts (e.g. `faces`, `person`, `red dress`) with post-tracking effects, focused on privacy masking via adjustable blur.

## Features

- Upload full video (MP4/MOV/AVI/MKV/WEBM)
- Text-only target definition (`What do you want to track?`)
- Full-duration frame processing (no frame subset sampling)
- Semantic segmentation + temporal multi-object tracking
- Effects applied strictly to tracked regions:
  - Blur (mandatory, adjustable intensity)
  - Pixelation
  - Solid masking
  - Outline-only rendering
  - Background dimming
- Decoupled architecture: tracking cached once, effects re-rendered without re-running segmentation
- In-app video playback preview and MP4 download

---

## 1) Overall System Architecture

```text
[Web UI]
  ├── Upload video
  ├── Enter text prompt
  ├── Start processing + progress polling
  ├── Effect controls (type + intensity)
  └── Preview + download
        |
        v
[FastAPI Backend]
  ├── /api/upload            -> store source video
  ├── /api/process           -> async tracking pipeline
  ├── /api/status/{job_id}   -> progress/state
  ├── /api/effect            -> effect-only re-render
  └── /api/download/...      -> MP4 export
        |
        v
[Tracking + Rendering Pipeline]
  ├── Decode all frames via OpenCV
  ├── Text-guided detection (OWLv2)
  ├── Segmentation with SAM
  ├── Temporal association (Hungarian + IoU)
  ├── Persist per-frame masks + metadata
  └── Apply visual effect + encode MP4
```

### Model Stack (SAM-like behavior)

- **Text-guided selection**: `google/owlv2-base-patch16-ensemble`
- **Mask segmentation**: `facebook/sam-vit-base`

This pairing creates a text-driven segmentation workflow analogous to SAM-style prompting without manual clicks.

---

## 2) Detailed Processing Pipeline

1. **Video decode**: Read source video frame-by-frame until end-of-stream.
2. **Semantic detection**: For each frame, OWLv2 predicts boxes for the user text prompt.
3. **Temporal tracking**:
   - Associate detections with existing tracks using IoU cost matrix.
   - Solve assignments using Hungarian algorithm.
   - Keep multiple simultaneous instances.
4. **Segmentation**:
   - Feed tracked boxes into SAM.
   - Generate per-instance masks.
   - Merge to one binary region mask per frame.
5. **Cache artifacts**:
   - Save raw frame PNG + mask NPY for every frame.
   - Save metadata JSON (track IDs, boxes, paths).
6. **Effect rendering**:
   - Use cached masks to apply chosen effect.
   - Intensity slider changes trigger only re-render stage.
7. **Video assembly**:
   - Re-encode all frames to MP4.
8. **Preview + export**:
   - Preview `<video>` from generated MP4.
   - Download final MP4 from API endpoint.

---

## 3) UI Layout + Interaction Logic

Step-based web UI:

1. Upload video
2. Enter prompt (`What do you want to track?`)
3. Process video (progress bar with status messages)
4. Choose effect and intensity slider
5. Preview rendered video
6. Download final MP4

Interaction flow:

- `POST /api/upload` stores video.
- `POST /api/process` starts async job.
- Client polls `GET /api/status/{job_id}`.
- Once ready, preview auto-loads.
- User adjusts effect/intensity; `POST /api/effect` regenerates MP4 from cached masks.

---

## 4) Core Tracking + Effect Logic

### Temporal consistency and multi-instance handling

- Each frame’s detections are matched to existing tracks with IoU-based Hungarian assignment.
- Unmatched detections spawn new track IDs.
- Active track boxes are segmented and merged into a frame mask.

### Effect decoupling

- Tracking stage stores masks once.
- Effect stage consumes existing masks and original frames only.
- Blur strength slider can be changed instantly relative to full segmentation reruns.

Effects are implemented in a single `_apply_effect` function with an extensible enum-based branch.

---

## 5) Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

---

## Notes for cloud deployment

- Works on CPU but GPU strongly recommended.
- Persist `data/artifacts/` to object storage for long jobs.
- Move job execution to a queue worker (Celery/RQ) for production scale.
- Add auth and quota controls for public deployment.
