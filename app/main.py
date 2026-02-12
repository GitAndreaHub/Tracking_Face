from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel

from app.pipeline import TextDrivenVideoTracker

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
ARTIFACT_DIR = BASE_DIR / "data" / "artifacts"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Semantic Video Tracking Studio")
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACT_DIR)), name="artifacts")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

tracker = TextDrivenVideoTracker(artifact_root=ARTIFACT_DIR)

job_status: dict[str, dict] = {}


class EffectRequest(BaseModel):
    job_id: str
    effect: Literal["blur", "pixelate", "solid", "outline", "dim_bg"]
    intensity: int = 15


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename")
    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail="Unsupported format")

    target = UPLOAD_DIR / video.filename
    with target.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    return {"video_path": str(target.name)}


@app.post("/api/process")
def process_video(video_path: str = Form(...), prompt: str = Form(...)):
    source = UPLOAD_DIR / video_path
    if not source.exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found")

    placeholder_job = f"pending-{threading.get_ident()}-{len(job_status)}"
    job_status[placeholder_job] = {"progress": 0.0, "message": "Queued", "state": "processing"}

    def _run():
        def progress(p: float, message: str):
            if placeholder_job in job_status:
                job_status[placeholder_job].update({"progress": float(p), "message": message, "state": "processing"})

        try:
            result = tracker.process_video(source, prompt, progress)
            job_status[placeholder_job].update(
                {
                    "state": "ready",
                    "progress": 1.0,
                    "message": "Processing completed",
                    **result,
                }
            )
        except Exception as exc:
            job_status[placeholder_job].update({"state": "error", "message": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": placeholder_job}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job_status[job_id]


@app.post("/api/effect")
def apply_effect(payload: EffectRequest):
    if payload.job_id not in job_status:
        raise HTTPException(status_code=404, detail="Unknown job")

    job = job_status[payload.job_id]
    if job.get("state") != "ready":
        raise HTTPException(status_code=400, detail="Job not ready")

    artifact_job_id = job["job_id"]

    out_path = tracker.render_effect(artifact_job_id, effect=payload.effect, intensity=payload.intensity)
    url = f"/artifacts/{artifact_job_id}/{out_path.name}"
    return {"video_url": url, "download_url": f"/api/download/{artifact_job_id}/{out_path.name}"}


@app.get("/api/download/{job_id}/{filename}")
def download(job_id: str, filename: str):
    path = ARTIFACT_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(path), media_type="video/mp4", filename=filename)
