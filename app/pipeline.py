from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForMaskGeneration, AutoModelForZeroShotObjectDetection, AutoProcessor


ProgressCallback = Callable[[float, str], None]


@dataclass
class TrackState:
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TextDrivenVideoTracker:
    def __init__(self, artifact_root: Path):
        self.artifact_root = artifact_root
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector_processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        ).to(self.device)

        self.segmenter_processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")
        self.segmenter_model = AutoModelForMaskGeneration.from_pretrained("facebook/sam-vit-base").to(self.device)

    def process_video(self, video_path: Path, prompt: str, progress: ProgressCallback | None = None) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.artifact_root / job_id
        frames_dir = job_dir / "frames"
        masks_dir = job_dir / "masks"
        frames_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tracks: list[TrackState] = []
        next_track_id = 1
        frame_idx = 0
        metadata = []

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            detections = self._detect(pil_image, prompt)
            associations, unmatched = self._associate_tracks(tracks, detections)

            for ti, di in associations:
                tracks[ti].bbox = detections[di]

            for di in unmatched:
                tracks.append(TrackState(track_id=next_track_id, bbox=detections[di]))
                next_track_id += 1

            active_boxes = [t.bbox for t in tracks if any(_iou(t.bbox, d) > 0.2 for d in detections)]
            masks = self._segment(frame_rgb, active_boxes)
            combined_mask = self._merge_masks(masks, (height, width))

            frame_path = frames_dir / f"{frame_idx:06d}.png"
            mask_path = masks_dir / f"{frame_idx:06d}.npy"
            cv2.imwrite(str(frame_path), frame_bgr)
            np.save(mask_path, combined_mask)

            tracked_boxes = []
            for t in tracks:
                if any(_iou(t.bbox, d) > 0.2 for d in detections):
                    tracked_boxes.append({"track_id": t.track_id, "bbox": t.bbox.tolist()})

            metadata.append(
                {
                    "frame": frame_idx,
                    "tracked_boxes": tracked_boxes,
                    "mask_path": str(mask_path.name),
                }
            )

            frame_idx += 1
            if progress and total_frames > 0:
                progress(frame_idx / total_frames * 0.7, f"Tracking frame {frame_idx}/{total_frames}")

        cap.release()

        meta = {
            "job_id": job_id,
            "prompt": prompt,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": frame_idx,
            "metadata": metadata,
        }
        with (job_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if progress:
            progress(0.72, "Tracking complete")

        preview_path = self.render_effect(job_id, effect="blur", intensity=15, progress=progress, as_preview=True)
        if progress:
            progress(1.0, "Done")

        return {
            "job_id": job_id,
            "preview_url": f"/artifacts/{job_id}/{preview_path.name}",
            "metadata_url": f"/artifacts/{job_id}/metadata.json",
        }

    def render_effect(
        self,
        job_id: str,
        effect: Literal["blur", "pixelate", "solid", "outline", "dim_bg"],
        intensity: int,
        progress: ProgressCallback | None = None,
        as_preview: bool = False,
    ) -> Path:
        job_dir = self.artifact_root / job_id
        frames_dir = job_dir / "frames"
        masks_dir = job_dir / "masks"

        with (job_dir / "metadata.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        out_name = "preview.mp4" if as_preview else f"final_{effect}_{intensity}.mp4"
        out_path = job_dir / out_name
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            meta["fps"],
            (meta["width"], meta["height"]),
        )

        total = meta["total_frames"]
        for i in range(total):
            frame = cv2.imread(str(frames_dir / f"{i:06d}.png"))
            mask = np.load(masks_dir / f"{i:06d}.npy") > 0
            effected = self._apply_effect(frame, mask, effect, intensity)
            writer.write(effected)
            if progress and total > 0:
                start = 0.75 if as_preview else 0.0
                span = 0.23 if as_preview else 1.0
                progress(start + ((i + 1) / total) * span, f"Rendering frame {i+1}/{total}")

        writer.release()
        return out_path

    def _detect(self, image: Image.Image, prompt: str) -> list[np.ndarray]:
        text_queries = [[prompt]]
        inputs = self.detector_processor(text=text_queries, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detector_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.detector_processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.2,
            text_threshold=0.2,
        )[0]
        boxes = results["boxes"].detach().cpu().numpy() if len(results["boxes"]) else np.empty((0, 4))
        return [b.astype(np.float32) for b in boxes]

    def _segment(self, image_rgb: np.ndarray, boxes: list[np.ndarray]) -> list[np.ndarray]:
        if not boxes:
            return []
        boxes_list = [b.tolist() for b in boxes]
        inputs = self.segmenter_processor(image_rgb, input_boxes=[boxes_list], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.segmenter_model(**inputs)
        masks = self.segmenter_processor.post_process_masks(
            masks=outputs.pred_masks.cpu(),
            original_sizes=inputs.original_sizes.cpu(),
            reshaped_input_sizes=inputs.reshaped_input_sizes.cpu(),
        )[0]
        return [m.squeeze().numpy() for m in masks]

    def _associate_tracks(self, tracks: list[TrackState], detections: list[np.ndarray]):
        if not tracks or not detections:
            return [], list(range(len(detections)))

        cost = np.ones((len(tracks), len(detections)), dtype=np.float32)
        for ti, track in enumerate(tracks):
            for di, det in enumerate(detections):
                cost[ti, di] = 1.0 - _iou(track.bbox, det)

        row_ind, col_ind = linear_sum_assignment(cost)
        associations = []
        matched_dets = set()
        for r, c in zip(row_ind, col_ind):
            if 1.0 - cost[r, c] > 0.2:
                associations.append((r, c))
                matched_dets.add(c)

        unmatched = [i for i in range(len(detections)) if i not in matched_dets]
        return associations, unmatched

    def _merge_masks(self, masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
        if not masks:
            return np.zeros(shape, dtype=np.uint8)
        combined = np.zeros(shape, dtype=np.uint8)
        for m in masks:
            bin_m = (m > 0).astype(np.uint8)
            if bin_m.shape != shape:
                bin_m = cv2.resize(bin_m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            combined = np.maximum(combined, bin_m)
        return combined

    def _apply_effect(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        effect: Literal["blur", "pixelate", "solid", "outline", "dim_bg"],
        intensity: int,
    ) -> np.ndarray:
        out = frame.copy()
        if not mask.any():
            return out

        if effect == "blur":
            k = max(3, 2 * intensity + 1)
            blurred = cv2.GaussianBlur(frame, (k, k), 0)
            out[mask] = blurred[mask]
        elif effect == "pixelate":
            scale = max(2, min(30, intensity))
            h, w = frame.shape[:2]
            temp = cv2.resize(frame, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            out[mask] = pixelated[mask]
        elif effect == "solid":
            out[mask] = np.array([0, 0, 0], dtype=np.uint8)
        elif effect == "outline":
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (0, 255, 0), max(1, intensity // 4))
        elif effect == "dim_bg":
            dim = (frame * max(0.1, 1 - intensity / 60)).astype(np.uint8)
            out[~mask] = dim[~mask]
        return out
