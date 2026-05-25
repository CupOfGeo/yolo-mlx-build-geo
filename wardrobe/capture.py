#!/usr/bin/env python3
"""Phase 1 of the wardrobe PoC: detect 'entry events' (a person walking through
the front door) from an RTSP stream and save best-quality crops to disk.

Pipeline:
    RTSP -> yolo26n + ByteTrack -> per-track best frame -> emit on track close

Each emitted event produces:
    wardrobe/data/crops/<stamp>_<trackid>_full.jpg   # full frame
    wardrobe/data/crops/<stamp>_<trackid>_crop.jpg   # padded person bbox crop
    wardrobe/data/events.jsonl                       # one appended JSON line

See wardrobe/README.md for the full design.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from urllib.parse import quote

import cv2
import numpy as np

from yolo26mlx import YOLO

PERSON_CLASS = 0  # COCO class index for "person"
logger = logging.getLogger("wardrobe.capture")


# ----- minimal .env loader (no external dep) -----
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# ----- RTSP URL assembly -----
def _build_rtsp_url() -> Optional[str]:
    """Construct an RTSP URL from env vars, URL-encoding the username and password.

    Preferred form (paste your RAW password — encoding is done for you):
        RTSP_USER, RTSP_PASS, RTSP_HOST  [+ optional RTSP_PORT, RTSP_PATH]

    Fallback: a fully pre-formed RTSP_URL (assumed already encoded).

    Returns None if neither form is set.
    """
    user = os.environ.get("RTSP_USER")
    password = os.environ.get("RTSP_PASS")
    host = os.environ.get("RTSP_HOST")
    if user and password and host:
        port = os.environ.get("RTSP_PORT", "554")
        path = os.environ.get("RTSP_PATH", "/cam/realmonitor?channel=1&subtype=0")
        # safe='' encodes everything that isn't unreserved — so '@', ':', '/', '#',
        # '&', '?', '%' in the password all get escaped.
        u = quote(user, safe="")
        p = quote(password, safe="")
        return f"rtsp://{u}:{p}@{host}:{port}{path}"
    return os.environ.get("RTSP_URL")


# ----- frame-quality scoring -----
def _sharpness(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Laplacian variance of the bbox region — a robust focus/blur metric.

    Higher = sharper. Typical values: < 50 motion-blurred, 50-200 mediocre,
    > 200 sharp. Insensitive to exposure but somewhat to scene texture (busy
    backgrounds inflate variance), so use as a relative score across frames
    of the same person against the same backdrop, not an absolute threshold.

    Frames from `result.orig_img` are RGB. cv2 gray conversion handles either.
    """
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ----- per-track open-event state -----
@dataclass
class Candidate:
    """One frame-of-interest from a track, ranked among the top-K by score."""
    frame: np.ndarray                       # full frame (RGB; predictor's convention)
    bbox: tuple[int, int, int, int]         # x1, y1, x2, y2 in frame pixels
    conf: float
    area_frac: float
    sharpness: float
    score: float                            # area_frac * conf * sharpness


@dataclass
class OpenTrack:
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    n_frames_seen: int = 0
    candidates: list[Candidate] = field(default_factory=list)  # sorted by score desc, len ≤ top_k

    @property
    def duration_frames(self) -> int:
        return self.last_seen_frame - self.first_seen_frame + 1

    @property
    def best(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _pad_bbox(bbox, pad_frac, frame_w, frame_h):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    px = int(w * pad_frac)
    py = int(h * pad_frac)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(frame_w, x2 + px),
        min(frame_h, y2 + py),
    )


def emit_event(
    track: OpenTrack,
    output_dir: Path,
    events_path: Path,
    pad_frac: float,
    min_area_frac: float,
    min_frames: int,
) -> bool:
    """Save top-K candidate crops + append JSONL row. Returns True if emitted."""
    if track.n_frames_seen < min_frames:
        logger.info(
            f"track {track.track_id}: dropped — only {track.n_frames_seen} frames "
            f"(< min-frames={min_frames})"
        )
        return False
    best = track.best
    if best is None:
        logger.info(f"track {track.track_id}: dropped — no candidates captured")
        return False
    if best.area_frac < min_area_frac:
        logger.info(
            f"track {track.track_id}: dropped — best area_frac {best.area_frac:.3f} "
            f"(< min-area-frac={min_area_frac})"
        )
        return False

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now_stamp()

    candidate_records: list[dict] = []
    for rank, c in enumerate(track.candidates):
        h, w = c.frame.shape[:2]
        px1, py1, px2, py2 = _pad_bbox(c.bbox, pad_frac, w, h)
        crop_pixels = c.frame[py1:py2, px1:px2]

        # The predictor converts incoming BGR -> RGB before storing as orig_img
        # (src/yolo26mlx/engine/predictor.py:186-188). Swap back for cv2.imwrite,
        # otherwise R and B end up swapped in the JPG.
        full_bgr = cv2.cvtColor(c.frame, cv2.COLOR_RGB2BGR)
        crop_bgr = cv2.cvtColor(crop_pixels, cv2.COLOR_RGB2BGR)

        full_path = crops_dir / f"{stamp}_{track.track_id}_{rank}_full.jpg"
        crop_path = crops_dir / f"{stamp}_{track.track_id}_{rank}_crop.jpg"
        cv2.imwrite(str(full_path), full_bgr)
        cv2.imwrite(str(crop_path), crop_bgr)

        candidate_records.append({
            "rank": rank,
            "full_path": str(full_path),
            "crop_path": str(crop_path),
            "bbox": [int(v) for v in c.bbox],
            "conf": round(float(c.conf), 4),
            "area_frac": round(float(c.area_frac), 4),
            "sharpness": round(float(c.sharpness), 2),
            "score": round(float(c.score), 4),
        })

    top = candidate_records[0]
    record = {
        "ts": _now_iso(),
        "track_id": int(track.track_id),
        "duration_frames": int(track.duration_frames),
        "n_frames_seen": int(track.n_frames_seen),
        # Top-level fields mirror candidates[0] for back-compat with old readers
        "best_conf": top["conf"],
        "bbox": top["bbox"],
        "area_frac": top["area_frac"],
        "sharpness": top["sharpness"],
        "full_path": top["full_path"],
        "crop_path": top["crop_path"],
        # New: full ranked list of candidates
        "candidates": candidate_records,
    }
    with events_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    logger.info(
        f"track {track.track_id}: EVENT  k={len(candidate_records)}  "
        f"best_conf={top['conf']:.2f}  best_area={top['area_frac']:.2%}  "
        f"best_sharp={top['sharpness']:.0f}  frames={track.n_frames_seen}  "
        f"-> {Path(top['crop_path']).name}"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1: detect entry events from an RTSP stream and save best-quality "
            "person crops to wardrobe/data/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=None,
        help="RTSP URL, video file path, or webcam index. Falls back to $RTSP_URL.",
    )
    parser.add_argument("--model", default="models/yolo26n.npz", help="Path to MLX weights")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--min-frames", type=int, default=10, dest="min_frames")
    parser.add_argument("--lost-frames", type=int, default=30, dest="lost_frames")
    parser.add_argument(
        "--min-area-frac",
        type=float,
        default=0.05,
        dest="min_area_frac",
        help="Drop tracks whose best frame has a person box smaller than this fraction",
    )
    parser.add_argument("--pad", type=float, default=0.1, help="Crop padding fraction")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        dest="top_k",
        help=(
            "Save top-K candidate frames per track, ranked by area*conf*sharpness "
            "(Laplacian variance). Default 3 gives angle + sharpness redundancy. "
            "Use 1 to keep only the single best (1/3 disk usage)."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("wardrobe/data"))
    parser.add_argument("--show", action="store_true", help="Live cv2 preview (press q to quit)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Open the source, read one frame, print shape, and exit. No model loaded.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _load_dotenv(Path("wardrobe/.env"))

    source = args.source or _build_rtsp_url()
    if not source:
        sys.exit(
            "ERROR: pass --source explicitly or set RTSP_USER + RTSP_PASS + RTSP_HOST "
            "(or RTSP_URL) in wardrobe/.env"
        )
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    src_display = source if not isinstance(source, str) else _redact_url(source)

    # --check mode: just open the source and confirm a frame is readable.
    if args.check:
        logger.info(f"checking source: {src_display}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f"FAIL: cannot open source {src_display!r}")
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            sys.exit(f"FAIL: opened {src_display!r} but read 0 frames")
        h, w = frame.shape[:2]
        print(f"OK stream readable: {w}x{h}  ({frame.shape})")
        return

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"

    logger.info(f"loading model {args.model}")
    model = YOLO(args.model)

    open_tracks: dict[int, OpenTrack] = {}
    frame_idx = 0
    interrupted = False

    def handle_sigint(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        logger.info("Ctrl+C received — flushing open tracks...")

    signal.signal(signal.SIGINT, handle_sigint)

    logger.info(f"opening source: {src_display}")

    try:
        try:
            stream = model.track(
                source,
                stream=True,
                persist=True,
                conf=args.conf,
                imgsz=args.imgsz,
            )
        except OSError as e:
            sys.exit(f"ERROR opening source {src_display!r}: {e}")

        for result in stream:
            frame_idx += 1
            frame = result.orig_img
            frame_h, frame_w = frame.shape[:2]
            frame_area = float(frame_h * frame_w)

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0 and boxes.is_track:
                xyxy = boxes.xyxy.astype(int)
                confs = boxes.conf
                classes = boxes.cls.astype(int)
                track_ids = boxes.id
                if track_ids is not None:
                    for i in range(len(boxes)):
                        if int(classes[i]) != PERSON_CLASS:
                            continue
                        tid = int(track_ids[i])
                        x1, y1, x2, y2 = xyxy[i]
                        bw = max(0, x2 - x1)
                        bh = max(0, y2 - y1)
                        bbox_area = float(bw * bh)
                        if bbox_area <= 0:
                            continue
                        area_frac = bbox_area / frame_area
                        conf = float(confs[i])
                        bbox = (int(x1), int(y1), int(x2), int(y2))

                        track = open_tracks.get(tid)
                        if track is None:
                            track = OpenTrack(
                                track_id=tid,
                                first_seen_frame=frame_idx,
                                last_seen_frame=frame_idx,
                            )
                            open_tracks[tid] = track
                            logger.info(f"track {tid}: OPENED at frame {frame_idx}")
                        track.last_seen_frame = frame_idx
                        track.n_frames_seen += 1

                        # Cheap pre-check: would this frame even displace the
                        # worst-ranked current candidate ignoring sharpness?
                        # Skip the expensive sharpness compute + frame copy if not.
                        cur = track.candidates
                        if len(cur) >= args.top_k:
                            # Worst candidate's score has a sharpness factor too;
                            # use a conservative bound: assume incoming frame
                            # could have very high sharpness. If even with that
                            # the area*conf component still trails the current
                            # worst's area*conf*sharpness divided by a large
                            # sharpness estimate, skip. In practice: only skip
                            # when area_frac*conf is < (worst_score / 2000).
                            min_score = cur[-1].score
                            optimistic = bbox_area * conf * 2000.0
                            if optimistic <= min_score:
                                continue

                        sharpness = _sharpness(frame, bbox)
                        score = bbox_area * conf * sharpness
                        if len(cur) >= args.top_k and score <= cur[-1].score:
                            continue

                        candidate = Candidate(
                            frame=frame.copy(),
                            bbox=bbox,
                            conf=conf,
                            area_frac=area_frac,
                            sharpness=sharpness,
                            score=score,
                        )
                        cur.append(candidate)
                        cur.sort(key=lambda c: c.score, reverse=True)
                        del cur[args.top_k:]

            # Close stale tracks (not seen for `lost_frames` consecutive frames)
            for tid in list(open_tracks):
                track = open_tracks[tid]
                if frame_idx - track.last_seen_frame > args.lost_frames:
                    open_tracks.pop(tid)
                    logger.info(
                        f"track {tid}: CLOSED  last_seen=frame{track.last_seen_frame}  "
                        f"now=frame{frame_idx}"
                    )
                    emit_event(
                        track,
                        output_dir,
                        events_path,
                        args.pad,
                        args.min_area_frac,
                        args.min_frames,
                    )

            if args.show:
                annotated = result.plot()  # RGB
                cv2.imshow(
                    "wardrobe.capture (q to quit)",
                    cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                )
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    interrupted = True

            if interrupted:
                break
    finally:
        # Flush any remaining open tracks on shutdown
        for tid in list(open_tracks):
            track = open_tracks.pop(tid)
            logger.info(f"track {tid}: FLUSH on shutdown")
            emit_event(
                track, output_dir, events_path, args.pad, args.min_area_frac, args.min_frames
            )
        if args.show:
            cv2.destroyAllWindows()
        logger.info(f"done. frames processed: {frame_idx}")


def _redact_url(s: str) -> str:
    """Hide credentials in RTSP URLs when logging."""
    if "://" in s and "@" in s:
        scheme, rest = s.split("://", 1)
        creds, host = rest.split("@", 1)
        return f"{scheme}://<redacted>@{host}"
    return s


if __name__ == "__main__":
    main()
