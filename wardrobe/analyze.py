#!/usr/bin/env python3
"""Phase 2 of the wardrobe PoC: group capture events into bursts and send each
burst's best crops to Claude vision for a structured outfit description.

Pipeline:
    events.jsonl -> burst grouping -> top-N crops per burst -> Claude vision ->
    Pydantic validation -> outfits.jsonl (append-only, one record per burst)

Idempotent: re-running skips bursts already present in outfits.jsonl (keyed on
burst_id). Pass --reprocess to override.

See wardrobe/README.md for the full design.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger("wardrobe.analyze")

MODEL = "claude-sonnet-4-6"


# ----- .env loader (no external dep, same as capture.py) -----
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


# ----- robust JSONL loader (tolerates pretty-printed objects too) -----
def _load_jsonl(path: Path) -> list[dict]:
    """Parse a stream of JSON objects from a file.

    Canonical JSONL is one object per line. We also handle the pretty-printed
    case (an editor's auto-format makes one object span many lines, no commas
    between objects) via json.JSONDecoder.raw_decode.
    """
    text = path.read_text().strip()
    if not text:
        return []
    decoder = json.JSONDecoder()
    items: list[dict] = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        items.append(obj)
        idx = end
    return items


# ----- Pydantic schema for the Claude response -----
#
# Wardrobe slots tracked independently. Each visible garment or accessory
# becomes one Piece record. In Phase 3, pieces from different sessions get
# matched against each other to build a catalog of distinct items the user
# owns (e.g. "navy crewneck tee" = 1 cluster, worn N times).
#
# TODO: add "shoes" to Slot once the camera angle is adjusted to reliably
#   capture feet. Right now the front-door camera frames torso + thighs and
#   cuts off below the knee, so shoe detection is unreliable. When enabling:
#     1. Add "shoes" to the Slot literal below.
#     2. Update SYSTEM_PROMPT to remove the "ignore items below the knee"
#        rule and add shoe-specific guidance (sneakers/boots/sandals/etc).
#     3. Consider whether socks are worth tracking (probably not).
Slot = Literal["shirt", "pants", "accessory"]


class Piece(BaseModel):
    slot: Slot = Field(description="Which wardrobe category this piece belongs to")
    type: str = Field(
        description=(
            "Generic garment type. "
            "shirt: 'crewneck t-shirt', 'long-sleeve henley', 'flannel button-down', "
            "'hoodie', 'cardigan'. "
            "pants: 'jeans', 'chinos', 'joggers', 'shorts', 'sweatpants'. "
            "accessory: 'baseball cap', 'beanie', 'glasses', 'watch', 'ring', "
            "'necklace', 'backpack', 'tote bag', 'scarf'."
        )
    )
    primary_color: Optional[str] = Field(
        None,
        description=(
            "Plain-English color name: 'navy blue', 'olive green', 'rust', 'cream'. "
            "Null when color isn't a useful descriptor (e.g. metallic jewelry — "
            "use notes instead)."
        ),
    )
    pattern: Optional[str] = Field(
        None,
        description=(
            "'solid', 'striped', 'graphic_logo', 'plaid', 'tie_dye', etc. "
            "Null if not applicable (e.g. most accessories)."
        ),
    )
    notes: Optional[str] = Field(
        None,
        description=(
            "Distinctive details that help re-identify THIS specific piece in "
            "future captures: 'small chest logo', 'gold band', 'rolled cuffs', "
            "'distressed knees', 'leather strap'."
        ),
    )


class Outfit(BaseModel):
    pieces: list[Piece] = Field(
        description=(
            "One Piece record per visible garment or accessory. Order: shirt(s) "
            "first, then pants, then accessories. Do NOT include pieces you "
            "cannot clearly see."
        )
    )
    one_line_summary: str = Field(
        description=(
            "How the user would describe the outfit to a friend: "
            "'navy graphic tee, dark jeans, olive cap'."
        )
    )
    confidence: Literal["low", "medium", "high"]
    notes: Optional[str] = Field(
        None,
        description="Outfit-level ambiguities, lighting issues, anything worth flagging.",
    )


SYSTEM_PROMPT = """You analyze photos from a static front-door security camera in a single-occupant apartment. All images you receive in one request are of the same person at the same moment, captured from up to four different angles as they pass through the doorway.

Your job: identify each individual garment and accessory visible in the images and produce one structured Piece record per item. The user is building a long-term wardrobe log — pieces from this session will be matched against pieces from other sessions to track distinct items over time, so consistency in how you describe garments matters more than poetic richness.

What counts as a "piece" and which slot it belongs to:
- "shirt" — any torso garment: t-shirt, long-sleeve, hoodie, button-down, sweater, jacket. If layered (tee under an open jacket), record each as a separate piece.
- "pants" — any leg garment: jeans, joggers, shorts, sweatpants, chinos, skirts.
- "accessory" — anything else worn or carried: hats, caps, glasses, watches, rings, necklaces, bags, scarves, ties, headphones.

Items below the knee (shoes, socks) are NOT in scope for this camera — the framing cuts off above the feet. If shoes happen to be visible at the edge of a frame, ignore them; do not record them.

Rules:
1. Synthesize across all provided images. Different angles reveal different details — a chest logo in the front view, a back graphic or silhouette in the side view.
2. Use plain English color names humans use: "navy blue", "olive green", "rust", "cream". No hex codes, no photography jargon.
3. Use generic garment types, not brand guesses: "navy crewneck t-shirt", not "Patagonia organic cotton tee". Brand identification is out of scope.
4. If a piece is clearly visible, include it. If it's only partially visible but identifiable, include it and reflect the uncertainty in `notes`. If it's not visible at all, omit it. Do NOT hallucinate pieces.
5. Set overall `confidence` honestly:
   - "high": multiple angles agree, lighting decent, all visible pieces clearly readable
   - "medium": partial views, single angle, or one ambiguous piece
   - "low": poor lighting, motion blur, conflicting signals
6. Treat the camera's auto white-balance as imperfect. If a color reads slightly blue- or warm-shifted, flag the uncertainty in the piece's `notes` rather than committing to a wrong color.
7. `one_line_summary` should sound like how the user would describe their outfit to a friend: "navy graphic tee, dark jeans, olive cap"."""


# ----- burst grouping -----
@dataclass
class Burst:
    events: list[dict]

    @property
    def burst_id(self) -> str:
        """Stable ID derived from the first event's timestamp."""
        return self.events[0]["ts"].replace(":", "").replace("-", "").replace("T", "_")

    @property
    def start_ts(self) -> str:
        return self.events[0]["ts"]

    @property
    def end_ts(self) -> str:
        return self.events[-1]["ts"]


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def group_into_bursts(events: list[dict], window_seconds: int) -> list[Burst]:
    """Sort by timestamp; split when gap > window_seconds."""
    if not events:
        return []
    sorted_events = sorted(events, key=lambda e: _parse_ts(e["ts"]))
    bursts: list[Burst] = []
    current = [sorted_events[0]]
    for ev in sorted_events[1:]:
        gap = (_parse_ts(ev["ts"]) - _parse_ts(current[-1]["ts"])).total_seconds()
        if gap <= window_seconds:
            current.append(ev)
        else:
            bursts.append(Burst(events=current))
            current = [ev]
    bursts.append(Burst(events=current))
    return bursts


def _candidates_for_event(ev: dict) -> list[dict]:
    """Yield candidate-crop dicts for an event.

    New-schema events (post-sharpness-scoring) carry a `candidates` list with
    per-candidate `conf`, `area_frac`, `sharpness`, `crop_path`. Old-schema
    events only have a single top-level `crop_path` — wrap it so the picker
    can score it uniformly (sharpness=0 -> falls back to area*conf ordering).
    """
    raw = ev.get("candidates")
    if raw:
        return [
            {
                "crop_path": c["crop_path"],
                "conf": float(c.get("conf", 0)),
                "area_frac": float(c.get("area_frac", 0)),
                "sharpness": float(c.get("sharpness", 0)),
                "_event_ts": ev["ts"],
                "_event_tid": ev["track_id"],
                "_rank": c.get("rank", 0),
            }
            for c in raw
        ]
    return [
        {
            "crop_path": ev["crop_path"],
            "conf": float(ev.get("best_conf", 0)),
            "area_frac": float(ev.get("area_frac", 0)),
            "sharpness": float(ev.get("sharpness", 0)),  # may be present on top-level too
            "_event_ts": ev["ts"],
            "_event_tid": ev["track_id"],
            "_rank": 0,
        }
    ]


def pick_top_crops(burst: Burst, max_images: int) -> list[dict]:
    """Pool candidate crops across all events in the burst, return top-N globally.

    Scoring prefers sharper crops when sharpness is known (new schema). For
    old-schema events lacking sharpness, falls back to area*conf — those will
    rank below any sharp new-schema candidate, which is the desired behavior
    once sharper data is available.
    """
    pool: list[dict] = []
    for ev in burst.events:
        pool.extend(_candidates_for_event(ev))

    def _score(c: dict) -> float:
        s = c["sharpness"]
        if s > 0:
            return s * c["conf"] * c["area_frac"]
        # No sharpness info -> use the legacy heuristic, scaled down so any
        # candidate WITH sharpness wins automatically.
        return c["conf"] * c["area_frac"]

    pool.sort(key=_score, reverse=True)
    return pool[:max_images]


# ----- Claude vision call -----
def _to_image_block(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }


def analyze_burst(
    client: anthropic.Anthropic, burst: Burst, max_images: int
) -> dict:
    """Call Claude vision on the burst's top-N crops, return a record dict."""
    top = pick_top_crops(burst, max_images)
    image_paths = [Path(e["crop_path"]) for e in top]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing crops: {missing}")

    content = [
        *(_to_image_block(p) for p in image_paths),
        {
            "type": "text",
            "text": (
                f"Analyze the outfit in the {len(image_paths)} attached image(s) "
                f"and return one structured outfit record per the schema."
            ),
        },
    ]

    logger.info(
        f"burst {burst.burst_id}: calling Claude with {len(image_paths)} image(s)"
    )
    t0 = time.monotonic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": content}],
        output_format=Outfit,
    )
    elapsed = time.monotonic() - t0
    outfit: Outfit = response.parsed_output
    u = response.usage
    logger.info(
        f"burst {burst.burst_id}: {outfit.one_line_summary!r}  "
        f"conf={outfit.confidence}  {elapsed:.1f}s  "
        f"in={u.input_tokens} cw={u.cache_creation_input_tokens} "
        f"cr={u.cache_read_input_tokens} out={u.output_tokens}"
    )

    return {
        "burst_id": burst.burst_id,
        "burst_start_ts": burst.start_ts,
        "burst_end_ts": burst.end_ts,
        "member_event_count": len(burst.events),
        "member_crops": [str(p) for p in image_paths],
        "model": MODEL,
        "outfit": outfit.model_dump(),
        "usage": {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_creation_input_tokens": u.cache_creation_input_tokens,
            "cache_read_input_tokens": u.cache_read_input_tokens,
        },
        "latency_seconds": round(elapsed, 2),
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
    }


# ----- main -----
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2: group capture events into bursts and send each burst's "
            "best crops to Claude vision for a structured outfit description."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--events", type=Path, default=Path("wardrobe/data/events.jsonl")
    )
    parser.add_argument(
        "--outfits", type=Path, default=Path("wardrobe/data/outfits.jsonl")
    )
    parser.add_argument(
        "--window",
        type=int,
        default=300,
        help=(
            "Burst window in seconds — events within this gap merge into one burst. "
            "Default 5 min handles typical in/out patterns while splitting on a "
            "realistic outfit-change interval. Lower for rapid same-outfit testing."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=4,
        dest="max_images",
        help="Maximum crops to send to Claude per burst",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print bursts + which crops would be sent. No API calls, no writes.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-analyze bursts already present in outfits.jsonl",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Loads RTSP_URL etc. — never overwrites an already-set env var, so
    # ANTHROPIC_API_KEY from your shell takes precedence.
    _load_dotenv(Path("wardrobe/.env"))

    if not args.events.exists():
        sys.exit(f"ERROR: events file not found: {args.events}")

    try:
        events = _load_jsonl(args.events)
    except json.JSONDecodeError as e:
        sys.exit(f"ERROR: could not parse {args.events}: {e}")
    logger.info(f"loaded {len(events)} events from {args.events}")

    bursts = group_into_bursts(events, args.window)
    logger.info(
        f"grouped into {len(bursts)} bursts (window={args.window}s, "
        f"max_images={args.max_images})"
    )

    # Skip already-processed bursts unless --reprocess
    already_done: set[str] = set()
    if args.outfits.exists() and not args.reprocess:
        for rec in _load_jsonl(args.outfits):
            if "burst_id" in rec:
                already_done.add(rec["burst_id"])
    pending = [b for b in bursts if b.burst_id not in already_done]
    logger.info(
        f"{len(pending)} bursts pending  "
        f"({len(bursts) - len(pending)} already in {args.outfits.name})"
    )

    if args.dry_run:
        for b in pending:
            pool_size = sum(len(_candidates_for_event(ev)) for ev in b.events)
            top = pick_top_crops(b, args.max_images)
            print(
                f"\nburst {b.burst_id}  events={len(b.events)}  "
                f"pool={pool_size} candidate(s)  "
                f"sending top {len(top)}:"
            )
            for c in top:
                # `sharpness=0` indicates an old-schema event without sharpness info
                sharp_str = f"{c['sharpness']:6.1f}" if c["sharpness"] else "  n/a"
                print(
                    f"  conf={c['conf']:.2f}  area={c['area_frac']:.2%}  "
                    f"sharp={sharp_str}  rank={c['_rank']}  "
                    f"src=evt{c['_event_tid']}@{c['_event_ts']}  {c['crop_path']}"
                )
        return

    if not pending:
        logger.info("nothing to do")
        return

    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit(
            "ERROR: ANTHROPIC_API_KEY not set — export it in your shell or add "
            "to wardrobe/.env"
        )

    client = anthropic.Anthropic()
    args.outfits.parent.mkdir(parents=True, exist_ok=True)
    with args.outfits.open("a") as fh:
        for burst in pending:
            try:
                record = analyze_burst(client, burst, args.max_images)
            except anthropic.APIStatusError as e:
                logger.error(
                    f"burst {burst.burst_id}: API error {e.status_code}: {e.message}"
                )
                continue
            except Exception as e:
                logger.error(f"burst {burst.burst_id}: failed: {e!r}")
                continue
            fh.write(json.dumps(record) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
