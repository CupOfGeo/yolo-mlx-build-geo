#!/usr/bin/env python3
"""Phase 4 of the wardrobe PoC: render a static HTML report from
clusters.jsonl + outfits.jsonl.

Pipeline:
    clusters.jsonl + outfits.jsonl
        -> template into a single self-contained HTML file
        -> write to wardrobe/data/report.html
        -> open in a browser (file://...) to view

By default the report references image files by relative path so it stays
small. Use --inline to base64-embed every image into the HTML itself,
producing a single shareable file (much larger).

See wardrobe/README.md for the design.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("wardrobe.report")

SLOT_PLURALS = {"shirt": "shirts", "pants": "pants", "accessory": "accessories"}
SLOT_ORDER = ["shirt", "pants", "accessory"]


# ----- JSONL loader (tolerates pretty-printed) -----
def _load_jsonl(path: Path) -> list[dict]:
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


# ----- helpers -----
def fmt_ts(iso_ts: Optional[str]) -> str:
    """ISO '2026-05-24T21:49:04' -> '2026-05-24 21:49'."""
    if not iso_ts:
        return ""
    return iso_ts.replace("T", " ")[:16]


def esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


def rel_to_report(path_str: Optional[str], report_path: Path) -> Optional[str]:
    """Relative path from report.html's directory to the given image."""
    if not path_str:
        return None
    p = Path(path_str)
    try:
        return str(p.relative_to(report_path.parent))
    except ValueError:
        # Fall back to the original (works if user pre-resolved paths)
        return path_str


def img_src(path_str: Optional[str], report_path: Path, inline: bool) -> Optional[str]:
    """Build an <img src=...> value: data: URI when --inline, else relative path."""
    if not path_str:
        return None
    if inline:
        full = Path(path_str)
        if not full.exists():
            return None
        data = base64.b64encode(full.read_bytes()).decode()
        return f"data:image/jpeg;base64,{data}"
    return rel_to_report(path_str, report_path)


# ----- CSS (system fonts, neutral palette, subtle shadows) -----
CSS = """
:root {
  --bg: #fafaf8;
  --card: #ffffff;
  --text: #1a1a1a;
  --muted: #6b7280;
  --border: #e5e7eb;
  --shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03);
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.container { max-width: 1200px; margin: 0 auto; padding: 40px 24px 80px; }

header { padding-bottom: 24px; border-bottom: 1px solid var(--border); margin-bottom: 40px; }
h1 { font-size: 34px; font-weight: 700; margin: 0 0 6px 0; letter-spacing: -0.01em; }
.subtitle { color: var(--muted); font-size: 13px; }
.stat-row { display: flex; gap: 40px; margin-top: 24px; flex-wrap: wrap; }
.stat-value { font-size: 30px; font-weight: 700; line-height: 1.1; }
.stat-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }

section { margin: 48px 0; }
h2 { font-size: 22px; font-weight: 600; margin: 0 0 24px 0; letter-spacing: -0.005em; }

.slot-group { margin-bottom: 36px; }
.slot-header { display: flex; align-items: baseline; gap: 10px; margin-bottom: 14px; }
.slot-name { font-size: 15px; font-weight: 600; text-transform: capitalize; }
.slot-count { font-size: 13px; color: var(--muted); }

.cluster-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }

.cluster-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  box-shadow: var(--shadow);
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.cluster-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.cluster-thumb { aspect-ratio: 3 / 4; background: #f3f4f6; overflow: hidden; }
.cluster-thumb a { display: block; width: 100%; height: 100%; cursor: zoom-in; }
.cluster-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.cluster-thumb.empty { display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 12px; }
.cluster-body { padding: 12px 14px 14px; }
.cluster-label { font-weight: 500; font-size: 14px; line-height: 1.3; margin-bottom: 6px; }
.cluster-id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: var(--muted); margin-bottom: 10px; }
.wear-row { display: flex; align-items: center; gap: 8px; }
.wear-dots { display: flex; gap: 3px; }
.wear-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text); }
.wear-dot.empty { background: var(--border); }
.wear-num { font-size: 12px; color: var(--muted); }
.cluster-meta { margin-top: 8px; font-size: 11px; color: var(--muted); font-family: ui-monospace, monospace; }

.outfit-list { display: flex; flex-direction: column; gap: 16px; }
.outfit-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; box-shadow: var(--shadow); }
.outfit-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 14px; }
.outfit-ts { font-family: ui-monospace, monospace; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.outfit-summary { font-size: 16px; font-weight: 500; margin-top: 4px; line-height: 1.4; }
.outfit-confidence { font-size: 10px; padding: 3px 9px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; white-space: nowrap; }
.confidence-high { background: #d1fae5; color: #065f46; }
.confidence-medium { background: #fef3c7; color: #92400e; }
.confidence-low { background: #fee2e2; color: #991b1b; }

.outfit-thumbs { display: flex; gap: 8px; margin-bottom: 14px; overflow-x: auto; padding-bottom: 4px; }
.outfit-thumb { width: 76px; height: 96px; flex-shrink: 0; border-radius: 6px; overflow: hidden; background: #f3f4f6; border: 1px solid var(--border); }
.outfit-thumb a { display: block; width: 100%; height: 100%; cursor: zoom-in; }
.outfit-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.outfit-thumb:hover { border-color: var(--text); }

.outfit-pieces { display: flex; gap: 6px; flex-wrap: wrap; }
.piece-chip { font-size: 12px; padding: 5px 10px; border-radius: 6px; background: #f9fafb; border: 1px solid var(--border); display: inline-flex; align-items: center; gap: 6px; line-height: 1; }
.piece-slot { font-family: ui-monospace, monospace; font-size: 9px; text-transform: uppercase; color: var(--muted); letter-spacing: 0.05em; }
.piece-cid { font-family: ui-monospace, monospace; font-size: 10px; color: var(--muted); }

footer { margin-top: 56px; padding-top: 24px; border-top: 1px solid var(--border); text-align: center; font-size: 12px; color: var(--muted); }
footer code { font-family: ui-monospace, monospace; }
"""


def _wear_dots(count: int, cap: int = 10) -> tuple[str, str]:
    """Returns (dots_html, overflow_suffix). E.g. count=12 → (10 dots, '  +2')."""
    filled = min(count, cap)
    parts = ["<span class='wear-dot'></span>"] * filled
    if count < cap:
        parts += ["<span class='wear-dot empty'></span>"] * (cap - count)
    out = "".join(parts)
    suffix = f"  +{count - cap}" if count > cap else ""
    return out, suffix


def render_html(outfits: list[dict], clusters: list[dict], report_path: Path,
                inline: bool) -> str:
    n_clusters = len(clusters)
    n_outfits = len(outfits)
    n_wears = sum(c.get("wear_count", 0) for c in clusters)

    if outfits:
        first_ts = min(o["burst_start_ts"] for o in outfits)
        last_ts = max(o["burst_end_ts"] for o in outfits)
    else:
        first_ts = last_ts = ""

    # Map (burst_id, piece_idx) -> cluster_id for outfit chips
    piece_to_cluster: dict[tuple[str, int], str] = {}
    for c in clusters:
        for m in c.get("members", []):
            piece_to_cluster[(m["burst_id"], m["piece_idx"])] = c["cluster_id"]

    # Group clusters by slot, sorted by wear_count desc within each
    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in clusters:
        grouped[c["slot"]].append(c)
    for slot in grouped:
        grouped[slot].sort(
            key=lambda c: (c.get("wear_count", 0), c.get("last_seen", "")),
            reverse=True,
        )

    # Outfits newest-first
    outfits_sorted = sorted(outfits, key=lambda o: o["burst_start_ts"], reverse=True)

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append("<html lang='en'><head>")
    parts.append("<meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>Digital Wardrobe</title>")
    parts.append(f"<style>{CSS}</style>")
    parts.append("</head><body><div class='container'>")

    # Header + stats
    parts.append("<header>")
    parts.append("<h1>Digital Wardrobe</h1>")
    if first_ts and last_ts:
        parts.append(
            f"<div class='subtitle'>captured {esc(fmt_ts(first_ts))} → {esc(fmt_ts(last_ts))}</div>"
        )
    parts.append("<div class='stat-row'>")
    for value, label in [
        (n_clusters, "distinct pieces"),
        (n_outfits, "outfits"),
        (n_wears, "total wears"),
    ]:
        parts.append(
            f"<div><div class='stat-value'>{value}</div>"
            f"<div class='stat-label'>{esc(label)}</div></div>"
        )
    parts.append("</div></header>")

    # Wardrobe catalog (cluster gallery)
    parts.append("<section>")
    parts.append("<h2>Wardrobe catalog</h2>")
    slots_present = [s for s in SLOT_ORDER if s in grouped] + [
        s for s in grouped if s not in SLOT_ORDER
    ]
    for slot in slots_present:
        clist = grouped[slot]
        plural = SLOT_PLURALS.get(slot, slot + "s")
        parts.append("<div class='slot-group'>")
        parts.append(
            f"<div class='slot-header'>"
            f"<span class='slot-name'>{esc(plural)}</span>"
            f"<span class='slot-count'>{len(clist)}</span></div>"
        )
        parts.append("<div class='cluster-grid'>")
        for c in clist:
            label = c.get("user_label") or c.get("canonical_label", "")
            cid = c["cluster_id"]
            wc = c.get("wear_count", 0)
            last = fmt_ts(c.get("last_seen", ""))
            crop = img_src(c.get("representative_crop"), report_path, inline)
            dots, dots_suffix = _wear_dots(wc)
            parts.append("<div class='cluster-card'>")
            if crop:
                parts.append(
                    f"<div class='cluster-thumb'>"
                    f"<a href='{esc(crop)}' target='_blank' rel='noopener noreferrer' "
                    f"title='Open full image in new tab'>"
                    f"<img src='{esc(crop)}' loading='lazy' alt='{esc(label)}'>"
                    f"</a></div>"
                )
            else:
                parts.append("<div class='cluster-thumb empty'>no image</div>")
            parts.append("<div class='cluster-body'>")
            parts.append(f"<div class='cluster-label'>{esc(label)}</div>")
            parts.append(f"<div class='cluster-id'>{esc(cid)}</div>")
            parts.append(
                f"<div class='wear-row'>"
                f"<span class='wear-dots'>{dots}</span>"
                f"<span class='wear-num'>{wc}×{esc(dots_suffix)}</span></div>"
            )
            if last:
                parts.append(f"<div class='cluster-meta'>last {esc(last)}</div>")
            parts.append("</div></div>")
        parts.append("</div></div>")
    parts.append("</section>")

    # Outfit timeline
    parts.append("<section>")
    parts.append("<h2>Recent outfits</h2>")
    parts.append("<div class='outfit-list'>")
    for o in outfits_sorted:
        bid = o["burst_id"]
        outfit = o["outfit"]
        ts = fmt_ts(o["burst_start_ts"])
        summary = outfit.get("one_line_summary", "")
        conf = outfit.get("confidence", "medium")
        crops = o.get("member_crops", [])
        pieces = outfit.get("pieces", [])

        parts.append("<div class='outfit-card'>")
        parts.append("<div class='outfit-header'>")
        parts.append(
            f"<div><div class='outfit-ts'>{esc(ts)}</div>"
            f"<div class='outfit-summary'>{esc(summary)}</div></div>"
        )
        parts.append(
            f"<div class='outfit-confidence confidence-{esc(conf)}'>{esc(conf)}</div>"
        )
        parts.append("</div>")

        if crops:
            parts.append("<div class='outfit-thumbs'>")
            for cp in crops:
                src = img_src(cp, report_path, inline)
                if src:
                    parts.append(
                        f"<div class='outfit-thumb'>"
                        f"<a href='{esc(src)}' target='_blank' rel='noopener noreferrer' "
                        f"title='Open full image in new tab'>"
                        f"<img src='{esc(src)}' loading='lazy'></a></div>"
                    )
            parts.append("</div>")

        if pieces:
            parts.append("<div class='outfit-pieces'>")
            for idx, p in enumerate(pieces):
                slot = p.get("slot", "")
                ptype = p.get("type") or "?"
                color = p.get("primary_color") or ""
                label = f"{color} {ptype}".strip()
                cid = piece_to_cluster.get((bid, idx))
                cid_html = f"<span class='piece-cid'>{esc(cid)}</span>" if cid else ""
                parts.append(
                    f"<span class='piece-chip'>"
                    f"<span class='piece-slot'>{esc(slot)}</span>"
                    f"{esc(label)}{cid_html}</span>"
                )
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</div></section>")

    parts.append(
        "<footer>generated by <code>wardrobe/report.py</code> — "
        "digital wardrobe PoC</footer>"
    )
    parts.append("</div></body></html>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4: render a static HTML wardrobe report from "
            "clusters.jsonl + outfits.jsonl."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--outfits", type=Path, default=Path("wardrobe/data/outfits.jsonl"),
    )
    parser.add_argument(
        "--clusters", type=Path, default=Path("wardrobe/data/clusters.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("wardrobe/data/report.html"),
        help="Output HTML file path. Image src paths are relative to its parent dir.",
    )
    parser.add_argument(
        "--inline", action="store_true",
        help=(
            "Base64-embed all images into the HTML so it's a single shareable "
            "file. Much larger output (~MB per crop)."
        ),
    )
    parser.add_argument(
        "--open", action="store_true", dest="open_after",
        help="Open the report in the default browser after writing.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.outfits.exists():
        sys.exit(f"ERROR: outfits file not found: {args.outfits}")
    if not args.clusters.exists():
        sys.exit(
            f"ERROR: clusters file not found: {args.clusters}\n"
            f"Run `uv run python wardrobe/dedup.py` first to build the catalog."
        )

    outfits = _load_jsonl(args.outfits)
    clusters = _load_jsonl(args.clusters)
    logger.info(f"loaded {len(outfits)} outfit(s) and {len(clusters)} cluster(s)")

    html_str = render_html(outfits, clusters, args.output, args.inline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_str)
    size_kb = len(html_str) / 1024
    logger.info(
        f"wrote {size_kb:,.1f} KB to {args.output} "
        f"(inline={args.inline})"
    )

    abs_url = f"file://{args.output.resolve()}"
    print(f"\nOpen: {abs_url}")

    if args.open_after:
        import webbrowser
        webbrowser.open(abs_url)


if __name__ == "__main__":
    main()
