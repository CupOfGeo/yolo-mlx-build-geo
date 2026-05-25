#!/usr/bin/env python3
"""Phase 3 of the wardrobe PoC: cluster outfit pieces from outfits.jsonl into
a wardrobe catalog of distinct garments and accessories.

Pipeline:
    outfits.jsonl
        -> per-piece text representation ("navy blue solid crewneck t-shirt")
        -> sentence-transformers MiniLM embedding (384-dim)
        -> match against existing cluster centroids of the same slot
        -> if best cosine similarity >= --threshold (default 0.85), JOIN
           that cluster and update its centroid (running mean of normalized
           vectors); otherwise open a NEW cluster
        -> append to wardrobe/data/clusters.jsonl

Incremental and idempotent: re-runs only consider pieces not yet in any
cluster's `members` list. Cluster IDs (e.g. "shirt_001") are stable across
re-runs. Use --reset to rebuild from scratch.

See wardrobe/README.md for the full design.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("wardrobe.dedup")

# 384-dim, ~80 MB download on first use. Cached locally in ~/.cache/huggingface/.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Pluralization for the per-slot report headers — naive `slot + "s"` produces
# "pantss" / "accessorys". Hardcoded since the Slot enum is tiny.
SLOT_PLURALS = {"shirt": "shirts", "pants": "pants", "accessory": "accessories"}

# Suppress the noisy "TOKENIZERS_PARALLELISM" warning from HF tokenizers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ----- robust JSONL loader (tolerates pretty-printed objects) -----
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


# ----- piece text representation -----
def piece_text(piece: dict) -> str:
    """Canonical text for embedding a piece. Notes are intentionally excluded
    (too high-variance — would prevent the same shirt described two slightly
    different ways from clustering).

    Examples:
        "navy blue solid crewneck t-shirt"
        "dark navy blue graphic_logo baseball cap"
        "gold ring"       (color is null for metallic jewelry → field omitted)
    """
    parts: list[str] = []
    for field_name in ("primary_color", "pattern", "type"):
        v = piece.get(field_name)
        if v:
            parts.append(str(v).strip())
    return " ".join(parts).strip().lower() or "(empty)"


# ----- cosine similarity (cosine() handles unnormalized inputs) -----
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


# ----- per-cluster state -----
@dataclass
class Cluster:
    cluster_id: str            # "shirt_001", "pants_003", etc.
    slot: str
    canonical_label: str       # most-common piece_text across members
    user_label: Optional[str]  # set via manual rename, future Phase 4 work
    centroid: np.ndarray       # 384-dim, unit-normalized
    members: list[dict] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    representative_crop: Optional[str] = None

    def to_record(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "slot": self.slot,
            "canonical_label": self.canonical_label,
            "user_label": self.user_label,
            "wear_count": len(self.members),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "representative_crop": self.representative_crop,
            "members": self.members,
            "centroid": [round(float(x), 6) for x in self.centroid],
        }

    @classmethod
    def from_record(cls, rec: dict) -> "Cluster":
        return cls(
            cluster_id=rec["cluster_id"],
            slot=rec["slot"],
            canonical_label=rec["canonical_label"],
            user_label=rec.get("user_label"),
            centroid=np.array(rec["centroid"], dtype=np.float32),
            members=rec.get("members", []),
            first_seen=rec.get("first_seen", ""),
            last_seen=rec.get("last_seen", ""),
            representative_crop=rec.get("representative_crop"),
        )

    def add(self, piece: dict, embedding: np.ndarray, burst: dict, piece_idx: int) -> None:
        """Append a member; refresh centroid (running mean of unit vectors),
        timestamps, canonical label, and representative crop.
        """
        n = len(self.members)
        emb_unit = _normalize(embedding)
        self.centroid = _normalize(self.centroid * n + emb_unit) if n > 0 else emb_unit

        self.members.append({
            "burst_id": burst["burst_id"],
            "burst_ts": burst["burst_start_ts"],
            "piece_idx": piece_idx,
            "type": piece.get("type"),
            "primary_color": piece.get("primary_color"),
            "pattern": piece.get("pattern"),
            "notes": piece.get("notes"),
            "text": piece_text(piece),
        })

        ts = burst["burst_start_ts"]
        if not self.first_seen or ts < self.first_seen:
            self.first_seen = ts
        if not self.last_seen or ts > self.last_seen:
            self.last_seen = ts

        # Use the most recently added burst's first crop as the representative.
        # Simple heuristic; future Phase 4 might let the user pick.
        if burst.get("member_crops"):
            self.representative_crop = burst["member_crops"][0]

        # Refresh canonical label = most common piece_text across members
        labels = [m["text"] for m in self.members]
        self.canonical_label = Counter(labels).most_common(1)[0][0]


# ----- catalog (cluster store) -----
@dataclass
class Catalog:
    clusters: dict[str, Cluster] = field(default_factory=dict)  # id -> Cluster
    seq_by_slot: dict[str, int] = field(default_factory=dict)   # slot -> next int

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        cat = cls()
        if not path.exists():
            return cat
        for rec in _load_jsonl(path):
            c = Cluster.from_record(rec)
            cat.clusters[c.cluster_id] = c
            try:
                n = int(c.cluster_id.rsplit("_", 1)[-1])
            except ValueError:
                n = 0
            cat.seq_by_slot[c.slot] = max(cat.seq_by_slot.get(c.slot, 0), n)
        return cat

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for c in self.clusters.values():
                fh.write(json.dumps(c.to_record()) + "\n")

    def new_cluster_id(self, slot: str) -> str:
        n = self.seq_by_slot.get(slot, 0) + 1
        self.seq_by_slot[slot] = n
        return f"{slot}_{n:03d}"

    def slot_clusters(self, slot: str) -> list[Cluster]:
        return [c for c in self.clusters.values() if c.slot == slot]


def already_processed_keys(catalog: Catalog) -> set[tuple[str, int]]:
    """Set of (burst_id, piece_idx) pairs already living in some cluster."""
    return {
        (m["burst_id"], m["piece_idx"])
        for c in catalog.clusters.values()
        for m in c.members
    }


# ----- the core matching loop -----
def process(
    outfits: list[dict],
    catalog: Catalog,
    model: SentenceTransformer,
    threshold: float,
) -> dict:
    processed = already_processed_keys(catalog)
    summary = {"new_pieces": 0, "joined": 0, "new_clusters": 0, "decisions": []}

    for burst in outfits:
        bid = burst["burst_id"]
        for piece_idx, piece in enumerate(burst["outfit"]["pieces"]):
            if (bid, piece_idx) in processed:
                continue
            summary["new_pieces"] += 1

            slot = piece["slot"]
            text = piece_text(piece)
            emb = model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            best_id, best_sim = None, -1.0
            for c in catalog.slot_clusters(slot):
                sim = cosine(c.centroid, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id = c.cluster_id

            # Always mutate the in-memory catalog so dry-run correctly simulates
            # the cluster-building order (later pieces should see earlier ones).
            # Only the on-disk save is gated by dry_run.
            if best_id is not None and best_sim >= threshold:
                decision = "join"
                catalog.clusters[best_id].add(piece, emb, burst, piece_idx)
                target = best_id
                summary["joined"] += 1
            else:
                decision = "new"
                target = catalog.new_cluster_id(slot)
                new = Cluster(
                    cluster_id=target,
                    slot=slot,
                    canonical_label=text,
                    user_label=None,
                    centroid=_normalize(emb),
                )
                new.add(piece, emb, burst, piece_idx)
                catalog.clusters[target] = new
                summary["new_clusters"] += 1

            summary["decisions"].append({
                "burst_id": bid,
                "piece_idx": piece_idx,
                "slot": slot,
                "text": text,
                "decision": decision,
                "target": target,
                "best_existing_sim": round(best_sim, 4) if best_sim > -1 else None,
            })

    return summary


# ----- catalog summary printer -----
def print_report(catalog: Catalog) -> None:
    if not catalog.clusters:
        print("\nNo clusters yet. Run `wardrobe/dedup.py` to build the catalog.")
        return

    n_clusters = len(catalog.clusters)
    n_members = sum(len(c.members) for c in catalog.clusters.values())
    bursts = {m["burst_id"] for c in catalog.clusters.values() for m in c.members}

    print(
        f"\nWardrobe catalog: {n_clusters} distinct piece(s) across "
        f"{len(bursts)} outfit(s), {n_members} total wear(s).\n"
    )

    for slot in sorted({c.slot for c in catalog.clusters.values()}):
        clusters = sorted(
            catalog.slot_clusters(slot),
            key=lambda c: (len(c.members), c.last_seen),
            reverse=True,
        )
        label = SLOT_PLURALS.get(slot, slot + "s")
        print(f"{label} ({len(clusters)}):")
        for c in clusters:
            label = c.user_label or c.canonical_label
            n = len(c.members)
            # Trim ISO ts to "YYYY-MM-DD HH:MM" for readability
            last = c.last_seen.replace("T", " ")[:16] if c.last_seen else "?"
            print(f"  {c.cluster_id:14s}  {label:48s}  worn {n}×  last={last}")
        print()


# ----- main -----
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3: cluster outfit pieces from outfits.jsonl into a "
            "wardrobe catalog (clusters.jsonl). Incremental — re-runs only "
            "process pieces not yet clustered."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outfits", type=Path, default=Path("wardrobe/data/outfits.jsonl"))
    parser.add_argument("--clusters", type=Path, default=Path("wardrobe/data/clusters.jsonl"))
    parser.add_argument(
        "--threshold", type=float, default=0.89,
        help=(
            "Cosine similarity threshold for joining an existing cluster. "
            "0.89 is conservative — splits same-pattern-different-color pieces "
            "(e.g. green vs black graphic tees) while keeping legit color-name "
            "synonyms together (navy/steel-blue tees, navy/dark-indigo jeans). "
            "Drop to 0.80 for more merging, raise to 0.92+ for stricter."
        ),
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe clusters.jsonl and rebuild from outfits.jsonl",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show match decisions without writing the catalog",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print the wardrobe catalog summary and exit (no processing)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence the chatty model-download / batch logs unless --verbose
    if not args.verbose:
        for noisy in ("httpx", "huggingface_hub", "huggingface_hub.utils._http",
                      "sentence_transformers", "sentence_transformers.base.model",
                      "transformers", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    # --report mode: print + exit, no model load
    if args.report:
        catalog = Catalog.load(args.clusters)
        print_report(catalog)
        return

    if not args.outfits.exists():
        sys.exit(f"ERROR: outfits file not found: {args.outfits}")

    # --reset deletes the catalog on disk before reloading (skip on dry-run
    # so a dry --reset shows what would happen without actually wiping)
    if args.reset and args.clusters.exists() and not args.dry_run:
        args.clusters.unlink()
        logger.info(f"--reset: removed {args.clusters}")

    outfits = _load_jsonl(args.outfits)
    logger.info(f"loaded {len(outfits)} outfit record(s) from {args.outfits}")

    catalog = Catalog() if args.reset else Catalog.load(args.clusters)
    if catalog.clusters:
        existing = sum(len(c.members) for c in catalog.clusters.values())
        logger.info(
            f"loaded {len(catalog.clusters)} existing cluster(s) "
            f"with {existing} member piece(s)"
        )

    logger.info(f"loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    logger.info(f"model ready; threshold={args.threshold}, dry_run={args.dry_run}")

    summary = process(outfits, catalog, model, args.threshold)

    logger.info(
        f"processed: new_pieces={summary['new_pieces']}, "
        f"joined={summary['joined']}, new_clusters={summary['new_clusters']}"
    )
    for d in summary["decisions"]:
        action = "JOIN" if d["decision"] == "join" else "NEW "
        sim_str = f"{d['best_existing_sim']:.3f}" if d["best_existing_sim"] is not None else " n/a "
        logger.info(
            f"  {action} {d['slot']:9s} [{d['text']:50s}] "
            f"-> {d['target']:14s} (best_sim={sim_str})"
        )

    if args.dry_run:
        logger.info("dry-run — clusters.jsonl not written")
    else:
        catalog.save(args.clusters)
        logger.info(f"wrote {len(catalog.clusters)} cluster(s) to {args.clusters}")

    print_report(catalog)


if __name__ == "__main__":
    main()
