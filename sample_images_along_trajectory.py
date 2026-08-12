#!/usr/bin/env python3
"""Sample images along the inspection-1 trajectory at a fixed distance interval.

Inspection 1 lives in the shared ``camera_init`` (FastLIO) global frame; each
left/right capture pair shares an identical pose, so the trajectory of distinct
viewpoints is the sequence of unique poses ordered by capture id.

This script walks that trajectory in capture order and greedily picks a
viewpoint whenever the cumulative arc-length (sum of consecutive Euclidean
translation deltas) since the last picked viewpoint is >= ``--interval-m``.
The LEFT image of each picked pair (the lower capture id, see
``match_images_by_pose.py`` for why LEFT) is copied to ``--out-dir`` as
``<id>.jpg`` - the exact layout ``match_images_by_pose.py`` consumes as its
source set.

Run with the backend venv::

    python backend/scripts/sample_images_along_trajectory.py \
        --inspection 1 --interval-m 1.0 --out-dir "MTR Inspection Database/sampled_images"

A ``--json`` file listing the picked ids and arc-length positions is also
written for diagnostics / reproducibility.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _REPO_ROOT / "MTR Inspection Database" / "inspection_v2.db"
_DEFAULT_IMAGE_DIR = _REPO_ROOT / "MTR Inspection Database" / "outputs" / "images"
_DEFAULT_OUT_DIR = _REPO_ROOT / "MTR Inspection Database" / "sampled_images"


def _load_pairs(
    conn: sqlite3.Connection, inspection_id: int
) -> tuple[list[dict[str, Any]], list[int]]:
    """Distinct pose viewpoints of one inspection, ordered by capture id.

    Consecutive captures with identical (translation, quaternion, timestamp)
    form one L/R pair sharing one pose. Returns ``(views, unpaired_ids)``:
    each view has ``left_id`` (lower id = LEFT), ``right_id`` (higher id, or
    None if the pair was dropped), and the pose fields. Unpaired singletons
 122.78 deg,   are reported separately and excluded from sampling.
    """
    raw = conn.execute(
        """
        SELECT id, timestamp_ns,
               tf_translation_x AS tx, tf_translation_y AS ty, tf_translation_z AS tz,
               tf_rotation_x AS rx, tf_rotation_y AS ry,
               tf_rotation_z AS rz, tf_rotation_w AS rw
        FROM images
        WHERE inspection_id = ? AND tf_translation_x IS NOT NULL
                                  AND tf_rotation_w IS NOT NULL
        ORDER BY id
        """,
        (inspection_id,),
    ).fetchall()

    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in raw:
        key = (
            round(r["tx"], 4), round(r["ty"], 4), round(r["tz"], 4),
            round(r["rx"], 6), round(r["ry"], 6),
            round(r["rz"], 6), round(r["rw"], 6),
            r["timestamp_ns"],
        )
        groups.setdefault(key, []).append(dict(r))

    views: list[dict[str, Any]] = []
    unpaired: list[int] = []
    for key, grp in groups.items():
        grp.sort(key=lambda d: d["id"])
        v = dict(grp[0])
        v["left_id"] = grp[0]["id"]
        v["right_id"] = grp[1]["id"] if len(grp) >= 2 else None
        if len(grp) == 1:
            unpaired.append(grp[0]["id"])
        elif len(grp) > 2:
            print(f"[warn] inspection {inspection_id}: pose group {key} has "
                  f"{len(grp)} images (ids {[g['id'] for g in grp]}); "
                  f"using first two as L/R.", file=sys.stderr)
        views.append(v)

    views.sort(key=lambda d: d["left_id"])
    return views, unpaired


def _sample_by_arclength(
    views: list[dict[str, Any]], interval_m: float, start_index: int
) -> list[dict[str, Any]]:
    """Greedy fixed-interval sampling along the cumulative arc-length trajectory.

    Arc length is the running sum of Euclidean translation deltas between
    consecutive viewpoints in capture order. The first viewpoint is always
    picked (at arc-length 0); thereafter a viewpoint is picked whenever the
    arc length has advanced by >= ``interval_m`` since the last pick. The final
    viewpoint is always picked so the end of the route is represented.
    """
    if not views:
        return []

    n = len(views)
    for i in range(n):
        p = views[i]
        p["arclength"] = 0.0 if i == 0 else (
            views[i - 1]["arclength"]
            + math.sqrt(
                (p["tx"] - views[i - 1]["tx"]) ** 2
                + (p["ty"] - views[i - 1]["ty"]) ** 2
                + (p["tz"] - views[i - 1]["tz"]) ** 2
            )
        )

    start_index = max(0, min(start_index, n - 1))
    picked: list[dict[str, Any]] = [views[start_index]]
    last_arc = views[start_index]["arclength"]
    for v in views[start_index + 1:]:
        if v["arclength"] - last_arc >= interval_m:
            picked.append(v)
            last_arc = v["arclength"]
    if picked[-1] is not views[-1]:
        picked.append(views[-1])
    return picked


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default=str(_DEFAULT_DB), help="Path to the inspection .db")
    p.add_argument("--inspection", type=int, default=1, help="Inspection id to sample from (default 1)")
    p.add_argument("--interval-m", type=float, default=1.0,
                   help="Fixed arc-length interval between sampled viewpoints in metres (default 1.0)")
    p.add_argument("--start-index", type=int, default=0,
                   help="Index (in capture order) of the first viewpoint to pick from (default 0)")
    p.add_argument("--lens", choices=["left", "right", "both"], default="left",
                   help="Which lens of each L/R pair to copy: left (default), right, or both")
    p.add_argument("--image-dir", default=str(_DEFAULT_IMAGE_DIR),
                   help="Source image folder containing <id>.jpg for all images")
    p.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR),
                   help="Destination folder; <id>.jpg files are written here (cleared first by default)")
    p.add_argument("--keep-existing", action="store_true",
                   help="Do not clear --out-dir before writing (default: clear it)")
    p.add_argument("--no-copy", action="store_true",
                   help="Do not copy files; only compute the sampled set and write --json")
    p.add_argument("--json", dest="json_path", default=None,
                   help="Write the sampled list (ids + arc-lengths) to this JSON file")
    args = p.parse_args(argv)

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"[error] database not found: {db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        views, unpaired = _load_pairs(conn, args.inspection)
        if not views:
            print(f"[error] no pose-bearing viewpoints in inspection {args.inspection}", file=sys.stderr)
            return 2
        if unpaired:
            print(f"[info] inspection {args.inspection}: {len(unpaired)} unpaired (lens-ambiguous) "
                  f"image(s) excluded: {unpaired}", file=sys.stderr)

        total_len = views[-1]["arclength"] if len(views) == 1 and "arclength" in views[0] else 0.0
        picked = _sample_by_arclength(views, args.interval_m, args.start_index)
        total_len = picked[-1]["arclength"] if picked else 0.0

        sampled_ids: list[int] = []
        for v in picked:
            if args.lens == "left":
                sampled_ids.append(v["left_id"])
            elif args.lens == "right":
                if v["right_id"] is None:
                    print(f"[warn] view left_id={v['left_id']} has no right pair; skipping",
                          file=sys.stderr)
                    continue
                sampled_ids.append(v["right_id"])
            else:
                sampled_ids.append(v["left_id"])
                if v["right_id"] is not None:
                    sampled_ids.append(v["right_id"])
        sampled_ids = sorted(set(sampled_ids))

        print(f"[info] db: {db_path}")
        print(f"[info] inspection {args.inspection}: {len(views)} distinct viewpoint(s), "
              f"trajectory length {total_len:.3f} m")
        print(f"[info] interval={args.interval_m} m, start_index={args.start_index}, lens={args.lens}")
        print(f"[info] picked {len(picked)} viewpoint(s) -> {len(sampled_ids)} image id(s):")
        print("  " + ", ".join(str(i) for i in sampled_ids))
        gaps = [picked[i + 1]["arclength"] - picked[i]["arclength"]
                for i in range(len(picked) - 1)]
        if gaps:
            print(f"[info] sample gaps (m): min={min(gaps):.3f} "
                  f"med={sorted(gaps)[len(gaps)//2]:.3f} max={max(gaps):.3f}")

        if args.json_path:
            payload = [
                {
                    "id": v["left_id"],
                    "left_id": v["left_id"],
                    "right_id": v["right_id"],
                    "arclength_m": round(v["arclength"], 4),
                    "tx": v["tx"], "ty": v["ty"], "tz": v["tz"],
                }
                for v in picked
            ]
            Path(args.json_path).write_text(json.dumps(payload, indent=2))
            print(f"[info] wrote {len(payload)} sampled viewpoint(s) to {args.json_path}")

        if args.no_copy:
            print("[info] --no-copy: skipping file copy")
            return 0

        image_dir = Path(args.image_dir)
        out_dir = Path(args.out_dir)
        if not image_dir.exists():
            print(f"[error] image dir not found: {image_dir}", file=sys.stderr)
            return 2
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.keep_existing:
            for f in out_dir.iterdir():
                if f.is_file():
                    f.unlink()

        copied = 0
        missing: list[int] = []
        for sid in sampled_ids:
            src = image_dir / f"{sid}.jpg"
            dst = out_dir / f"{sid}.jpg"
            if not src.exists():
                missing.append(sid)
                continue
            shutil.copy2(src, dst)
            copied += 1
        if missing:
            print(f"[warn] {len(missing)} sampled id(s) missing in {image_dir}: {missing}",
                  file=sys.stderr)
        print(f"[info] copied {copied}/{len(sampled_ids)} image(s) to {out_dir}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())