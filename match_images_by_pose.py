#!/usr/bin/env python3
"""Pose-based image matching across two inspection runs.

For each image in a *source* set, find the image in a *target* inspection that
was taken from the same viewpoint by comparing the camera pose stored on the
``images`` table (``cam_tf_translation_{x,y,z}`` + quaternion
``cam_tf_rotation_{x,y,z,w}``). Both inspections live in the shared
``camera_init`` (FastLIO) global frame, so poses are directly comparable - no
cross-run alignment is needed.

Matching is **lens-agnostic**: every image is treated as a unique capture and
compared against every candidate using the per-camera world pose
(``cam_tf_*``). There is no L/R labeling, no calibration, and no frustum
gate - just the pose gate and the cost function.

Source set resolution (in order):

1. ``--source-inspection``: every pose-bearing image of one inspection.
2. By default, the ``gt_image`` ids of ``abnormal_detections`` rows whose
   ``inspection_image`` is still NULL (gt images not yet matched).
3. ``--sampled-dir``: ``<id>.jpg`` filenames (use ``--no-gt-db`` to force this).
4. ``--sample-interval-m``: sample the source inspection trajectory at a fixed
   arc-length interval via ``sample_images_along_trajectory.py``.

**Image folders are optional** - with only ``--db`` (which holds all the poses)
the matcher still produces the pairs JSON and trajectory plot; image outputs
(``--matched-dir``, ``--copy-split-dir``) are skipped when ``--sampled-dir`` /
``--image-dir`` are not provided.

Matching direction (``--match-direction``, default ``target``): each target
image independently picks its lowest-cost feasible source, so the same source
may be matched by multiple targets. With ``source`` the direction is reversed.

Candidates pass a pose gate (``--max-dist-m`` / ``--max-rot-deg``). Cost::

    cost = trans + rot_weight * rot

Examples::

    # DB-only: just write the pairs JSON and trajectory plot
    python backend/scripts/match_images_by_pose.py --db inspection.db

    # Sample source images at 1 m, match against inspection 3 with a tight
    # 0.5 m / 10 deg gate, save all outputs under inspection_database/
    python backend/scripts/match_images_by_pose.py \
        --db "inspection_database/inspection_v2.db" \
        --image-dir "inspection_database/outputs/images" \
        --sampled-dir "inspection_database/sampled_images" \
        --target-inspection 3 --sample-inspection 1 \
        --sample-interval-m 1 --max-dist-m 0.5 --max-rot-deg 10 \
        --plot --plot-out "inspection_database/trajectory_insp1_vs_3.png" \
        --matched-dir "inspection_database/matched_pairs_tight" --no-show \
        --copy-split-dir "inspection_database/split_pairs" \
        --json "inspection_database/pairs_insp3_tight.json" \
        --config-out "inspection_database/config.json"

Add ``--commit`` to store the matched pairs in the ``abnormal_detections``
table (the ``gt_image`` column is pre-populated by
``sample_images_along_trajectory.py --commit-gt``; this matcher only fills the
``inspection_image`` column for each source gt image).
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

import numpy as np

import sample_images_along_trajectory as sampler

# Repo root is one level up from this script (image_selection/ -> repo root),
# so defaults resolve correctly regardless of the current working directory.
INSPECTION_DIRECTORY = "inspection_database"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DB = _REPO_ROOT / INSPECTION_DIRECTORY / "complete2" / "inspection_v2.db"


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_rotation_deg(
    q_src: np.ndarray, q_tgt: np.ndarray
) -> np.ndarray:
    """Geodesic rotation angle (degrees) between quaternions, broadcastable.

    ``q_src`` is (N, 4), ``q_tgt`` is (M, 4); returns (N, M). ``abs(dot)``
    folds the quaternion double-cover (q and -q are the same rotation).
    """
    dot = np.abs(q_src @ q_tgt.T)
    dot = np.clip(dot, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def _load_inspection(
    conn: sqlite3.Connection, inspection_id: int
) -> list[dict[str, Any]]:
    """All pose-bearing images of one inspection, ordered by id.

    Each image is treated as a unique capture. The pose used for matching is
    the **camera world pose** stored in the ``cam_tf_*`` columns (per-frame
    translation + quaternion of the camera in the world frame), aliased here
    as ``tx/ty/tz/rx/ry/rz/rw``.
    """
    raw = conn.execute(
        """
        SELECT id, inspection_id, filename, timestamp_ns,
               cam_tf_translation_x AS tx, cam_tf_translation_y AS ty, cam_tf_translation_z AS tz,
               cam_tf_rotation_x    AS rx, cam_tf_rotation_y    AS ry,
               cam_tf_rotation_z    AS rz, cam_tf_rotation_w    AS rw
          FROM images
          WHERE inspection_id = ? AND cam_tf_translation_x IS NOT NULL
                                     AND cam_tf_rotation_w IS NOT NULL
          ORDER BY id
          """,
        (inspection_id,),
    ).fetchall()
    return [dict(r) for r in raw]


def _match_all(
    sources: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    max_dist_m: float,
    max_rot_deg: float,
    rot_weight: float,
    per_target: bool = False,
) -> list[dict[str, Any]]:
    """Per-query nearest feasible candidate (the other side may be reused).

    With ``per_target=False`` each source independently picks its lowest-cost
    feasible target; the same target can therefore be matched by multiple
    sources (one-to-many on the target side). With ``per_target=True`` the
    direction is reversed: each target independently picks its lowest-cost
    feasible source, so the same source can be matched by multiple targets
    (one-to-many on the source side). Either way each query image gets at most
    one counterpart.

    Cost is ``trans + rot_weight * rot``.

    Returns one dict per kept pair (``sampled_id`` = source/gt id,
    ``target_id`` = target id, regardless of direction). Query images with no
    acceptable candidate are omitted; the caller reports them.
    """
    if not sources or not candidates:
        return []

    if per_target:
        queries, pool = candidates, sources
    else:
        queries, pool = sources, candidates

    P = np.array([[q["tx"], q["ty"], q["tz"]] for q in queries], dtype=float)
    C = np.array([[c["tx"], c["ty"], c["tz"]] for c in pool], dtype=float)
    qP = np.array([[q["rx"], q["ry"], q["rz"], q["rw"]] for q in queries], dtype=float)
    qC = np.array([[c["rx"], c["ry"], c["rz"], c["rw"]] for c in pool], dtype=float)

    trans = np.sqrt(((P[:, None, :] - C[None, :, :]) ** 2).sum(-1))  # (Q, M)
    rot = _quat_rotation_deg(qP, qC)                                  # (Q, M)

    cost = trans + rot_weight * rot

    # Pose gate: rotation near 0 and translation within max_dist_m.
    feasible = (trans <= max_dist_m) & (rot <= max_rot_deg)

    pairs: list[dict[str, Any]] = []
    for i in range(len(queries)):
        feas_idx = np.flatnonzero(feasible[i])
        if feas_idx.size == 0:
            continue
        q = queries[i]
        best: dict[str, Any] | None = None
        for j in feas_idx:
            c = pool[int(j)]
            if per_target:
                s, t = c, q  # pool member is the source, query is the target
            else:
                s, t = q, c
            cj = float(cost[i, j])
            if best is None or cj < best["cost"]:
                best = {
                    "sampled_id": int(s["id"]),
                    "target_id": int(t["id"]),
                    "translation_m": float(trans[i, j]),
                    "rotation_deg": float(rot[i, j]),
                    "cost": cj,
                }
        if best is not None:
            pairs.append(best)
    return pairs


def _nearest_distance(
    src: dict[str, Any], candidates: list[dict[str, Any]],
) -> tuple[float, float, int | None]:
    """Nearest candidate's (translation_m, rotation_deg, id) for diagnostics."""
    if not candidates:
        return float("inf"), float("inf"), None
    P = np.array([[src["tx"], src["ty"], src["tz"]]], dtype=float)
    C = np.array([[c["tx"], c["ty"], c["tz"]] for c in candidates], dtype=float)
    qP = np.array([[src["rx"], src["ry"], src["rz"], src["rw"]]], dtype=float)
    qC = np.array([[c["rx"], c["ry"], c["rz"], c["rw"]] for c in candidates], dtype=float)
    trans = np.sqrt(((P[:, None, :] - C[None, :, :]) ** 2).sum(-1))[0]
    rot = _quat_rotation_deg(qP, qC)[0]
    j = int(np.argmin(trans))
    return float(trans[j]), float(rot[j]), int(candidates[j]["id"])


def _load_sampled_ids(sampled_dir: Path) -> list[int]:
    """Image ids from ``<id>.jpg`` filenames in ``sampled_dir``."""
    ids: list[int] = []
    for p in sorted(sampled_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            stem = p.stem
            if stem.isdigit():
                ids.append(int(stem))
    return sorted(ids)


def _resolve_source_inspection(conn: sqlite3.Connection, sampled_ids: list[int]) -> int | None:
    """Inspection id that the sampled images belong to (all should match)."""
    if not sampled_ids:
        return None
    placeholders = ",".join("?" * len(sampled_ids))
    row = conn.execute(
        f"SELECT inspection_id, COUNT(*) AS n FROM images "
        f"WHERE id IN ({placeholders}) GROUP BY inspection_id ORDER BY n DESC LIMIT 1",
        sampled_ids,
    ).fetchone()
    return int(row["inspection_id"]) if row else None


def _commit_pairs(
    conn: sqlite3.Connection,
    pairs: list[dict[str, Any]],
    source_ids: list[int],
    skip_existing: bool,
    delete_unmatched: bool = False,
) -> int:
    """Store only ``inspection_image`` on existing ``abnormal_detections`` rows.

    The ``gt_image`` column is populated up front by
    ``sample_images_along_trajectory.py --commit-gt``; this matcher only fills
    in the matched ``inspection_image`` (the target inspection id) for each
    source gt id. It never inserts new rows and never touches the ``gt_image``
    column.

    Rows whose ``gt_image`` belongs to this run's source set but found no
    acceptable target (i.e. the source id is not in any matched pair) are left
    with a NULL ``inspection_image`` so a later run can retry them. With
    ``delete_unmatched`` they are instead **deleted** from
    ``abnormal_detections`` so the table only carries pairs that actually
    matched.

    With ``skip_existing`` (default), matched pairs whose row already has a
    non-NULL ``inspection_image`` are left untouched so previously matched /
    LLM-annotated rows are never clobbered. Returns the number of rows whose
    ``inspection_image`` was set by this call.
    """
    if not source_ids:
        return 0

    matched: dict[int, int] = {p["sampled_id"]: p["target_id"] for p in pairs}
    unmatched = [sid for sid in source_ids if sid not in matched]

    updated = 0
    for gt_id, tgt_id in matched.items():
        if skip_existing:
            row = conn.execute(
                "SELECT 1 FROM abnormal_detections "
                "WHERE gt_image = ? AND inspection_image IS NOT NULL LIMIT 1",
                (gt_id,),
            ).fetchone()
            if row:
                continue
        cur = conn.execute(
            "UPDATE abnormal_detections SET inspection_image = ? "
            "WHERE gt_image = ?",
            (tgt_id, gt_id),
        )
        updated += cur.rowcount

    if unmatched and delete_unmatched:
        placeholders = ",".join("?" * len(unmatched))
        cur = conn.execute(
            f"DELETE FROM abnormal_detections "
            f"WHERE gt_image IN ({placeholders})",
            unmatched,
        )
        print(f"[info] removed {cur.rowcount} unmatched gt row(s) "
              f"(gt_image in {unmatched})", file=sys.stderr)

    conn.commit()
    return updated


def _load_timestamps(conn: sqlite3.Connection) -> dict[int, int]:
    """Lookup image id -> timestamp_ns from the database."""
    return {row["id"]: row["timestamp_ns"] for row in conn.execute(
        "SELECT id, timestamp_ns FROM images WHERE timestamp_ns IS NOT NULL"
    )}


def _copy_matched_images(
    pairs: list[dict[str, Any]],
    sampled_dir: Path,
    image_dir: Path,
    out_dir: Path,
    show: bool = True,
) -> list[Path]:
    """Write each matched pair as a single merged image (source left, target
    right) into ``out_dir``, named ``<src_ts>__<tgt_ts>.jpg`` where ``_ts`` is
    the ``timestamp_ns`` of each image.

    Optionally spawn a cv2 window showing each merged pair. Returns the list of
    merged files written. Requires opencv-python (``cv2``).
    """
    if not pairs:
        return []
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        print(f"[error] --matched-dir needs opencv (cv2): missing: {e}", file=sys.stderr)
        raise

    out_dir.mkdir(parents=True, exist_ok=True)
    merged: list[Path] = []
    for pp in pairs:
        src_id, tgt_id = pp["sampled_id"], pp["target_id"]
        src_ts = pp.get("sampled_timestamp", src_id)
        tgt_ts = pp.get("target_timestamp", tgt_id)
        src = sampled_dir / f"{src_id}.jpg"
        tgt = image_dir / f"{tgt_id}.jpg"
        if not src.exists() or not tgt.exists():
            print(f"[warn] matched pair {src_id}->{tgt_id}: missing image "
                  f"({src} / {tgt}), skipped", file=sys.stderr)
            continue
        srcimg = cv2.imread(str(src))
        tgtimg = cv2.imread(str(tgt))
        if srcimg is None or tgtimg is None:
            print(f"[warn] matched pair {src_id}->{tgt_id}: could not decode "
                  f"image, skipped", file=sys.stderr)
            continue
        # Place both on a common canvas, side by side (src left, tgt right).
        height = max(srcimg.shape[0], tgtimg.shape[0])
        width = srcimg.shape[1] + tgtimg.shape[1]
        canvas = np.full((height, width, 3), 128, dtype=np.uint8)
        canvas[:srcimg.shape[0], :srcimg.shape[1]] = srcimg
        canvas[:tgtimg.shape[0], srcimg.shape[1]:] = tgtimg
        merged_path = out_dir / f"{src_ts}__{tgt_ts}.jpg"
        if not cv2.imwrite(str(merged_path), canvas):
            print(f"[warn] could not write merged image: {merged_path}", file=sys.stderr)
            continue
        merged.append(merged_path)

    if show and merged:
        print("[info] opening cv2 window of matched pairs (press any key to advance, "
              "ESC to quit)")
        window = "matched pairs (src | tgt)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 800, 600)
        for mp in merged:
            img = cv2.imread(str(mp))
            if img is None:
                continue
            cv2.imshow(window, img)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break
        cv2.destroyAllWindows()
    return merged


def _copy_split_images(
    pairs: list[dict[str, Any]],
    sampled_dir: Path,
    image_dir: Path,
    out_dir: Path,
) -> tuple[int, int]:
    """Copy matched source and target images into two separate subfolders.

    Creates ``out_dir/source/`` and ``out_dir/target/`` and copies each matched
    pair's source image (from ``sampled_dir``) and target image (from
    ``image_dir``) there as ``<timestamp_ns>.jpg`` (using the image's
    ``timestamp_ns`` from the database). Existing files in the two subfolders
    are cleared first. Returns ``(n_source, n_target)`` copied counts.
    """
    src_out = out_dir / "source"
    tgt_out = out_dir / "target"
    src_out.mkdir(parents=True, exist_ok=True)
    tgt_out.mkdir(parents=True, exist_ok=True)
    for d in (src_out, tgt_out):
        for f in d.iterdir():
            if f.is_file():
                f.unlink()

    n_src = n_tgt = 0
    missing: list[int] = []
    for pp in pairs:
        sid, tid = pp["sampled_id"], pp["target_id"]
        src_ts = pp.get("sampled_timestamp", sid)
        tgt_ts = pp.get("target_timestamp", tid)
        src_name = f"{src_ts}.jpg"
        tgt_name = f"{tgt_ts}.jpg"
        s = sampled_dir / f"{sid}.jpg"
        t = image_dir / f"{tid}.jpg"
        if s.exists():
            shutil.copy2(s, src_out / src_name)
            n_src += 1
        else:
            missing.append(sid)
        if t.exists():
            shutil.copy2(t, tgt_out / tgt_name)
            n_tgt += 1
        else:
            missing.append(tid)
    if missing:
        print(f"[warn] {len(missing)} image(s) missing when copying to split dirs: "
              f"{missing}", file=sys.stderr)
    return n_src, n_tgt


def _load_unmatched_gt_ids(conn: sqlite3.Connection) -> list[int]:
    """``gt_image`` ids of ``abnormal_detections`` rows not yet matched.

    Returns an empty list when the table does not exist (e.g. the sampler has
    never committed gt images to this database).
    """
    try:
        rows = conn.execute(
            "SELECT gt_image FROM abnormal_detections "
            "WHERE gt_image IS NOT NULL AND inspection_image IS NULL "
            "ORDER BY gt_image"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [int(r["gt_image"]) for r in rows]


def _resample_images(
    conn: sqlite3.Connection,
    out_dir: Path,
    src_dir: Path,
    inspection: int,
    interval_m: float,
    start_index: int,
    keep_existing: bool,
) -> int:
    """Populate ``out_dir`` with images sampled along ``inspection``'s trajectory.

    Loads every pose-bearing image of the inspection (each treated as a
    unique viewpoint), runs the greedy arc-length sampler from
    ``sample_images_along_trajectory``, and copies the picked ``<id>.jpg``
    files from ``src_dir`` into ``out_dir`` (cleared first unless
    ``keep_existing``). Returns the number of images copied, or -1 on error.
    """
    views = _load_inspection(conn, inspection)
    if not views:
        print(f"[error] no pose-bearing viewpoints in inspection {inspection}", file=sys.stderr)
        return -1

    picked = sampler._sample_by_arclength(views, interval_m, start_index)
    sampled_ids = sorted({int(v["id"]) for v in picked})

    out_dir.mkdir(parents=True, exist_ok=True)
    if not keep_existing:
        for f in out_dir.iterdir():
            if f.is_file():
                f.unlink()

    copied = 0
    missing: list[int] = []
    for sid in sampled_ids:
        src = src_dir / f"{sid}.jpg"
        dst = out_dir / f"{sid}.jpg"
        if not src.exists():
            missing.append(sid)
            continue
        shutil.copy2(src, dst)
        copied += 1
    if missing:
        print(f"[warn] {len(missing)} sampled id(s) missing in {src_dir}: {missing}",
              file=sys.stderr)
    return copied


def _plot_trajectories(
    conn: sqlite3.Connection,
    source_inspection: int,
    target_inspection: int,
    pairs: list[dict[str, Any]] | None = None,
    out_path: str | None = None,
) -> None:
    """Plot source vs target inspection trajectories (cam_tf x/z) with matplotlib.

    Each inspection's images are projected to the ground plane using their
    ``cam_tf_translation_x`` / ``cam_tf_translation_z`` and drawn as a light gray
    trajectory line. When ``pairs`` is given, each matched source-target pair
    is drawn as two points sharing a single colour from a red-to-blue colormap
    (``RdBu``), with a thin dashed line connecting them, so matched
    correspondences are visually traceable. Unmatched points are omitted for
    clarity. If ``out_path`` is given the figure is saved there instead of shown
    interactively.
    """
    import matplotlib.pyplot as plt

    def _pos(insp: int) -> tuple[list[float], list[float]]:
        xs, zs = [], []
        for row in conn.execute(
            "SELECT cam_tf_translation_x AS tx, cam_tf_translation_z AS tz "
            "FROM images WHERE inspection_id = ? "
            "AND cam_tf_translation_x IS NOT NULL ORDER BY id",
            (insp,),
        ):
            xs.append(row["tx"])
            zs.append(row["tz"])
        return xs, zs

    plt.figure(figsize=(12, 7))

    # Draw both full trajectories as faint gray lines for context.
    for insp in (source_inspection, target_inspection):
        xs, zs = _pos(insp)
        if xs:
            plt.plot(xs, zs, "-", color="lightgray", markersize=0, linewidth=0.8,
                     zorder=1)

    # Draw matched pairs: each pair gets a unique colour from RdBu (red->blue).
    if pairs:
        # Build id -> (tx, tz) lookup for both inspections.
        src_pos = {}
        tgt_pos = {}
        for row in conn.execute(
            "SELECT id, inspection_id, cam_tf_translation_x AS tx, cam_tf_translation_z AS tz "
            "FROM images WHERE cam_tf_translation_x IS NOT NULL ORDER BY id",
        ):
            if row["inspection_id"] == source_inspection:
                src_pos[row["id"]] = (row["tx"], row["tz"])
            elif row["inspection_id"] == target_inspection:
                tgt_pos[row["id"]] = (row["tx"], row["tz"])

        n = len(pairs)
        cmap = plt.cm.RdBu
        colors = [cmap(i / max(1, n - 1)) for i in range(n)]
        for idx, pp in enumerate(pairs):
            sp = src_pos.get(pp["sampled_id"])
            tp = tgt_pos.get(pp["target_id"])
            if sp is None or tp is None:
                continue
            c = colors[idx]
            # Connect matched pair with a thin dashed line.
            plt.plot([sp[0], tp[0]], [sp[1], tp[1]], "--",
                     color=c, linewidth=0.6, alpha=0.5, zorder=2)
            plt.plot(sp[0], sp[1], "o", color=c, markersize=8, zorder=3)
            plt.plot(tp[0], tp[1], "s", color=c, markersize=8, zorder=3)
            # Annotate every 10th source point with its image id to avoid clutter.
            if idx % 10 == 0:
                plt.annotate(
                    str(pp["sampled_id"]),
                    xy=sp, xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=c, zorder=4,
                )

        plt.plot([], [], "o", color="gray", markersize=8, label="source (circle)")
        plt.plot([], [], "s", color="gray", markersize=8, label="target (square)")
        plt.plot([], [], "--", color="gray", linewidth=0.6, label="matched link")
        plt.title(f"Matched pairs: inspection {source_inspection} -> {target_inspection} "
                  f"({n} pairs, red->blue, ids=source)")
    else:
        for insp, color, label in (
            (source_inspection, "tab:blue", f"Inspection {source_inspection} (source)"),
            (target_inspection, "tab:orange", f"Inspection {target_inspection} (target)"),
        ):
            xs, zs = _pos(insp)
            if xs:
                plt.plot(xs, zs, "-o", color=color, markersize=2, linewidth=1, label=label)
        plt.title("Inspection trajectories (camera_init frame)")

    plt.xlabel("cam_tf x (m)")
    plt.ylabel("cam_tf z (m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"[info] saved trajectory plot to {out_path}")
    else:
        plt.show()


def _fmt_num(x: float) -> str:
    """Format a float for use in a folder name: 0.75 -> '0p75', 1.0 -> '1', 15.0 -> '15'."""
    s = f"{x:g}"
    return s.replace(".", "p")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(_DEFAULT_DB), help="Path to the inspection .db")
    p.add_argument(
        "--sampled-dir",
        default=None,
        help="Directory of <id>.jpg source images from inspection 1. Optional: "
             "with only --db, the matcher uses --source-inspection or the "
             "abnormal_detections gt_image rows; image outputs needing this "
             "folder are skipped.",
    )
    p.add_argument(
        "--source-inspection",
        type=int,
        default=None,
        help="Override: use ALL images of this inspection as the source instead of --sampled-dir",
    )
    p.add_argument(
        "--sample-interval-m",
        type=float,
        default=None,
        help="Re-populate --sampled-dir by sampling the source inspection trajectory "
             "at this fixed arc-length interval (metres) via sample_images_along_trajectory.py",
    )
    p.add_argument(
        "--sample-inspection",
        type=int,
        default=1,
        help="Inspection id to sample from when --sample-interval-m is set (default 1)",
    )
    p.add_argument(
        "--sample-start-index",
        type=int,
        default=0,
        help="Index (in capture order) of the first sampled viewpoint (default 0)",
    )
    p.add_argument(
        "--keep-sampled",
        action="store_true",
        help="When --sample-interval-m is set, do not clear --sampled-dir before writing",
    )
    p.add_argument("--target-inspection", type=int, default=2, help="Inspection id to match against (default 2)")
    p.add_argument(
        "--no-gt-db",
        dest="gt_db",
        action="store_false",
        help="Do NOT take the source set from unmatched gt_image rows of the "
             "abnormal_detections table; use --sampled-dir instead. By default "
             "the table is used whenever it holds unmatched gt rows.",
    )
    p.set_defaults(gt_db=True)
    p.add_argument("--max-dist-m", type=float, default=1.5, help="Max translation (m) for a valid pair")
    p.add_argument(
        "--match-direction",
        choices=["source", "target"],
        default="target",
        help="Which side independently picks its nearest feasible counterpart "
             "(the other side may be reused, i.e. one-to-many on the other "
             "side). 'target' (default): each target image picks its best "
             "source. 'source': each source image picks its best target.",
    )
    p.add_argument("--max-rot-deg", type=float, default=12.0,
                   help="Max rotation (deg) for a matched pair (near 0 deg)")
    p.add_argument("--rot-weight", type=float, default=0.15, help="m-per-deg rotation weight in the cost")
    p.add_argument("--json", dest="json_path", default=None, help="Write the pair list to this JSON file")
    p.add_argument(
        "--image-dir",
        default=None,
        help="Image folder for reported target file paths and image outputs "
             "(--matched-dir, --copy-split-dir). Optional when "
             "only the pairs JSON / trajectory plot are needed.",
    )
    p.add_argument(
        "--matched-dir",
        default=None,
        help="Write each matched pair as a merged image (src left, tgt right) "
             "into this folder as <src_id>__<tgt_id>.jpg",
    )
    p.add_argument(
        "--copy-split-dir",
        default=None,
        help="Copy the matched source and target images into two separate "
             "subfolders (``source/`` and ``target/``) inside this directory, "
             "each named <id>.jpg. Useful for feeding downstream tools that "
             "expect flat per-side folders.",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Name of the run folder under inspection_database/runs/. If omitted, "
             "auto-generated as '<src>vs<tgt>_<dist>m_<rot>deg' (e.g. '1vs2_0p75m_15deg'). "
             "All outputs are auto-placed inside with clean names: insp<src>vs<tgt>/ "
             "(merged images), split_pairs/ (source+target subfolders), pairs.json, "
             "config.json, trajectory.png. Overrides --json/--matched-dir/--copy-split-dir/"
             "--plot-out/--config-out when set.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="With --matched-dir, do not open the cv2 display window",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Plot the source and target inspection trajectories with matplotlib",
    )
    p.add_argument(
        "--plot-out",
        default=None,
        help="Save the trajectory plot to this file (default: show interactively)",
    )
    p.add_argument(
        "--commit",
        action="store_true",
        help="Fill inspection_image on abnormal_detections for matched pairs "
             "(gt_image is assumed to be pre-populated by "
             "sample_images_along_trajectory.py --commit-gt). Unmatched gt rows "
             "are kept with NULL inspection_image unless --delete-unmatched is set.",
    )
    p.add_argument(
        "--delete-unmatched",
        action="store_true",
        help="With --commit, DELETE abnormal_detections rows whose gt_image "
             "found no acceptable target in this run instead of keeping them "
             "for a later retry.",
    )
    p.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="With --commit, also overwrite inspection_image on rows already matched",
    )
    p.add_argument(
        "--config-out",
        default=None,
        help="Write a config.json with the parameters used to run this matching "
             "session (db, inspections, gates, sampling, output paths, etc.)",
    )
    p.set_defaults(skip_existing=True)
    args = p.parse_args(argv)

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"[error] database not found: {db_path}", file=sys.stderr)
        return 2
    sampled_dir: Path | None = (
        Path(args.sampled_dir).resolve() if args.sampled_dir else None
    )
    image_dir: Path | None = (
        Path(args.image_dir).resolve() if args.image_dir else None
    )

    # If --run-name is set, create a run folder and auto-route all outputs
    # inside it with clean names.
    run_dir: Path | None = None
    if args.run_name:
        run_name = args.run_name
    else:
        src_insp_for_name = (args.source_inspection if args.source_inspection is not None
                             else args.sample_inspection)
        run_name = f"{src_insp_for_name}vs{args.target_inspection}"
        if args.sample_interval_m is not None:
            run_name += f"_{_fmt_num(args.sample_interval_m)}m"
        run_name += f"_{_fmt_num(args.max_dist_m)}m_{_fmt_num(args.max_rot_deg)}deg"
    run_dir = _REPO_ROOT / INSPECTION_DIRECTORY / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    src_insp_for_name = (args.source_inspection if args.source_inspection is not None
                         else args.sample_inspection)
    label = f"insp{src_insp_for_name}vs{args.target_inspection}"
    # Override output paths with clean names inside the run folder. Image-based
    # outputs are only enabled when their required image folders are present.
    args.json_path = str(run_dir / "pairs.json")
    args.config_out = str(run_dir / "config.json")
    args.plot_out = str(run_dir / "trajectory.png")
    if sampled_dir is not None:
        args.matched_dir = str(run_dir / label)
        args.copy_split_dir = str(run_dir / "split_pairs")
    else:
        args.matched_dir = None
        args.copy_split_dir = None
    args.plot = True

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    if args.sample_interval_m is not None:
        if sampled_dir is None or image_dir is None:
            print("[error] --sample-interval-m needs --sampled-dir and --image-dir "
                  "to read/write source images", file=sys.stderr)
            conn.close()
            return 2
        n = _resample_images(
            conn,
            sampled_dir,
            image_dir,
            args.sample_inspection,
            args.sample_interval_m,
            args.sample_start_index,
            args.keep_sampled,
        )
        if n < 0:
            conn.close()
            return 2
        print(f"[info] sampled {n} image(s) from inspection {args.sample_inspection} "
              f"at {args.sample_interval_m} m into {sampled_dir}")
    try:
        target_rows = _load_inspection(conn, args.target_inspection)
        if not target_rows:
            print(f"[error] no pose-bearing images in inspection {args.target_inspection}", file=sys.stderr)
            return 2

        # Resolve the source set.
        if args.source_inspection is not None:
            source_rows = _load_inspection(conn, args.source_inspection)
            src_label = f"all of inspection {args.source_inspection}"
        else:
            source_ids: list[int] = []
            if args.gt_db:
                source_ids = _load_unmatched_gt_ids(conn)
                if source_ids:
                    print(f"[info] source: {len(source_ids)} unmatched gt_image(s) "
                          f"from abnormal_detections (inspection_image IS NULL)")
            if not source_ids:
                if args.gt_db:
                    print("[info] no unmatched gt rows in abnormal_detections; "
                          "falling back to --sampled-dir", file=sys.stderr)
                if sampled_dir is None or not sampled_dir.exists():
                    missing_dir = sampled_dir if sampled_dir is not None else "--sampled-dir"
                    print(f"[error] sampled dir not found: {missing_dir} "
                          f"(needed when no --source-inspection and no unmatched gt rows)",
                          file=sys.stderr)
                    return 2
                source_ids = _load_sampled_ids(sampled_dir)
                if not source_ids:
                    print(f"[error] no <id>.jpg files found in {sampled_dir}", file=sys.stderr)
                    return 2
            src_insp = _resolve_source_inspection(conn, source_ids)
            if src_insp is None:
                print("[error] could not resolve which inspection the sampled images belong to", file=sys.stderr)
                return 2
            # Load every image of the source inspection and keep the sampled ids.
            all_src = _load_inspection(conn, src_insp)
            id_to_row = {r["id"]: r for r in all_src}
            source_rows: list[dict[str, Any]] = []
            missing: list[int] = []
            for sid in source_ids:
                r = id_to_row.get(sid)
                if r is not None:
                    source_rows.append(r)
                else:
                    missing.append(sid)
            if missing:
                print(f"[warn] {len(missing)} sampled id(s) not found in inspection "
                      f"{src_insp} (skipped): {missing}", file=sys.stderr)
            src_label = f"{len(source_rows)} sampled image(s) from inspection {src_insp}"

        print(f"[info] db: {db_path}")
        print(f"[info] source: {src_label}")
        print(f"[info] target: inspection {args.target_inspection} ({len(target_rows)} imgs)")
        print(f"[info] gate: max_dist={args.max_dist_m} m, max_rot={args.max_rot_deg} deg, "
              f"rot_weight={args.rot_weight}")
        if sampled_dir is None:
            print("[info] no --sampled-dir: image outputs (matched/split) "
                  "will be skipped; only pairs JSON / trajectory plot produced")
        if image_dir is None:
            print("[info] no --image-dir: target image outputs (matched) "
                  "will be skipped")

        all_pairs = _match_all(
            source_rows, target_rows,
            args.max_dist_m, args.max_rot_deg,
            args.rot_weight,
            per_target=args.match_direction == "target",
        )
        per_target = args.match_direction == "target"
        if per_target:
            query_rows, pool_rows = target_rows, source_rows
            query_label, pool_label = "target", "source"
        else:
            query_rows, pool_rows = source_rows, target_rows
            query_label, pool_label = "source", "target"
        query_id_key = "target_id" if per_target else "sampled_id"
        matched_query_ids: set[int] = {pp[query_id_key] for pp in all_pairs}
        print(f"[info] matched {len(all_pairs)}/{len(query_rows)} {query_label} image(s) "
              f"against {len(pool_rows)} {pool_label} candidate(s)")

        # Candidate pool for unmatched diagnostics.
        diag_pool = pool_rows

        # Diagnostics for unmatched query images: nearest candidate distance.
        unmatched = [r for r in query_rows if r["id"] not in matched_query_ids]
        if unmatched:
            print(f"\n[info] {len(unmatched)} unmatched {query_label} image(s) - "
                  f"nearest candidate:")
            for r in sorted(unmatched, key=lambda x: x["id"]):
                nt, nr, nid = _nearest_distance(r, diag_pool)
                print(f"  {query_label} id={r['id']:>4}  "
                      f"nearest {pool_label} id={nid}  "
                      f"trans={nt:.3f} m  rot={nr:.2f} deg  "
                      f"(over gate -> no match)")

        all_pairs.sort(key=lambda pp: pp["sampled_id"])

        # Enrich pairs with timestamps from the database (used for output
        # filenames and JSON keys).
        ts_lookup = _load_timestamps(conn)
        for pp in all_pairs:
            pp["sampled_timestamp"] = ts_lookup.get(pp["sampled_id"])
            pp["target_timestamp"] = ts_lookup.get(pp["target_id"])

        if args.plot:
            src_insp = args.sample_inspection if args.sample_interval_m is not None \
                else (args.source_inspection if args.source_inspection is not None
                      else args.sample_inspection)
            _plot_trajectories(conn, src_insp, args.target_inspection,
                               pairs=all_pairs, out_path=args.plot_out)

        print("\n=== matched pairs ===")
        print(f"{'src_id':>7} {'tgt_id':>7} {'trans_m':>9} {'rot_deg':>8} {'cost':>7}")
        for pp in all_pairs:
            print(f"{pp['sampled_id']:>7} {pp['target_id']:>7} "
                  f"{pp['translation_m']:>9.3f} {pp['rotation_deg']:>8.2f} "
                  f"{pp['cost']:>7.3f}")

        if args.json_path:
            # Only matched pairs go into the JSON; unmatched sources are
            # reported in the console/summary but excluded here.
            json_pairs = []
            src_insp_val = (args.source_inspection if args.source_inspection is not None
                            else args.sample_inspection)
            for pp in all_pairs:
                jp = {
                    "source_inspection": src_insp_val,
                    "target_inspection": args.target_inspection,
                    "sampled_id": pp["sampled_id"],
                    "target_id": pp["target_id"],
                    "translation_m": pp["translation_m"],
                    "rotation_deg": pp["rotation_deg"],
                    "cost": pp["cost"],
                    "sampled_timestamp": pp.get("sampled_timestamp"),
                    "target_timestamp": pp.get("target_timestamp"),
                }
                json_pairs.append(jp)
            Path(args.json_path).write_text(json.dumps(json_pairs, indent=2))
            print(f"[info] wrote {len(json_pairs)} matched pair(s) to {args.json_path}")

        if args.matched_dir:
            if sampled_dir is None or image_dir is None:
                print("[warn] --matched-dir needs --sampled-dir and --image-dir; "
                      "skipping merged image output", file=sys.stderr)
            else:
                matched_dir = Path(args.matched_dir).resolve()
                matched_dir.mkdir(parents=True, exist_ok=True)
                merged = _copy_matched_images(
                    all_pairs, sampled_dir, image_dir, matched_dir,
                    show=not args.no_show,
                )
                print(f"[info] wrote {len(merged)} merged pair image(s) to {matched_dir}")
                if merged and args.no_show:
                    print("[info] --no-show set: skipping cv2 display window")

        if args.copy_split_dir:
            if sampled_dir is None or image_dir is None:
                print("[warn] --copy-split-dir needs --sampled-dir and --image-dir; "
                      "skipping split folder output", file=sys.stderr)
            else:
                split_dir = Path(args.copy_split_dir).resolve()
                n_src, n_tgt = _copy_split_images(
                    all_pairs, sampled_dir, image_dir, split_dir,
                )
                print(f"[info] copied {n_src} source / {n_tgt} target image(s) to "
                      f"{split_dir / 'source'} and {split_dir / 'target'}")

        # Results summary: matching stats + names of unmatched images.
        print("\n=== results summary ===")
        print(f"  {query_label} images:     {len(query_rows)}")
        print(f"  {pool_label} images:      {len(pool_rows)}")
        print(f"  matched pairs:      {len(all_pairs)}")
        print(f"  unmatched images:   {len(unmatched)}  (unmatched {query_label} image(s))")
        if all_pairs:
            t = [pp["translation_m"] for pp in all_pairs]
            r = [pp["rotation_deg"] for pp in all_pairs]
            print(f"  translation (m):   min={min(t):.3f} med={sorted(t)[len(t)//2]:.3f} "
                  f"max={max(t):.3f}")
            print(f"  rotation (deg):    min={min(r):.2f} med={sorted(r)[len(r)//2]:.2f} "
                  f"max={max(r):.2f}")
        if unmatched:
            print(f"\n  unmatched {query_label} image(s) (name -> id | nearest cost):")
            for r in sorted(unmatched, key=lambda x: x["id"]):
                nt, nr, nid = _nearest_distance(r, diag_pool)
                cost = nt + args.rot_weight * nr
                name = r["filename"] or f"{r['id']}.jpg"
                print(f"    {name}  (id={r['id']}, "
                      f"nearest {pool_label}={nid}, trans={nt:.3f} m, "
                      f"rot={nr:.2f} deg, cost={cost:.3f})")

        if args.commit:
            if per_target:
                print("[warn] --commit with --match-direction target: a source gt image "
                      "may match several targets, but abnormal_detections stores only one "
                      "inspection_image per gt_image row (one pair per gt is written)",
                      file=sys.stderr)
            all_source_ids = [r["id"] for r in source_rows]
            n = _commit_pairs(conn, all_pairs, all_source_ids, args.skip_existing,
                              delete_unmatched=args.delete_unmatched)
            n_unmatched = len(all_source_ids) - len(all_pairs)
            print(f"[info] set inspection_image on {n} abnormal_detections row(s); "
                  f"{n_unmatched} unmatched source id(s) "
                  f"{'deleted' if args.delete_unmatched else 'kept (NULL inspection_image)'}")

        if args.config_out:
            config = {
                "db": str(db_path),
                "image_dir": str(image_dir) if image_dir else None,
                "sampled_dir": str(sampled_dir) if sampled_dir else None,
                "source_inspection": args.source_inspection,
                "target_inspection": args.target_inspection,
                "sample_inspection": args.sample_inspection,
                "sample_interval_m": args.sample_interval_m,
                "sample_start_index": args.sample_start_index,
                "keep_sampled": args.keep_sampled,
                "max_dist_m": args.max_dist_m,
                "max_rot_deg": args.max_rot_deg,
                "rot_weight": args.rot_weight,
                "match_direction": args.match_direction,
                "gt_db_source": args.gt_db,
                "delete_unmatched": args.delete_unmatched,
                "json_path": args.json_path,
                "matched_dir": args.matched_dir,
                "copy_split_dir": args.copy_split_dir,
                "plot": args.plot,
                "plot_out": args.plot_out,
                "commit": args.commit,
                "skip_existing": args.skip_existing,
                "run_name": args.run_name,
                "run_dir": str(run_dir) if run_dir else None,
                "results": {
                    "source_images": len(source_rows),
                    "matched_pairs": len(all_pairs),
                    "unmatched_images": len(unmatched),
                },
            }
            Path(args.config_out).write_text(json.dumps(config, indent=2))
            print(f"[info] wrote config to {args.config_out}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())