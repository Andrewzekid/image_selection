#!/usr/bin/env python3
"""Pose-based image matching across two inspection runs.

For each image in a *source* set (by default the files in
``MTR Inspection Database/sampled_images/``, which are a subset of inspection 1,
named ``<image_id>.jpg``), find the image in a *target* inspection (default 2)
that was taken from the same viewpoint by comparing the camera pose stored on the
``images`` table (``cam_tf_translation_{x,y,z}`` + quaternion
``cam_tf_rotation_{x,y,z,w}``).

Both inspections live in the shared ``camera_init`` (FastLIO) global frame and
traverse the same route, so poses are directly comparable - no cross-run
alignment is needed.

By default the source set is read from the database: the ``gt_image`` ids of
``abnormal_detections`` rows whose ``inspection_image`` is still NULL (i.e. gt
images not yet matched). Use ``--no-gt-db`` to fall back to ``--sampled-dir``.

Matching is per-source nearest feasible candidate: each source independently
picks its lowest-cost feasible target, so the same target may be matched by
multiple sources (one-to-many on the target side). Candidates must pass a pose
gate (``max_dist_m`` / ``max_rot_deg``) and a **frustum-overlap gate**: using
the camera intrinsics from ``--calibration`` (default
``info/calibration.json``) and the relative pose, a planar homography at
``--plane-depth-m`` maps each image into the other and the symmetric overlap
ratio must be >= ``--min-overlap``. Cost is

    cost = translation_m + rot_weight * rotation_deg
           + overlap_weight * (1 - overlap)

With ``--aligned-dir`` each matched pair is fisheye-undistorted, the target is
warped into the source frame via the inverse homography, and both are cropped
to their common visible area - aligned pairs ready for VLM comparison.

Run with the backend venv so numpy is available, e.g.::

    python backend/scripts/match_images_by_pose.py \
        --target-inspection 2 --json pairs.json

To sample the source images on the fly at a fixed distance interval (via
``sample_images_along_trajectory.py``) before matching instead of relying on an
existing ``--sampled-dir``, add ``--sample-interval-m`` (and the inspection to
sample from with ``--sample-inspection``)

Match L+R pairs from inspection 1 against target inspection 3 at 1 m
sampling with a tight 0.5 m translation / 10 deg rotation gate (candidate images outside of this limit are discarded), using the
database and images inside ``inspection_database/`` and saving the
matched-pairs JSON, the merged side-by-side matched images, the split
source/target folders, and the trajectory plot all under
``inspection_database/``::

    python backend/scripts/match_images_by_pose.py \
        --db "inspection_database/inspection_v2.db" \
        --image-dir "inspection_database/outputs/images" \
        --sampled-dir "inspection_database/sampled_images" \
        --target-inspection 3 --sample-inspection 1 \
        --sample-interval-m 1 --max-dist-m 0.5 --max-rot-deg 10 \
        --select-pair \
        --plot --plot-out "inspection_database/trajectory_insp1_vs_3.png" \
        --matched-dir "inspection_database/matched_pairs_tight" --no-show \
        --copy-split-dir "inspection_database/split_pairs" \
        --json "inspection_database/pairs_insp3_tight.json" \
        --config-out "inspection_database/config.json"

Need to add --commit to store the matched pairs in the ``abnormal_detections`` table of the database (the ``gt_image`` column is pre-populated by ``sample_images_along_trajectory.py --commit-gt``; this matcher only fills in the ``inspection_image`` column for each source gt image).
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
_DEFAULT_SAMPLED_DIR = _REPO_ROOT / INSPECTION_DIRECTORY / "sampled_images"
_DEFAULT_IMAGE_DIR = _REPO_ROOT / INSPECTION_DIRECTORY / "complete2" / "outputs" / "images"
_DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "info" / "calibration.json"


# ---------------------------------------------------------------------------
# Camera model / geometry helpers
# ---------------------------------------------------------------------------

def _load_calibration(path: Path) -> dict[str, dict[str, Any]]:
    """Load per-lens intrinsics from ``info/calibration.json``.

    Returns ``{"L": {...}, "R": {...}}`` where each entry holds the 3x3
    pinhole intrinsic ``K`` (numpy), the OpenCV fisheye distortion coeffs
    ``dist`` (numpy, 4), and the calibrated image size ``(width, height)``.
    Lens parity "L"/"R" maps to the calibration entries named "left"/"right".
    """
    data = json.loads(Path(path).read_text())
    out: dict[str, dict[str, Any]] = {}
    for cam in data["cameras"]:
        parity = "L" if cam["name"] == "left" else "R"
        intr = cam["intrinsic"]
        K = np.array(
            [[intr["fl_x"], 0.0, intr["cx"]],
             [0.0, intr["fl_y"], intr["cy"]],
             [0.0, 0.0, 1.0]],
            dtype=float,
        )
        dp = cam["distortion"]["params"]
        dist = np.array([dp["k1"], dp["k2"], dp["k3"], dp["k4"]], dtype=float)
        out[parity] = {
            "K": K,
            "dist": dist,
            "width": int(cam["width"]),
            "height": int(cam["height"]),
        }
    if "L" not in out or "R" not in out:
        raise ValueError(f"calibration {path} must define 'left' and 'right' cameras")
    return out


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


def _pose_rt(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) camera-to-world from a row's ``tx/ty/tz/rx/ry/rz/rw`` fields."""
    R = _quat_to_R(np.array([row["rx"], row["ry"], row["rz"], row["rw"]], dtype=float))
    t = np.array([row["tx"], row["ty"], row["tz"]], dtype=float)
    return R, t


def _homography_src_to_tgt(
    K_s: np.ndarray, K_t: np.ndarray,
    R_s: np.ndarray, t_s: np.ndarray,
    R_t: np.ndarray, t_t: np.ndarray,
    plane_depth_m: float,
) -> np.ndarray:
    """Planar homography mapping source-image pixels to target-image pixels.

    Uses a fronto-parallel plane (normal [0,0,1]) at ``plane_depth_m`` in the
    source camera frame: ``H = K_t (R_rel + t_rel n^T / d) K_s^-1`` with
    ``X_tgt = R_rel X_src + t_rel``.
    """
    R_rel = R_t.T @ R_s
    t_rel = R_t.T @ (t_s - t_t)
    n = np.array([[0.0], [0.0], [1.0]])
    H = K_t @ (R_rel + (t_rel.reshape(3, 1) @ n.T) / plane_depth_m) @ np.linalg.inv(K_s)
    return H


def _scaled_H(H: np.ndarray, scale_s: float, scale_t: float) -> np.ndarray:
    """Rescale a homography for images downscaled by ``scale_s`` / ``scale_t``."""
    S_s = np.diag([scale_s, scale_s, 1.0])
    S_t = np.diag([scale_t, scale_t, 1.0])
    H2 = S_t @ H @ np.linalg.inv(S_s)
    return H2 / H2[2, 2]


def _frustum_overlap(
    H: np.ndarray,
    src_wh: tuple[int, int],
    tgt_wh: tuple[int, int],
    scale: float = 0.25,
) -> float:
    """Symmetric 2D overlap ratio between the two views under homography ``H``.

    Warps a full-white source mask into the target frame (and vice versa with
    ``H^-1``) at a reduced ``scale`` for speed and returns
    ``min(frac_of_target_covered, frac_of_source_covered)`` in [0, 1]. This is
    the depth-slice frustum overlap: with a plane at the assumed scene depth,
    it measures how much of each image's footprint lands inside the other.
    """
    import cv2

    sw, sh = max(1, int(src_wh[0] * scale)), max(1, int(src_wh[1] * scale))
    tw, th = max(1, int(tgt_wh[0] * scale)), max(1, int(tgt_wh[1] * scale))
    H_s2t = _scaled_H(H, scale, scale)
    ones_s = np.full((sh, sw), 255, dtype=np.uint8)
    ones_t = np.full((th, tw), 255, dtype=np.uint8)
    warped_s = cv2.warpPerspective(ones_s, H_s2t, (tw, th)) > 0
    frac_t = float(warped_s.sum()) / float(tw * th)
    try:
        warped_t = cv2.warpPerspective(ones_t, np.linalg.inv(H_s2t), (sw, sh)) > 0
        frac_s = float(warped_t.sum()) / float(sw * sh)
    except np.linalg.LinAlgError:
        frac_s = 0.0
    return min(frac_s, frac_t)


class _Undistorter:
    """Caches cv2.fisheye undistort maps per (lens, image size)."""

    def __init__(self, calib: dict[str, dict[str, Any]]):
        self._calib = calib
        self._maps: dict[tuple[str, int, int], tuple[Any, Any]] = {}

    def undistort(self, img: np.ndarray, parity: str) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        key = (parity, w, h)
        if key not in self._maps:
            cam = self._calib[parity]
            K = cam["K"].copy()
            # Rescale intrinsics if the stored image size differs from the
            # calibrated sensor size.
            sx, sy = w / cam["width"], h / cam["height"]
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy
            self._maps[key] = cv2.fisheye.initUndistortRectifyMap(
                K, cam["dist"], np.eye(3), K, (w, h), cv2.CV_16SC2
            )
        m1, m2 = self._maps[key]
        return cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR)

    def K(self, parity: str, img_wh: tuple[int, int] | None = None) -> np.ndarray:
        """Pinhole K for ``parity``, rescaled to ``img_wh`` if given."""
        cam = self._calib[parity]
        K = cam["K"].copy()
        if img_wh is not None:
            K[0, 0] *= img_wh[0] / cam["width"]
            K[0, 2] *= img_wh[0] / cam["width"]
            K[1, 1] *= img_wh[1] / cam["height"]
            K[1, 2] *= img_wh[1] / cam["height"]
        return K


def _fmt_num(x: float) -> str:
    """Format a float for use in a folder name: 0.75 -> '0p75', 1.0 -> '1', 15.0 -> '15'."""
    s = f"{x:g}"
    return s.replace(".", "p")


def _load_inspection(
    conn: sqlite3.Connection, inspection_id: int
) -> tuple[list[dict[str, Any]], list[int]]:
    """All pose-bearing images of one inspection with a known lens, ordered by id.

    L/R is assigned by clustering images on their identical **body-frame** pose
    (``tf_*``) + timestamp: the two cameras of a stereo rig share one body
    pose, so consecutive captures with identical ``tf_translation_*`` /
    ``tf_rotation_*`` and the same ``timestamp_ns`` form an L/R pair (lower id
    = LEFT). Size-1 clusters are unpaired frames whose lens is ambiguous and
    are excluded from matching.

    The pose used for matching is the **camera world pose** stored in the
    ``cam_tf_*`` columns (per-frame translation + quaternion of the camera
    in the world frame). L and R of the same rig have different ``cam_tf_*``
    values (~108-128 deg apart in rotation), so a cross-lens match between
    siblings no longer collapses to rot=0.

    Returns ``(rows, unpaired_ids)``. Each row gets a ``parity`` of "L" or "R"
    and ``tx/ty/tz/rx/ry/rz/rw`` holding the camera world pose.
    """
    raw = conn.execute(
        """
        SELECT id, inspection_id, filename, timestamp_ns,
               tf_translation_x  AS btx, tf_translation_y  AS bty, tf_translation_z  AS btz,
               tf_rotation_x     AS brx, tf_rotation_y     AS bry,
               tf_rotation_z     AS brz, tf_rotation_w     AS brw,
               cam_tf_translation_x AS tx, cam_tf_translation_y AS ty, cam_tf_translation_z AS tz,
               cam_tf_rotation_x    AS rx, cam_tf_rotation_y    AS ry,
               cam_tf_rotation_z    AS rz, cam_tf_rotation_w    AS rw
         FROM images
         WHERE inspection_id = ? AND tf_translation_x IS NOT NULL
                                    AND tf_rotation_w IS NOT NULL
                                    AND cam_tf_translation_x IS NOT NULL
                                    AND cam_tf_rotation_w IS NOT NULL
         ORDER BY id
         """,
        (inspection_id,),
    ).fetchall()

    rows: list[dict[str, Any]] = []
    unpaired: list[int] = []

    # Group consecutive rows by identical (timestamp_ns, BODY-frame pose).
    # The body pose is identical for the L/R pair of one rig; the per-camera
    # pose (cam_tf_*, aliased as tx/..rw above) differs and is matched on later.
    i = 0
    while i < len(raw):
        d = dict(raw[i])
        j = i + 1
        while j < len(raw):
            nxt = raw[j]
            if (nxt["timestamp_ns"] == d["timestamp_ns"]
                    and nxt["btx"] == d["btx"]
                    and nxt["bty"] == d["bty"]
                    and nxt["btz"] == d["btz"]
                    and nxt["brx"] == d["brx"]
                    and nxt["bry"] == d["bry"]
                    and nxt["brz"] == d["brz"]
                    and nxt["brw"] == d["brw"]):
                j += 1
            else:
                break
        grp = [dict(r) for r in raw[i:j]]
        if len(grp) == 2:
            grp[0]["parity"] = "L"  # lower id = LEFT
            grp[1]["parity"] = "R"
            rows.extend(grp)
        elif len(grp) == 1:
            unpaired.append(grp[0]["id"])
        else:
            # Unexpected cluster size (>2): assign L/R to first two, rest unpaired.
            grp[0]["parity"] = "L"
            grp[1]["parity"] = "R"
            rows.extend(grp[:2])
            for r in grp[2:]:
                unpaired.append(r["id"])
        i = j
    return rows, unpaired


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


def _match_all(
    sources: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    max_dist_m: float,
    max_rot_deg: float,
    rot_weight: float,
    cross_lens_penalty: float = 0.1,
    calib: dict[str, dict[str, Any]] | None = None,
    min_overlap: float = 0.0,
    plane_depth_m: float = 3.0,
    overlap_weight: float = 0.0,
    overlap_scale: float = 0.25,
) -> list[dict[str, Any]]:
    """Per-source nearest feasible candidate (targets may be reused).

    Each source independently picks its lowest-cost feasible target. The same
    target can therefore be matched by multiple sources (one-to-many on the
    target side).

    Both same-lens (L->L, R->R) and cross-lens (L->R, R->L) pairs are
    considered. A small ``cross_lens_penalty`` (in metre-equivalent cost
    units) is added to cross-lens pairs so same-lens wins ties - this also
    prevents a source from matching its own L/R sibling, which shares an
    identical pose (trans=0, rot=0) and would otherwise always win.

    When ``calib`` is given, every pose-feasible candidate is additionally
    gated on **frustum overlap**: a planar homography at ``plane_depth_m``
    (fronto-parallel depth slice, using the calibration intrinsics and the
    relative pose) maps each image into the other, and the symmetric overlap
    ratio must be >= ``min_overlap``. The homography and overlap are stored on
    the returned pair so the caller can warp/crop aligned image pairs.

    Cost is ``trans + rot_weight * rot + overlap_weight * (1 - overlap)``
    (plus ``cross_lens_penalty`` for cross-lens pairs).

    Returns one dict per kept pair. Sources with no acceptable candidate are
    omitted; the caller reports them.
    """
    if not sources or not candidates:
        return []

    P = np.array([[s["tx"], s["ty"], s["tz"]] for s in sources], dtype=float)
    C = np.array([[c["tx"], c["ty"], c["tz"]] for c in candidates], dtype=float)
    qP = np.array([[s["rx"], s["ry"], s["rz"], s["rw"]] for s in sources], dtype=float)
    qC = np.array([[c["rx"], c["ry"], c["rz"], c["rw"]] for c in candidates], dtype=float)

    trans = np.sqrt(((P[:, None, :] - C[None, :, :]) ** 2).sum(-1))  # (N, M)
    rot = _quat_rotation_deg(qP, qC)                                  # (N, M)

    src_lens = np.array([s["parity"] for s in sources])[:, None]  # (N, 1)
    tgt_lens = np.array([c["parity"] for c in candidates])[None, :]  # (1, M)
    same_lens = (src_lens == tgt_lens)  # (N, M) bool

    base_cost = trans + rot_weight * rot
    # Add a small penalty to cross-lens pairs so same-lens wins ties.
    base_cost = np.where(same_lens, base_cost, base_cost + cross_lens_penalty)

    # Pose gate: rotation near 0 (both same- and cross-lens). Translation within max_dist_m.
    feasible = (trans <= max_dist_m) & (rot <= max_rot_deg)

    pairs: list[dict[str, Any]] = []
    for i in range(len(sources)):
        feas_idx = np.flatnonzero(feasible[i])
        if feas_idx.size == 0:
            continue
        s = sources[i]
        R_s, t_s = _pose_rt(s)
        best: dict[str, Any] | None = None
        for j in feas_idx:
            c = candidates[int(j)]
            cost = float(base_cost[i, j])
            overlap: float | None = None
            H: np.ndarray | None = None
            if calib is not None:
                cam_s, cam_t = calib[s["parity"]], calib[c["parity"]]
                R_t, t_t = _pose_rt(c)
                H = _homography_src_to_tgt(
                    cam_s["K"], cam_t["K"], R_s, t_s, R_t, t_t, plane_depth_m
                )
                overlap = _frustum_overlap(
                    H,
                    (cam_s["width"], cam_s["height"]),
                    (cam_t["width"], cam_t["height"]),
                    scale=overlap_scale,
                )
                if overlap < min_overlap:
                    continue
                cost += overlap_weight * (1.0 - overlap)
            if best is None or cost < best["cost"]:
                best = {
                    "sampled_id": int(s["id"]),
                    "target_id": int(c["id"]),
                    "parity": s["parity"],
                    "target_parity": c["parity"],
                    "match_type": "same_lens" if same_lens[i, j] else "cross_lens",
                    "translation_m": float(trans[i, j]),
                    "rotation_deg": float(rot[i, j]),
                    "overlap": overlap,
                    "homography_src_to_tgt": H.tolist() if H is not None else None,
                    "cost": cost,
                }
        if best is not None:
            pairs.append(best)
    return pairs


def _nearest_distance(
    src: dict[str, Any], candidates: list[dict[str, Any]],
    cross_lens_penalty: float = 0.1,
) -> tuple[float, float, int | None, str | None]:
    """Nearest candidate's (translation_m, rotation_deg, id, match_type) for diagnostics.

    Considers all candidates (same- and cross-lens), returns the one with the
    lowest ``trans + cross_lens_penalty * (not same lens)`` cost so the
    diagnostic reflects the same logic as the matcher.
    """
    if not candidates:
        return float("inf"), float("inf"), None, None
    P = np.array([[src["tx"], src["ty"], src["tz"]]], dtype=float)
    C = np.array([[c["tx"], c["ty"], c["tz"]] for c in candidates], dtype=float)
    qP = np.array([[src["rx"], src["ry"], src["rz"], src["rw"]]], dtype=float)
    qC = np.array([[c["rx"], c["ry"], c["rz"], c["rw"]] for c in candidates], dtype=float)
    trans = np.sqrt(((P[:, None, :] - C[None, :, :]) ** 2).sum(-1))[0]
    rot = _quat_rotation_deg(qP, qC)[0]
    same = np.array([c["parity"] == src["parity"] for c in candidates], dtype=bool)
    cost = trans + np.where(same, 0.0, cross_lens_penalty)
    j = int(np.argmin(cost))
    mt = "same_lens" if same[j] else "cross_lens"
    return float(trans[j]), float(rot[j]), int(candidates[j]["id"]), mt


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
        merged_path = out_dir / f"{src_ts}_{pp['parity']}__{tgt_ts}_{pp.get('target_parity', pp['parity'])}.jpg"
        if not cv2.imwrite(str(merged_path), canvas):
            print(f"[warn] could not write merged image: {merged_path}", file=sys.stderr)
            continue
        merged.append(merged_path)

    if show and merged:
        print("[info] opening cv2 window of matched pairs (press any key to advance, "
              "ESC to quit)")
        # 2. Create a named window with the resizable flag
        window = "matched pairs (src | tgt)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        # 3. Explicitly resize the window dimensions (width, height)
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
        # L/R pairs share the same timestamp_ns, so append parity to the
        # filename to avoid the second lens overwriting the first.
        src_par = pp.get("parity", "")
        tgt_par = pp.get("target_parity", "")
        src_name = f"{src_ts}_{src_par}.jpg" if src_par else f"{src_ts}.jpg"
        tgt_name = f"{tgt_ts}_{tgt_par}.jpg" if tgt_par else f"{tgt_ts}.jpg"
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


def _write_aligned_pairs(
    pairs: list[dict[str, Any]],
    sampled_dir: Path,
    image_dir: Path,
    out_dir: Path,
    calib: dict[str, dict[str, Any]],
    max_dim: int = 1024,
) -> list[Path]:
    """Write geometry-aligned, overlap-cropped image pairs for VLM input.

    For each matched pair the source (reference) and target images are
    fisheye-undistorted, the target is warped into the source frame with the
    inverse of the stored homography, and both are cropped to the bounding
    box of their common visible area (the warped-target footprint). Per pair,
    three files are written into ``out_dir``:

    - ``<src_ts>_<par>__<tgt_ts>_<tpar>_ref.jpg``  cropped source (reference)
    - ``<src_ts>_<par>__<tgt_ts>_<tpar>_tgt.jpg``  cropped warped target
    - ``<src_ts>_<par>__<tgt_ts>_<tpar>_merged.jpg`` side-by-side debug view

    Crops are downscaled so their longest side is at most ``max_dim``. The
    crop rectangle (in undistorted source pixels) is written back onto each
    pair dict as ``crop_xywh``. Returns the list of merged debug files.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        print(f"[error] aligned output needs opencv (cv2): missing: {e}", file=sys.stderr)
        raise

    undistorter = _Undistorter(calib)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged: list[Path] = []
    for pp in pairs:
        H_list = pp.get("homography_src_to_tgt")
        if H_list is None:
            print(f"[warn] pair {pp['sampled_id']}->{pp['target_id']}: no homography "
                  f"(matched without calibration), skipped", file=sys.stderr)
            continue
        src_path = sampled_dir / f"{pp['sampled_id']}.jpg"
        tgt_path = image_dir / f"{pp['target_id']}.jpg"
        if not src_path.exists():
            src_path = image_dir / f"{pp['sampled_id']}.jpg"
        src_img = cv2.imread(str(src_path))
        tgt_img = cv2.imread(str(tgt_path))
        if src_img is None or tgt_img is None:
            print(f"[warn] pair {pp['sampled_id']}->{pp['target_id']}: missing/"
                  f"undecodable image, skipped", file=sys.stderr)
            continue

        src_und = undistorter.undistort(src_img, pp["parity"])
        tgt_und = undistorter.undistort(tgt_img, pp["target_parity"])
        sh, sw = src_und.shape[:2]
        th, tw = tgt_und.shape[:2]

        # Rescale the homography (computed on calibration-size images) to the
        # actual undistorted image sizes.
        cam_s = calib[pp["parity"]]
        cam_t = calib[pp["target_parity"]]
        H = _scaled_H(
            np.array(H_list, dtype=float),
            sw / cam_s["width"],
            tw / cam_t["width"],
        )
        # Warp target into the source frame.
        H_t2s = np.linalg.inv(H)
        warped_tgt = cv2.warpPerspective(tgt_und, H_t2s, (sw, sh))
        valid = cv2.warpPerspective(
            np.full((th, tw), 255, dtype=np.uint8), H_t2s, (sw, sh)
        ) > 0
        ys, xs = np.nonzero(valid)
        if ys.size == 0:
            print(f"[warn] pair {pp['sampled_id']}->{pp['target_id']}: empty overlap "
                  f"after warp, skipped", file=sys.stderr)
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        pp["crop_xywh"] = [x0, y0, x1 - x0, y1 - y0]

        ref_crop = src_und[y0:y1, x0:x1]
        tgt_crop = warped_tgt[y0:y1, x0:x1]
        scale = min(1.0, max_dim / max(ref_crop.shape[:2]))
        if scale < 1.0:
            new_wh = (int(ref_crop.shape[1] * scale), int(ref_crop.shape[0] * scale))
            ref_crop = cv2.resize(ref_crop, new_wh, interpolation=cv2.INTER_AREA)
            tgt_crop = cv2.resize(tgt_crop, new_wh, interpolation=cv2.INTER_AREA)

        src_ts = pp.get("sampled_timestamp", pp["sampled_id"])
        tgt_ts = pp.get("target_timestamp", pp["target_id"])
        stem = f"{src_ts}_{pp['parity']}__{tgt_ts}_{pp['target_parity']}"
        cv2.imwrite(str(out_dir / f"{stem}_ref.jpg"), ref_crop)
        cv2.imwrite(str(out_dir / f"{stem}_tgt.jpg"), tgt_crop)
        side = np.hstack([ref_crop, tgt_crop])
        merged_path = out_dir / f"{stem}_merged.jpg"
        cv2.imwrite(str(merged_path), side)
        merged.append(merged_path)
    return merged


def _resample_images(
    conn: sqlite3.Connection,
    out_dir: Path,
    src_dir: Path,
    inspection: int,
    interval_m: float,
    start_index: int,
    lens: str,
    keep_existing: bool,
) -> int:
    """Populate ``out_dir`` with images sampled along ``inspection``'s trajectory.

    Reuses ``sample_images_along_trajectory``'s pose clustering and greedy
    arc-length sampling, then copies the picked <id>.jpg files from ``src_dir``
    into ``out_dir`` (cleared first unless ``keep_existing``). Returns the number
    of images copied, or -1 on error.
    """
    views, unpaired = sampler._load_pairs(conn, inspection)
    if not views:
        print(f"[error] no pose-bearing viewpoints in inspection {inspection}", file=sys.stderr)
        return -1
    if unpaired:
        print(f"[warn] re-sampling inspection {inspection}: {len(unpaired)} unpaired "
              f"(lens-ambiguous) image(s) excluded: {unpaired}", file=sys.stderr)

    picked = sampler._sample_by_arclength(views, interval_m, start_index)

    sampled_ids: list[int] = []
    for v in picked:
        if lens == "left":
            sampled_ids.append(v["left_id"])
        elif lens == "right":
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(_DEFAULT_DB), help="Path to inspection_v2_mtr_new.db")
    p.add_argument(
        "--sampled-dir",
        default=str(_DEFAULT_SAMPLED_DIR),
        help="Directory of <id>.jpg source images from inspection 1",
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
        "--sample-lens",
        choices=["left", "right", "both"],
        default="left",
        help="Which lens of each L/R pair to copy when sampling (default left). "
             "Ignored when --select-pair is set (both lenses are taken).",
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
    p.add_argument(
        "--calibration",
        default=str(_DEFAULT_CALIBRATION),
        help="Camera calibration JSON with per-lens intrinsics + fisheye "
             "distortion (default: info/calibration.json next to this script)",
    )
    p.add_argument(
        "--min-overlap",
        type=float,
        default=0.5,
        help="Minimum symmetric frustum overlap ratio [0..1] for a valid pair "
             "(default 0.5). Computed via a planar homography at "
             "--plane-depth-m using the calibration intrinsics.",
    )
    p.add_argument(
        "--plane-depth-m",
        type=float,
        default=3.0,
        help="Assumed scene depth (m) of the fronto-parallel plane used for "
             "the frustum-overlap homography (default 3.0)",
    )
    p.add_argument(
        "--overlap-weight",
        type=float,
        default=0.5,
        help="Cost weight on (1 - overlap); higher prefers larger common "
             "visible area over smaller pose error (default 0.5)",
    )
    p.add_argument(
        "--aligned-dir",
        default=None,
        help="Write aligned/cropped image pairs (undistorted, target warped "
             "into the source frame, cropped to the common overlap ROI) into "
             "this folder, ready for VLM comparison.",
    )
    p.add_argument("--max-dist-m", type=float, default=1.5, help="Max translation (m) for a valid pair")
    p.add_argument("--max-rot-deg", type=float, default=12.0,
                   help="Max rotation (deg) for a matched pair (near 0 deg)")
    p.add_argument("--rot-weight", type=float, default=0.15, help="m-per-deg rotation weight in the cost")
    p.add_argument(
        "--cross-lens-penalty",
        type=float,
        default=0.1,
        help="Cost (in metre-equivalent units) added to cross-lens (L->R, R->L) "
             "pairs so same-lens wins ties. Set to a large value to effectively "
             "disable cross-lens matching (default 0.1).",
    )
    p.add_argument(
        "--select-pair",
        action="store_true",
        help="Select both lenses of each sampled viewpoint (the full L+R pair) "
             "as the source set, instead of only --sample-lens. Matching still "
             "runs per parity class (L->L, R->R).",
    )
    p.add_argument("--json", dest="json_path", default=None, help="Write the pair list to this JSON file")
    p.add_argument("--image-dir", default=str(_DEFAULT_IMAGE_DIR), help="Image folder for reported target file paths")
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
    sampled_dir = Path(args.sampled_dir).resolve()

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
    # Override output paths with clean names inside the run folder.
    args.json_path = str(run_dir / "pairs.json")
    args.config_out = str(run_dir / "config.json")
    args.plot_out = str(run_dir / "trajectory.png")
    args.matched_dir = str(run_dir / label)
    args.copy_split_dir = str(run_dir / "split_pairs")
    args.aligned_dir = str(run_dir / "aligned_pairs")
    args.plot = True

    calib_path = Path(args.calibration).resolve()
    if not calib_path.exists():
        print(f"[error] calibration not found: {calib_path}", file=sys.stderr)
        return 2
    calib = _load_calibration(calib_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    if args.sample_interval_m is not None:
        lens = "both" if args.select_pair else args.sample_lens
        n = _resample_images(
            conn,
            sampled_dir,
            Path(args.image_dir),
            args.sample_inspection,
            args.sample_interval_m,
            args.sample_start_index,
            lens,
            args.keep_sampled,
        )
        if n < 0:
            conn.close()
            return 2
        print(f"[info] sampled {n} image(s) from inspection {args.sample_inspection} "
              f"at {args.sample_interval_m} m into {sampled_dir}"
              f"{' (L+R pairs via --select-pair)' if args.select_pair else ''}")
    try:
        target_rows, tgt_unpaired = _load_inspection(conn, args.target_inspection)
        if not target_rows:
            print(f"[error] no pose-bearing L/R pairs in inspection {args.target_inspection}", file=sys.stderr)
            return 2
        if tgt_unpaired:
            print(f"[info] target inspection {args.target_inspection}: {len(tgt_unpaired)} unpaired "
                  f"image(s) excluded (ambiguous lens): {tgt_unpaired}", file=sys.stderr)

        # Resolve the source set.
        if args.source_inspection is not None:
            source_rows, src_unpaired = _load_inspection(conn, args.source_inspection)
            if src_unpaired:
                print(f"[info] source inspection {args.source_inspection}: {len(src_unpaired)} unpaired "
                      f"image(s) excluded (ambiguous lens): {src_unpaired}", file=sys.stderr)
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
                if not sampled_dir.exists():
                    print(f"[error] sampled dir not found: {sampled_dir}", file=sys.stderr)
                    return 2
                source_ids = _load_sampled_ids(sampled_dir)
                if not source_ids:
                    print(f"[error] no <id>.jpg files found in {sampled_dir}", file=sys.stderr)
                    return 2
            src_insp = _resolve_source_inspection(conn, source_ids)
            if src_insp is None:
                print("[error] could not resolve which inspection the sampled images belong to", file=sys.stderr)
                return 2
            # L/R is assigned by pose-clustering within the source inspection.
            all_src, src_unpaired = _load_inspection(conn, src_insp)
            id_to_row = {r["id"]: r for r in all_src}
            unpaired_set = set(src_unpaired)
            source_rows: list[dict[str, Any]] = []
            missing: list[int] = []
            ambiguous: list[int] = []
            for sid in source_ids:
                r = id_to_row.get(sid)
                if r is not None:
                    source_rows.append(r)
                elif sid in unpaired_set:
                    ambiguous.append(sid)
                else:
                    missing.append(sid)
            if missing:
                print(f"[warn] {len(missing)} sampled id(s) not found in inspection "
                      f"{src_insp} (skipped): {missing}", file=sys.stderr)
            if ambiguous:
                print(f"[warn] {len(ambiguous)} sampled id(s) are unpaired in inspection "
                      f"{src_insp} (lens ambiguous, skipped): {ambiguous}", file=sys.stderr)
            src_label = f"{len(source_rows)} sampled image(s) from inspection {src_insp}"

        by_parity_src = {"L": [r for r in source_rows if r["parity"] == "L"],
                         "R": [r for r in source_rows if r["parity"] == "R"]}
        by_parity_tgt = {"L": [r for r in target_rows if r["parity"] == "L"],
                         "R": [r for r in target_rows if r["parity"] == "R"]}

        print(f"[info] db: {db_path}")
        print(f"[info] source: {src_label}")
        print(f"[info] target: inspection {args.target_inspection} "
              f"({len(target_rows)} imgs: {len(by_parity_tgt['L'])} L / {len(by_parity_tgt['R'])} R)")
        print(f"[info] source parity: {len(by_parity_src['L'])} L / {len(by_parity_src['R'])} R")
        print(f"[info] gate: max_dist={args.max_dist_m} m, max_rot={args.max_rot_deg} deg, "
              f"min_overlap={args.min_overlap}, rot_weight={args.rot_weight}, "
              f"overlap_weight={args.overlap_weight}, "
              f"cross_lens_penalty={args.cross_lens_penalty}")
        print(f"[info] calibration: {calib_path} (plane_depth={args.plane_depth_m} m)")

        all_pairs = _match_all(
            source_rows, target_rows,
            args.max_dist_m, args.max_rot_deg,
            args.rot_weight,
            args.cross_lens_penalty,
            calib=calib,
            min_overlap=args.min_overlap,
            plane_depth_m=args.plane_depth_m,
            overlap_weight=args.overlap_weight,
        )
        matched_src_ids: set[int] = {pp["sampled_id"] for pp in all_pairs}
        print(f"[info] matched {len(all_pairs)}/{len(source_rows)} source image(s) "
              f"against {len(target_rows)} candidate(s)")

        # Candidate pool for unmatched diagnostics.
        diag_tgt = target_rows

        # Diagnostics for unmatched sources: nearest candidate distance.
        unmatched = [r for r in source_rows if r["id"] not in matched_src_ids]
        if unmatched:
            print(f"\n[info] {len(unmatched)} unmatched source image(s) - "
                  f"nearest candidate:")
            for r in sorted(unmatched, key=lambda x: x["id"]):
                nt, nr, nid, mt = _nearest_distance(r, diag_tgt, args.cross_lens_penalty)
                print(f"  src id={r['id']:>4} ({r['parity']})  nearest target id={nid}  "
                      f"trans={nt:.3f} m  rot={nr:.2f} deg  match_type={mt}  "
                      f"(over gate -> no match)")

        all_pairs.sort(key=lambda pp: pp["sampled_id"])

        # Enrich pairs with timestamps from the database (used for output
        # filenames and JSON keys). Parity (L/R) is kept for reference.
        ts_lookup = _load_timestamps(conn)
        for pp in all_pairs:
            pp["sampled_timestamp"] = ts_lookup.get(pp["sampled_id"])
            pp["target_timestamp"] = ts_lookup.get(pp["target_id"])

        if args.aligned_dir:
            aligned_dir = Path(args.aligned_dir).resolve()
            merged = _write_aligned_pairs(
                all_pairs, sampled_dir, Path(args.image_dir), aligned_dir, calib,
            )
            print(f"[info] wrote {len(merged)} aligned/cropped pair(s) to {aligned_dir}")

        if args.plot:
            src_insp = args.sample_inspection if args.sample_interval_m is not None \
                else (args.source_inspection if args.source_inspection is not None
                      else args.sample_inspection)
            _plot_trajectories(conn, src_insp, args.target_inspection,
                               pairs=all_pairs, out_path=args.plot_out)

        print("\n=== matched pairs ===")
        print(f"{'src_id':>7} {'parity':>6} {'tgt_id':>7} {'tgt_par':>7} {'type':>9} "
              f"{'trans_m':>9} {'rot_deg':>8} {'overlap':>8} {'cost':>7}")
        for pp in all_pairs:
            ov = f"{pp['overlap']:.3f}" if pp.get("overlap") is not None else "-"
            print(f"{pp['sampled_id']:>7} {pp['parity']:>6} {pp['target_id']:>7} {pp['target_parity']:>7} "
                  f"{pp['match_type']:>9} {pp['translation_m']:>9.3f} {pp['rotation_deg']:>8.2f} "
                  f"{ov:>8} {pp['cost']:>7.3f}")

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
                    "parity": pp["parity"],
                    "target_parity": pp["target_parity"],
                    "match_type": pp["match_type"],
                    "translation_m": pp["translation_m"],
                    "rotation_deg": pp["rotation_deg"],
                    "overlap": pp.get("overlap"),
                    "cost": pp["cost"],
                    "homography_src_to_tgt": pp.get("homography_src_to_tgt"),
                    "crop_xywh": pp.get("crop_xywh"),
                    "sampled_timestamp": pp.get("sampled_timestamp"),
                    "target_timestamp": pp.get("target_timestamp"),
                }
                json_pairs.append(jp)
            Path(args.json_path).write_text(json.dumps(json_pairs, indent=2))
            print(f"[info] wrote {len(json_pairs)} matched pair(s) to {args.json_path}")

        if args.matched_dir:
            matched_dir = Path(args.matched_dir).resolve()
            matched_dir.mkdir(parents=True, exist_ok=True)
            merged = _copy_matched_images(
                all_pairs, sampled_dir, Path(args.image_dir), matched_dir,
                show=not args.no_show,
            )
            print(f"[info] wrote {len(merged)} merged pair image(s) to {matched_dir}")
            if merged and args.no_show:
                print("[info] --no-show set: skipping cv2 display window")

        if args.copy_split_dir:
            split_dir = Path(args.copy_split_dir).resolve()
            n_src, n_tgt = _copy_split_images(
                all_pairs, sampled_dir, Path(args.image_dir), split_dir,
            )
            print(f"[info] copied {n_src} source / {n_tgt} target image(s) to "
                  f"{split_dir / 'source'} and {split_dir / 'target'}")

        # Results summary: matching stats + names of unmatched images.
        print("\n=== results summary ===")
        print(f"  source images:      {len(source_rows)}")
        print(f"  matched pairs:      {len(all_pairs)}")
        print(f"  unmatched images:   {len(unmatched)}")
        if all_pairs:
            t = [pp["translation_m"] for pp in all_pairs]
            r = [pp["rotation_deg"] for pp in all_pairs]
            print(f"  translation (m):   min={min(t):.3f} med={sorted(t)[len(t)//2]:.3f} "
                  f"max={max(t):.3f}")
            print(f"  rotation (deg):    min={min(r):.2f} med={sorted(r)[len(r)//2]:.2f} "
                  f"max={max(r):.2f}")
        if unmatched:
            print("\n  unmatched image(s) (name -> id | nearest cost):")
            for r in sorted(unmatched, key=lambda x: x["id"]):
                nt, nr, nid, mt = _nearest_distance(r, diag_tgt, args.cross_lens_penalty)
                cost = nt + args.rot_weight * nr
                name = r["filename"] or f"{r['id']}.jpg"
                print(f"    {name}  (id={r['id']}, parity={r['parity']}, "
                      f"nearest tgt={nid}, match_type={mt}, trans={nt:.3f} m, "
                      f"rot={nr:.2f} deg, cost={cost:.3f})")

        if args.commit:
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
                "image_dir": args.image_dir,
                "sampled_dir": str(sampled_dir),
                "source_inspection": args.source_inspection,
                "target_inspection": args.target_inspection,
                "sample_inspection": args.sample_inspection,
                "sample_interval_m": args.sample_interval_m,
                "sample_start_index": args.sample_start_index,
                "sample_lens": "both" if args.select_pair else args.sample_lens,
                "select_pair": args.select_pair,
                "keep_sampled": args.keep_sampled,
                "max_dist_m": args.max_dist_m,
                "max_rot_deg": args.max_rot_deg,
                "rot_weight": args.rot_weight,
                "cross_lens_penalty": args.cross_lens_penalty,
                "calibration": str(calib_path),
                "min_overlap": args.min_overlap,
                "plane_depth_m": args.plane_depth_m,
                "overlap_weight": args.overlap_weight,
                "gt_db_source": args.gt_db,
                "aligned_dir": args.aligned_dir,
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
