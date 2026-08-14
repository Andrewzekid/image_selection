#!/usr/bin/env python3
"""Plot inspection trajectories from the DB as a heatmap where each point is
colored by its **capture index** (position in the ordered sequence), so the
color gradually transitions from red (start) through light shades to blue
(end) along the route — making capture order visible at a glance.

The trajectory is drawn as a light gray path. Each image point is plotted
semi-transparently, colored by its index (0 = red, N-1 = blue) on a smooth
red -> blue gradient.

Multiple inspections can be overlaid on the same map (each gets its own gray
trajectory and index-colored points, identified in the legend).

Run with the backend venv so matplotlib/numpy are available, e.g.::

    # Single inspection
    python backend/scripts/plot_trajectory_heatmap.py \
        --inspection 3 --out inspection_database/trajectory_heatmap_insp3.png

    # Multiple inspections on one map
    python backend/scripts/plot_trajectory_heatmap.py \
        --inspection 1 2 3 --out inspection_database/trajectory_heatmap_1_2_3.png

    # All inspections, one file each
    python backend/scripts/plot_trajectory_heatmap.py \
        --all --out-dir inspection_database

    # All inspections on a single combined map
    python backend/scripts/plot_trajectory_heatmap.py \
        --all --combined --out inspection_database/trajectory_heatmap_all.png
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Repo root is one level up from this script (image_selection/ -> repo root).
INSPECTION_DIRECTORY = "inspection_database"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DB = _REPO_ROOT / INSPECTION_DIRECTORY / "complete2" / "inspection_v2.db"


def _load_inspection_points(
    conn: sqlite3.Connection, inspection_id: int, lens: str = "left",
    ground_plane: str = "xz",
) -> list[tuple[int, float, float, int]]:
    """Return list of (id, gx, gy, timestamp_ns) ordered by id for one inspection.

    ``gx``/``gy`` are the two ground-plane coordinates taken from the
    **camera world pose** (``cam_tf_*``), not the body/lidar pose, so the
    plotted trajectory is the path of the actual camera. ``ground_plane``
    selects which pair of camera axes to plot:

    - ``xy`` (default) — cam_tf_x vs cam_tf_y  (body/lidar ground plane)
    - ``xz``           — cam_tf_x vs cam_tf_z  (camera ground plane for this rig)
    - ``yz``           — cam_tf_y vs cam_tf_z

    The camera is rotated ~90 deg about one axis relative to the body, so its
    ground plane is typically ``xz`` even when the body's ground plane is ``xy``.

    ``lens`` selects which lens to plot:

    - ``left``  — only the LEFT camera of each L/R pair (default)
    - ``right`` — only the RIGHT camera
    - ``both``  — every frame (L and R, in capture order)

    L/R is assigned by clustering consecutive captures on identical body-frame
    pose (``tf_*``) + ``timestamp_ns``; lower id = LEFT. Only rows with
    non-null body pose, camera pose, and timestamp are returned.
    """
    axis_map = {"x": "ctx", "y": "cty", "z": "ctz"}
    h_axis, v_axis = ground_plane[0], ground_plane[1]
    h_col, v_col = axis_map[h_axis], axis_map[v_axis]
    print(f"H_axis={h_axis} ({h_col}), V_axis={v_axis} ({v_col})")
    raw = conn.execute(
        f"""
        SELECT id, timestamp_ns,
               tf_translation_x  AS btx, tf_translation_y  AS bty, tf_translation_z  AS btz,
               tf_rotation_x     AS brx, tf_rotation_y     AS bry,
               tf_rotation_z     AS brz, tf_rotation_w     AS brw,
               cam_tf_translation_x AS ctx, cam_tf_translation_y AS cty, cam_tf_translation_z AS ctz
         FROM images
         WHERE inspection_id = ?
           AND tf_translation_x IS NOT NULL
           AND tf_rotation_w IS NOT NULL
           AND cam_tf_translation_x IS NOT NULL
           AND cam_tf_translation_y IS NOT NULL
           AND cam_tf_translation_z IS NOT NULL
           AND timestamp_ns IS NOT NULL
         ORDER BY id
         """,
        (inspection_id,),
    ).fetchall()

    def _pt(r):
        return (int(r["id"]), float(r[h_col]), float(r[v_col]), int(r["timestamp_ns"]))

    if lens == "both":
        return [_pt(r) for r in raw]

    # Cluster consecutive rows on identical body pose + timestamp -> L/R pair.
    pts: list[tuple[int, float, float, int]] = []
    i = 0
    while i < len(raw):
        d = raw[i]
        j = i + 1
        while j < len(raw):
            nxt = raw[j]
            if (nxt["timestamp_ns"] == d["timestamp_ns"]
                    and nxt["btx"] == d["btx"] and nxt["bty"] == d["bty"]
                    and nxt["btz"] == d["btz"]
                    and nxt["brx"] == d["brx"] and nxt["bry"] == d["bry"]
                    and nxt["brz"] == d["brz"] and nxt["brw"] == d["brw"]):
                j += 1
            else:
                break
        grp = list(raw[i:j])
        if len(grp) >= 2:
            # Lower id = LEFT, higher = RIGHT.
            r = grp[0] if lens == "left" else grp[1]
            pts.append(_pt(r))
        elif len(grp) == 1 and lens == "left":
            # Unpaired singleton: keep it so the trajectory isn't gappy.
            pts.append(_pt(grp[0]))
        i = j
    return pts


def _red_blue_cmap():
    """Custom smooth colormap: red -> light red -> orange -> light blue -> blue.

    Red = start of capture (index 0), blue = end of capture (last index),
    with a gradual perceptually monotonic transition so consecutive points
    differ only slightly in color.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "red_to_blue",
        [
            (0.00, "#a50f15"),   # dark red (start)
            (0.20, "#e6550d"),   # orange-red
            (0.40, "#fdae6b"),   # light orange
            (0.50, "#fee08b"),   # pale yellow (midpoint)
            (0.60, "#c6dbef"),   # very pale blue
            (0.80, "#6baed6"),   # light blue
            (1.00, "#08306b"),   # dark blue (end)
        ],
    )


def _plot_combined(
    inspections: list[tuple[int, list[tuple[int, float, float, int]]]],
    cmap_name: str | None,
    out_path: Path | None,
    show: bool,
    title_extra: str = "",
    ground_plane: str = "xz",
) -> None:
    """Plot multiple inspection trajectories on one map.

    Each inspection is drawn as a light gray trajectory path with
    semi-transparent points colored by capture index (red=start, blue=end).
    A single shared colorbar encodes the index within each inspection
    (normalized 0..1).
    """
    cmap = plt.get_cmap(cmap_name) if cmap_name else _red_blue_cmap()

    fig, ax = plt.subplots(figsize=(14, 10))

    # Normalized index 0..1 for each inspection's points — used for color.
    sc_handle = None
    for insp_id, pts in inspections:
        if len(pts) < 2:
            print(f"[warn] inspection {insp_id}: only {len(pts)} pose-bearing "
                  f"point(s), skipping on combined plot", file=sys.stderr)
            continue
        xs = np.array([p[1] for p in pts])
        zs = np.array([p[2] for p in pts])
        n = len(pts)
        idx_norm = np.linspace(0.0, 1.0, n)  # 0 at start, 1 at end

        # Trajectory as light gray path.
        ax.plot(xs, zs, "-", color="lightgray", linewidth=2.5, zorder=1)

        # Semi-transparent points colored by capture index.
        sc = ax.scatter(xs, zs, c=idx_norm, cmap=cmap,
                        s=50, alpha=0.5, edgecolors="none", zorder=2)
        if sc_handle is None:
            sc_handle = sc

        # Start / end markers
        ax.plot(xs[0], zs[0], "^", color="lime", markersize=10,
                markeredgecolor="black", zorder=5)
        ax.plot(xs[-1], zs[-1], "v", color="red", markersize=10,
                markeredgecolor="black", zorder=5)

        # Legend entry for this inspection (use a proxy line in gray).
        ax.plot([], [], "-", color="lightgray", linewidth=3.0,
                label=f"inspection {insp_id} ({n} pts)")

    if sc_handle is not None:
        cbar = fig.colorbar(sc_handle, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("capture index (0=start → 1=end, per inspection)")

    insp_ids = [str(i) for i, _ in inspections]
    title = f"Trajectory heatmap — inspection(s) {', '.join(insp_ids)}"
    if title_extra:
        title += f"  {title_extra}"
    ax.set_title(title)
    ax.set_xlabel(f"cam_tf {ground_plane[0]} (m)")
    ax.set_ylabel(f"cam_tf {ground_plane[1]} (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(str(out_path), dpi=150)
        print(f"[info] saved combined trajectory heatmap to {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def _plot_one(
    inspection_id: int,
    pts: list[tuple[int, float, float, int]],
    cmap_name: str | None,
    out_path: Path | None,
    show: bool,
    ground_plane: str = "xz",
) -> None:
    """Plot a single inspection trajectory colored by capture index."""
    if len(pts) < 2:
        print(f"[warn] inspection {inspection_id}: only {len(pts)} pose-bearing "
              f"point(s), skipping plot", file=sys.stderr)
        return

    ids = np.array([p[0] for p in pts])
    xs = np.array([p[1] for p in pts])
    zs = np.array([p[2] for p in pts])
    ts = np.array([p[3] for p in pts], dtype=float)
    n = len(pts)
    idx_norm = np.linspace(0.0, 1.0, n)  # 0 at start, 1 at end

    cmap = plt.get_cmap(cmap_name) if cmap_name else _red_blue_cmap()
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(13, 9))

    # Draw the full trajectory as a light gray path for context.
    ax.plot(xs, zs, "-", color="lightgray", linewidth=2.5, zorder=1, label="trajectory")

    # Semi-transparent scatter of points colored by capture index
    # (red=start, blue=end, gradual transition).
    sc = ax.scatter(xs, zs, c=idx_norm, cmap=cmap, norm=norm,
                    s=50, alpha=0.5, edgecolors="none", zorder=2)

    # Start / end markers
    ax.plot(xs[0], zs[0], "^", color="lime", markersize=12,
            markeredgecolor="black", label="start", zorder=5)
    ax.plot(xs[-1], zs[-1], "v", color="red", markersize=12,
            markeredgecolor="black", label="end", zorder=5)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("capture index (0=start → 1=end)")

    title = (f"Inspection {inspection_id} trajectory heatmap "
             f"({n} pts, span {(ts[-1]-ts[0])/1e9:.1f}s)")
    ax.set_title(title)
    ax.set_xlabel(f"cam_tf {ground_plane[0]} (m)")
    ax.set_ylabel(f"cam_tf {ground_plane[1]} (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(str(out_path), dpi=150)
        print(f"[info] saved trajectory heatmap for inspection {inspection_id} "
              f"to {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default=str(_DEFAULT_DB),
                   help="Path to inspection_v2.db")
    p.add_argument("--inspection", type=int, nargs="+", default=None,
                   help="Inspection id(s) to plot. Pass multiple to overlay "
                        "them on one map (e.g. --inspection 1 2 3).")
    p.add_argument("--all", action="store_true",
                   help="Plot every inspection in the DB.")
    p.add_argument("--combined", action="store_true",
                   help="With --all (or multiple --inspection), draw all "
                        "trajectories on a single combined map instead of "
                        "one file each.")
    p.add_argument("--cmap", default=None,
                   help="matplotlib colormap to override the default "
                        "red->blue gradient")
    p.add_argument("--out", default=None,
                   help="Output PNG path. Default: "
                        "inspection_database/trajectory_heatmap_insp<N>.png "
                        "(single) or trajectory_heatmap_combined.png (combined).")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for per-inspection files in --all "
                        "non-combined mode. Default: inspection_database/")
    p.add_argument("--lens", choices=["left", "right", "both"], default="left",
                   help="Which camera lens to plot: left (default), right, or both. "
                        "Uses the camera world pose (cam_tf_*); L/R is assigned by "
                        "clustering on the body-frame pose (tf_*).")
    p.add_argument("--ground-plane", choices=["xy", "xz", "yz"], default="xz",
                   help="Which pair of camera axes to plot as the ground plane. "
                        "The camera is rotated ~90 deg relative to the body, so its "
                        "ground plane is xz (default) even though the body/lidar "
                        "ground plane is xy. Use xy for the body-frame convention.")
    p.add_argument("--show", action="store_true",
                   help="Also show the plot interactively (needs a display)")
    args = p.parse_args(argv)

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"[error] database not found: {db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Resolve the set of inspection ids to plot.
    if args.all:
        insp_ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT inspection_id FROM images ORDER BY inspection_id"
        ).fetchall()]
    elif args.inspection is not None:
        insp_ids = list(args.inspection)
    else:
        print("[error] provide --inspection <id> [<id> ...] or --all",
              file=sys.stderr)
        conn.close()
        return 2

    # Load points for each requested inspection.
    loaded: list[tuple[int, list[tuple[int, float, float, int]]]] = []
    for insp in insp_ids:
        pts = _load_inspection_points(conn, insp, args.lens, args.ground_plane)
        if not pts:
            print(f"[warn] no pose-bearing points for inspection {insp}, "
                  f"skipping", file=sys.stderr)
            continue
        loaded.append((insp, pts))
    conn.close()

    if not loaded:
        print("[error] no inspections with pose-bearing points found",
              file=sys.stderr)
        return 2

    # Decide mode: combined single map vs per-inspection files.
    use_combined = args.combined or (args.all is False and len(loaded) > 1)

    if use_combined:
        out = Path(args.out).resolve() if args.out \
            else _REPO_ROOT / INSPECTION_DIRECTORY / "trajectory_heatmap_combined.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        _plot_combined(loaded, args.cmap, out, args.show,
                       ground_plane=args.ground_plane)
    elif args.all:
        # One file per inspection.
        out_dir = Path(args.out_dir).resolve() if args.out_dir \
            else _REPO_ROOT / INSPECTION_DIRECTORY
        out_dir.mkdir(parents=True, exist_ok=True)
        for insp, pts in loaded:
            out = out_dir / f"trajectory_heatmap_insp{insp}.png"
            _plot_one(insp, pts, args.cmap, out, args.show,
                      ground_plane=args.ground_plane)
    else:
        # Single inspection.
        insp, pts = loaded[0]
        print(f"[info] plotting inspection {insp} with {len(pts)} pose-bearing points {pts[:5]}")
        out = Path(args.out).resolve() if args.out \
            else _REPO_ROOT / INSPECTION_DIRECTORY / f"trajectory_heatmap_insp{insp}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        _plot_one(insp, pts, args.cmap, out, args.show,
                  ground_plane=args.ground_plane)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())