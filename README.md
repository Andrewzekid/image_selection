# image_selection

Pose-based tooling for selecting and matching inspection images across multiple
runs of the same route. All scripts assume the shared `camera_init` (FastLIO)
global frame, so poses from different inspections are directly comparable
without any cross-run alignment.

## Scripts

### `sample_images_along_trajectory.py`

Walks one inspection's trajectory in capture order and greedily picks a
viewpoint whenever the cumulative arc-length since the last pick is `>= --interval-m`.
The LEFT image of each picked L/R pair is copied to `--out-dir` as `<id>.jpg`
— the exact layout `match_images_by_pose.py` consumes as its source set.

```bash
python sample_images_along_trajectory.py \
    --inspection 1 --interval-m 1.25 \
    --out-dir ../inspection_database/sampled_images
```

### `match_images_by_pose.py`

For each source image (default: the `<id>.jpg` files in `--sampled-dir`, a
subset of inspection 1), finds the image in a target inspection taken from
the same viewpoint by comparing the **camera world pose** stored on the
`images` table: `cam_tf_translation_{x,y,z}` + quaternion
`cam_tf_rotation_{x,y,z,w}`. These are per-frame camera poses in the shared
`camera_init` (FastLIO) global frame; L and R of the same rig have different
`cam_tf_*` values (~108-128° apart in rotation), so cross-lens siblings
don't collapse to rot=0.

L/R lens assignment still clusters on the **body-frame** pose (`tf_*` +
`timestamp_ns`), since the two cameras of a stereo rig share one body pose.
The default database is `inspection_database/complete2/inspection_v2.db`
(where `cam_tf_*` is per-frame); pass `--db` to use another DB.

**Matching mode:** per-source nearest feasible candidate. Each source
independently picks its lowest-cost feasible target, so the **same target may
be matched by multiple sources** (one-to-many on the target side). Feasibility
is gated by:

- `--max-dist-m` (translation, metres)
- `--max-rot-deg` (geodesic quaternion rotation, degrees)
- both same-lens (L→L, R→R) **and** cross-lens (L→R, R→L) pairs are
  considered; a small `--cross-lens-penalty` (default 0.1 m-equivalent) is
  added to cross-lens cost so same-lens wins ties. Set the penalty high to
  effectively disable cross-lens matching.

Cost = `translation_m + rot_weight * rotation_deg` (plus `cross_lens_penalty`
for cross-lens pairs).

L/R lens assignment is done by clustering images within each inspection on
identical pose + `timestamp_ns`: every size-2 cluster is one L/R pair (lower
id = LEFT); size-1 clusters are unpaired frames whose lens is ambiguous and
are excluded from matching. Because an L and R of the same pose share an
identical pose, the cross-lens penalty is what stops a source from matching
its own sibling.

Outputs (auto-routed into `inspection_database/runs/<run_name>/`):

| File / dir            | Contents                                              |
| --------------------- | ----------------------------------------------------- |
| `pairs.json`          | Matched pair list (src/tgt ids, parity, trans, rot)   |
| `inspXvsY/`           | Merged side-by-side images (src left, tgt right)      |
| `split_pairs/source/` | Source images as `<timestamp>_<L|R>.jpg`              |
| `split_pairs/target/` | Target images as `<timestamp>_<L|R>.jpg`             |
| `trajectory.png`      | Source/target trajectories with matched links         |
| `config.json`         | Parameters and result counts for this run             |

Example — match L+R pairs from inspection 1 against inspection 4 at 1.25 m
sampling with a 1.5 m / 12 deg gate:

```bash
python match_images_by_pose.py \
    --target-inspection 4 --sample-inspection 1 \
    --sample-interval-m 1.25 --select-pair --no-show
```

Use `--source-inspection N` to match *all* images of inspection N instead of a
sampled subset. Use `--commit` to insert matched pairs into the
`abnormal_detections` table (off by default; existing rows are preserved unless
`--no-skip-existing` is set).

### `plot_trajectory_heatmap.py`

Plots one or more inspection trajectories as a heatmap where each point is
colored by its capture index (red = start → blue = end), so capture order is
visible at a glance. Supports single inspection, multiple overlaid, per-inspection
files, or all inspections on one combined map.

```bash
# Single inspection
python plot_trajectory_heatmap.py --inspection 4 --out ../inspection_database/heatmap_insp4.png

# Multiple inspections overlaid
python plot_trajectory_heatmap.py --inspection 1 4 --out ../inspection_database/heatmap_1_4.png

# All inspections, one combined map
python plot_trajectory_heatmap.py --all --combined --out ../inspection_database/heatmap_all.png
```

## Requirements

Run with the backend venv (provides `numpy`, `opencv-python`, `matplotlib`):

```bash
../backend/.venv/bin/python <script>.py ...
```

## Notes

- `_REPO_ROOT` is resolved from each script's location, so commands work from
  any working directory. Scripts in this folder use `parents[1]` (one level up
  to the repo root); the copies in `backend/scripts/` use `parents[2]`.
- Default DB is `inspection_database/inspection_v2.db`; default image dir is
  `inspection_database/outputs/images/`.