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

L/R lens assignment clusters on the **body-frame** pose (`tf_*` +
`timestamp_ns`), since the two cameras of a stereo rig share one body pose.
Every size-2 cluster is one L/R pair (lower id = LEFT); size-1 clusters are
unpaired frames whose lens is ambiguous and are excluded from matching.

The default database is `inspection_database/complete2/inspection_v2.db`
(where `cam_tf_*` is per-frame); pass `--db` to use another DB.

**Matching mode:** per-source nearest feasible candidate. Each source
independently picks its lowest-cost feasible target, so the **same target may
be matched by multiple sources** (one-to-many on the target side). Feasibility
is gated by:

- `--max-dist-m` (translation, metres; default 1.5)
- `--max-rot-deg` (geodesic quaternion rotation, degrees; default 12)
- both same-lens (L→L, R→R) **and** cross-lens (L→R, R→L) pairs are
  considered; a small `--cross-lens-penalty` (default 0.1 m-equivalent) is
  added to cross-lens cost so same-lens wins ties. Set the penalty high to
  effectively disable cross-lens matching.

Cost = `translation_m + rot_weight * rotation_deg` (plus `cross_lens_penalty`
for cross-lens pairs).

Outputs (auto-routed into `inspection_database/runs/<run_name>/`):

| File / dir            | Contents                                              |
| --------------------- | ----------------------------------------------------- |
| `pairs.json`          | Matched pair list (src/tgt ids, parity, trans, rot)   |
| `inspXvsY/`           | Merged side-by-side images (src left, tgt right)      |
| `split_pairs/source/` | Source images as `<timestamp>_<L|R>.jpg`              |
| `split_pairs/target/` | Target images as `<timestamp>_<L|R>.jpg`             |
| `trajectory.png`      | Source/target trajectories with matched links         |
| `config.json`         | Parameters and result counts for this run             |

#### Run examples

Default — match L+R pairs from inspection 1 against inspection 4 at 1.25 m
sampling with the default 1.5 m / 12 deg gate:

```bash
python match_images_by_pose.py \
    --target-inspection 4 --sample-inspection 1 \
    --sample-interval-m 1.25 --select-pair --no-show
```

Tight gate for high-quality pairs only — 0.5 m translation, 5 deg rotation,
matching inspection 1 against inspection 3:

```bash
python match_images_by_pose.py \
    --target-inspection 3 --sample-inspection 1 \
    --sample-interval-m 1.25 --select-pair \
    --max-dist-m 0.5 --max-rot-deg 5 --no-show
```

Loose gate to maximise coverage — 3.0 m translation, 20 deg rotation:

```bash
python match_images_by_pose.py \
    --target-inspection 4 --sample-inspection 1 \
    --sample-interval-m 1.25 --select-pair \
    --max-dist-m 3.0 --max-rot-deg 20 --no-show
```

Match only the LEFT lens (drop `--select-pair`, default `--sample-lens left`):

```bash
python match_images_by_pose.py \
    --target-inspection 4 --sample-inspection 1 \
    --sample-interval-m 1.25 --no-show
```

Match the RIGHT lens explicitly:

```bash
python match_images_by_pose.py \
    --target-inspection 4 --sample-inspection 1 \
    --sample-interval-m 1.25 --sample-lens right --no-show
```

Same-lens only (disable cross-lens by setting a large penalty):

```bash
python match_images_by_pose.py \
    --target-inspection 4 --sample-inspection 1 \
    --sample-interval-m 1.25 --select-pair \
    --cross-lens-penalty 1000 --no-show
```

Denser sampling — 0.5 m interval (more source viewpoints, more pairs):

```bash
python match_images_by_pose.py \
    --target-inspection 4 --sample-inspection 1 \
    --sample-interval-m 0.5 --select-pair --no-show
```

Match ALL images of an inspection as the source (no sampling; pass
`--source-inspection` instead of `--sampled-dir`/`--sample-*`):

```bash
python match_images_by_pose.py \
    --source-inspection 1 --target-inspection 4 --no-show
```

Weight rotation more heavily in the cost (default `--rot-weight 0.1`,
i.e. 1 deg ≈ 0.1 m). Raise it to prefer pairs that are rotationally closer
even if translationally further:

```bash
python match_images_by_pose.py \
    --target-inspection 4 --sample-inspection 1 \
    --sample-interval-m 1.25 --select-pair \
    --rot-weight 0.5 --no-show
```

Commit matched pairs into the `abnormal_detections` table (off by default;
existing rows are preserved unless `--no-skip-existing` is set):

```bash
python match_images_by_pose.py \
    --target-inspection 4 --sample-inspection 1 \
    --sample-interval-m 1.25 --select-pair --no-show --commit
```

Name the run explicitly (otherwise auto-named `<src>vs<tgt>_<interval>m_<dist>m_<rot>deg`,
e.g. `1vs4_1p25m_1p5m_12deg`):

```bash
python match_images_by_pose.py \
    --target-inspection 4 --sample-inspection 1 \
    --sample-interval-m 1.25 --select-pair --no-show \
    --run-name tight_insp1_vs_4
```

Show the cv2 display window of merged pairs (omit `--no-show`):

```bash
python match_images_by_pose.py \
    --target-inspection 4 --sample-inspection 1 \
    --sample-interval-m 1.25 --select-pair
```

#### Parameter reference

| Flag                  | Default | Description                                                  |
| --------------------- | ------- | ------------------------------------------------------------ |
| `--db`                | `inspection_database/complete2/inspection_v2.db` | SQLite DB path. |
| `--sampled-dir`       | `inspection_database/sampled_images` | Source `<id>.jpg` folder (ignored with `--source-inspection`). |
| `--source-inspection` | None    | Use ALL images of this inspection as the source (overrides `--sampled-dir`). |
| `--target-inspection` | 2       | Inspection id to match against.                              |
| `--sample-inspection` | 1       | Inspection to sample from when `--sample-interval-m` is set.  |
| `--sample-interval-m` | None    | Re-sample the source trajectory at this fixed arc-length interval (m). |
| `--sample-start-index`| 0       | Index (in capture order) of the first sampled viewpoint.     |
| `--sample-lens`        | `left`  | `left` / `right` / `both` — which lens of each pair to copy when sampling. Ignored when `--select-pair` is set (both lenses taken). |
| `--select-pair`       | off     | Select both L+R of each sampled viewpoint (full pair) as the source. |
| `--max-dist-m`        | 1.5     | Max translation (m) for a valid pair.                        |
| `--max-rot-deg`       | 12      | Max rotation (deg) for a valid pair.                         |
| `--rot-weight`        | 0.1     | m-per-deg rotation weight in the cost.                        |
| `--cross-lens-penalty`| 0.1     | Cost (m-equivalent) added to cross-lens pairs so same-lens wins ties. Set high to disable cross-lens. |
| `--keep-sampled`      | off     | With `--sample-interval-m`, don't clear `--sampled-dir` before writing. |
| `--run-name`          | auto    | Name of the run folder under `inspection_database/runs/`.    |
| `--no-show`           | off     | Don't open the cv2 display window of merged pairs.           |
| `--commit`            | off     | Insert matched pairs into `abnormal_detections`.             |
| `--no-skip-existing`  | off     | With `--commit`, also re-insert pairs already present.       |

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

Python 3.10+. The scripts in this folder depend on a subset of the packages
listed in [`requirements.txt`](requirements.txt). The third-party libraries
actually imported by these three scripts are:

- `numpy` — pose arrays, quaternion math, cost matrix
- `opencv-python` (`cv2`) — reading/writing/merging images for `--matched-dir`
- `matplotlib` — trajectory heatmap and matched-pair trajectory plots

(`scipy` is no longer required — the previous Hungarian-algorithm 1:1
matching was replaced by per-source nearest-candidate matching.)

Install via the backend venv, which already provides these:

```bash
pip install -r requirements.txt
# or just the subset needed here:
pip install "numpy>=2.1.3,<2.3" "opencv-python>=4.10.0" "matplotlib>=3.7"
```

Run with the backend venv so the libraries are on the path:

```bash
../backend/.venv/bin/python <script>.py ...
```

The full `requirements.txt` also pins backend web/ASR/TTS dependencies
(fastapi, uvicorn, funasr, faster-whisper, piper-tts, rerun-sdk, etc.) that
these image-selection scripts do **not** use, but are required by the rest of
the repo.

## Notes

- `_REPO_ROOT` is resolved from each script's location, so commands work from
  any working directory. Scripts in this folder use `parents[1]` (one level up
  to the repo root); the copies in `backend/scripts/` use `parents[2]`.
- Default DB is `inspection_database/complete2/inspection_v2.db`; default
  image dir is `inspection_database/complete2/outputs/images/`. Pass `--db`
  and `--image-dir` to override.
- The `cam_tf_*` columns must be per-frame camera world poses for matching to
  work. In the `complete2/` DB they are (every row has a unique `cam_tf_*`).
  In the older top-level `inspection_database/inspection_v2.db` they were a
  constant 2-value static extrinsic and matching on them would collapse — use
  `--db inspection_database/complete2/inspection_v2.db` in that case.