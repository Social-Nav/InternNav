#!/usr/bin/env python3
"""Inspect SocialGen trajectory sampling on real traj_data/socialnav episodes.

Loads logged camera extrinsics from parquet, builds relative xy like training,
runs subsample_trajectory_arclength_capped (3.3 m), prints stats, saves a plot.

Does not need torch training — only numpy (+ pandas/pyarrow for parquet, matplotlib for plot).

Examples:
  cd ~/SocialNav/InternNav
  conda activate internnav
  python scripts/inspect_socialgen_trajectory_sample.py
  python scripts/inspect_socialgen_trajectory_sample.py --scene grscenes_1 --episode 4 --out /tmp/traj.png
"""

from __future__ import annotations

import argparse
import ast
import glob
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "internnav" / "dataset" / "internvla_n1_lerobot_dataset.py"
DEFAULT_DATA = ROOT / "traj_data" / "socialnav"
POSE_KEY = "pose.132cm_30deg"


def _load_helpers():
    tree = ast.parse(SRC.read_text())
    wanted = {
        "subsample_trajectory_arclength_capped",
        "get_trajectory_relative_to_frame",
        "interpolate_and_resample_trajectory",
        "xy_to_delta_xyt",
    }
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    missing = wanted - {n.name for n in body}
    if missing:
        raise RuntimeError(f"Missing helpers in {SRC}: {sorted(missing)}")
    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict = {"np": np}
    exec(compile(mod, str(SRC), "exec"), ns)  # noqa: S102
    return ns


def _load_extrinsics(parquet_path: pathlib.Path) -> np.ndarray:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if POSE_KEY not in df.columns:
        raise KeyError(f"{POSE_KEY} not in {parquet_path}; cols={list(df.columns)}")
    # Same as internvla_n1_lerobot_dataset.get_annotations_from_lerobot_data:
    # parquet cells are object-arrays-of-rows; .tolist() yields nested lists → (N,4,4).
    poses = df[POSE_KEY].apply(lambda x: x.tolist()).tolist()
    return np.asarray(poses, dtype=np.float64)


def _arc_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.sqrt(((np.diff(xy, axis=0)) ** 2).sum(axis=1)).sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA)
    parser.add_argument("--scene", default="grscenes_1")
    parser.add_argument("--episode", type=int, default=None, help="episode_index; default=first parquet found")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "scripts" / "socialgen_traj_inspect.png")
    parser.add_argument("--max-distance", type=float, default=3.3)
    parser.add_argument("--sample-length", type=int, default=33)
    args = parser.parse_args()

    if not args.data_root.is_dir():
        print(f"ERROR: missing data root {args.data_root}", file=sys.stderr)
        print("Expected symlink: traj_data/socialnav -> .../traj_data/grscenes", file=sys.stderr)
        return 1

    scene_dir = args.data_root / args.scene
    if args.episode is None:
        files = sorted(glob.glob(str(scene_dir / "data" / "chunk-*" / "episode_*.parquet")))
        if not files:
            print(f"ERROR: no parquet under {scene_dir}", file=sys.stderr)
            return 1
        parquet_path = pathlib.Path(files[0])
    else:
        parquet_path = scene_dir / "data" / f"chunk-{args.episode // 1000:03d}" / f"episode_{args.episode:06d}.parquet"
        if not parquet_path.is_file():
            print(f"ERROR: missing {parquet_path}", file=sys.stderr)
            return 1

    ns = _load_helpers()
    get_rel = ns["get_trajectory_relative_to_frame"]
    subsample = ns["subsample_trajectory_arclength_capped"]
    interpolate = ns["interpolate_and_resample_trajectory"]

    extrinsics = _load_extrinsics(parquet_path)
    # Match training: pitch_2=30 for socialnav_132cm_30_30
    rel = get_rel(extrinsics, camera_deg=30)
    xy_full = rel[:, :2].copy()
    # Training prepends origin inside interpolate_and_resample_trajectory
    xy_for_label = np.concatenate([np.zeros((1, 2)), xy_full[1:]], axis=0)

    sampled = subsample(xy_for_label, sample_length=args.sample_length, max_distance=args.max_distance)
    _, delta = interpolate(rel, predict_step_num=args.sample_length - 1)

    L_full = _arc_length(xy_for_label)
    L_samp = _arc_length(sampled)

    print(f"scene={args.scene}  file={parquet_path}")
    print(f"frames={len(extrinsics)}  full_path_len={L_full:.3f} m")
    print(f"sampled_points={len(sampled)}  sampled_path_len={L_samp:.3f} m  (cap={args.max_distance} m)")
    print(f"delta_shape={delta.shape}  start={sampled[0]}  end={sampled[-1]}")
    if L_full > args.max_distance + 0.05:
        ok = L_samp <= args.max_distance + 0.15
        print(f"cap_check: {'OK' if ok else 'FAIL'} (long episode should stay near {args.max_distance} m)")
    else:
        print("cap_check: N/A (episode shorter than cap; sampled length ≈ full path)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (stats above still valid)")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xy_for_label[:, 0], xy_for_label[:, 1], "-", color="0.7", lw=1.5, label=f"raw ({L_full:.2f} m)")
    ax.plot(sampled[:, 0], sampled[:, 1], "o-", color="C0", ms=4, label=f"sampled ({L_samp:.2f} m)")
    ax.scatter([sampled[0, 0]], [sampled[0, 1]], c="green", s=60, zorder=3, label="start")
    ax.scatter([sampled[-1, 0]], [sampled[-1, 1]], c="red", s=60, zorder=3, label="end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{args.scene} / {parquet_path.name}\n3.3 m arclength real-pose subsample")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
