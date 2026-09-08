#!/usr/bin/env python3
"""Smoke-test SocialGen trajectory helpers in internvla_n1_lerobot_dataset.py.

Does not load the full dataset module (avoids torch / transformers).
Does not need traj_data/socialnav — uses synthetic poses only.

Run:
  cd ~/SocialNav/InternNav
  python3 scripts/test_socialgen_trajectory_resample.py
"""

from __future__ import annotations

import ast
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "internnav" / "dataset" / "internvla_n1_lerobot_dataset.py"


def _load_helpers():
    """Parse the dataset file and exec only the pure-numpy helper defs."""
    tree = ast.parse(SRC.read_text())
    wanted = {
        "interpolate_and_resample_trajectory",
        "subsample_trajectory_arclength_capped",
        "xy_to_delta_xyt",
        "clip_or_pad",
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


def main() -> int:
    ns = _load_helpers()
    interpolate = ns["interpolate_and_resample_trajectory"]
    subsample = ns["subsample_trajectory_arclength_capped"]
    fails: list[str] = []

    # 1) Tiny per-frame motion that old steps_sq > 0.05 mask would wipe out.
    t = np.cumsum(np.ones((200, 3)) * 0.01, axis=0)
    t[:, 2] = 0.0
    xy, delta = interpolate(t, predict_step_num=32)
    if xy.shape != (33, 2):
        fails.append(f"xy shape {xy.shape} != (33, 2)")
    if delta.shape != (32, 3):
        fails.append(f"delta shape {delta.shape} != (32, 3)")
    if not np.isfinite(xy).all() or not np.isfinite(delta).all():
        fails.append("non-finite values in outputs")
    # 200 * 0.01 = 2 m < 3.3 m → end should stay near full path (~2 m)
    if not (1.5 < xy[-1, 0] < 2.2):
        fails.append(f"short-traj end expected ~2 m, got {xy[-1]}")

    # 2) Long path: must cap near 3.3 m (not stretch to 10 m).
    pts = np.stack([np.linspace(0, 10, 500), np.zeros(500)], axis=1)
    out = subsample(pts, sample_length=33, max_distance=3.3)
    if out.shape != (33, 2):
        fails.append(f"long-path shape {out.shape} != (33, 2)")
    if not np.allclose(out[0], [0.0, 0.0], atol=1e-6):
        fails.append(f"long-path start {out[0]} != origin")
    if out[-1, 0] > 3.35 or out[-1, 0] < 3.0:
        fails.append(f"long-path end expected ~3.3 m, got {out[-1]}")
    if np.allclose(out[-1], [10.0, 0.0], atol=0.1):
        fails.append("long-path still ends at 10 m (cap missing)")
    # Sampled points must be real logged poses (subset of pts rows).
    for p in out:
        if not np.any(np.all(np.isclose(pts, p, atol=1e-8), axis=1)):
            fails.append(f"sampled point {p} is not a real logged pose")
            break

    # 3) Very short path: still returns sample_length points; last is path end.
    short = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out2 = subsample(short, sample_length=33, max_distance=3.3)
    if out2.shape != (33, 2) or not np.allclose(out2[-1], [2.0, 0.0]):
        fails.append(f"short-path resample failed: shape={out2.shape}, end={out2[-1]}")

    # 4) Source-level guards for SocialGen rules.
    src = SRC.read_text()
    for line in src.splitlines():
        if "mask = steps_sq > 0.05" in line and not line.lstrip().startswith("#"):
            fails.append("steps_sq mask still active")
            break
    if "def smooth_and_resample_trajectory" in src:
        fails.append("old smooth_and_resample_trajectory still present")
    if "def subsample_trajectory_arclength_capped" not in src:
        fails.append("subsample_trajectory_arclength_capped missing")
    if "max_distance=3.3" not in src:
        fails.append("3.3 m max_distance not wired in dataset helpers")

    if fails:
        print("FAIL:")
        for f in fails:
            print(" -", f)
        return 1

    print("PASS: SocialGen trajectory helpers look correct")
    print(f"  file: {SRC}")
    print(f"  xy shape={xy.shape}, delta shape={delta.shape}")
    print(f"  long-path capped end x={out[-1, 0]:.3f} m (target ~3.3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
