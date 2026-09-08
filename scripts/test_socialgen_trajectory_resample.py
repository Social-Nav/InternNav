#!/usr/bin/env python3
"""Smoke-test SocialGen trajectory helpers in internvla_n1_lerobot_dataset.py.

Does not load the full dataset module (avoids torch / transformers).
Run:
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
        "subsample_trajectory_uniform",
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
    subsample = ns["subsample_trajectory_uniform"]
    fails: list[str] = []

    # Tiny per-frame motion that old steps_sq > 0.05 mask would wipe out.
    t = np.cumsum(np.ones((200, 3)) * 0.01, axis=0)
    t[:, 2] = 0.0
    xy, delta = interpolate(t, predict_step_num=32)
    if xy.shape != (33, 2):
        fails.append(f"xy shape {xy.shape} != (33, 2)")
    if delta.shape != (32, 3):
        fails.append(f"delta shape {delta.shape} != (32, 3)")
    if not np.isfinite(xy).all() or not np.isfinite(delta).all():
        fails.append("non-finite values in outputs")

    pts = np.stack([np.linspace(0, 10, 100), np.zeros(100)], axis=1)
    out = subsample(pts, 33)
    if not np.allclose(out[0], [0.0, 0.0]) or not np.allclose(out[-1], [10.0, 0.0]):
        fails.append(f"bad endpoints: {out[0]}, {out[-1]}")

    short = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out2 = subsample(short, 33)
    if out2.shape != (33, 2) or not np.allclose(out2[-1], [2.0, 0.0]):
        fails.append("short-path pad failed")

    src = SRC.read_text()
    for line in src.splitlines():
        if "mask = steps_sq > 0.05" in line and not line.lstrip().startswith("#"):
            fails.append("steps_sq mask still active")
            break
    if "def smooth_and_resample_trajectory" in src:
        fails.append("old smooth_and_resample_trajectory still present")
    if "def subsample_trajectory_uniform" not in src:
        fails.append("subsample_trajectory_uniform missing")

    if fails:
        print("FAIL:")
        for f in fails:
            print(" -", f)
        return 1

    print("PASS: SocialGen trajectory helpers look correct")
    print(f"  file: {SRC}")
    print(f"  xy shape={xy.shape}, delta shape={delta.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
