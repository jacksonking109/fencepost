#!/usr/bin/env python3
"""Run each Metal 4 timing case three times and validate its trace."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
RUNS = 3
# trace_check's mode, mtl4test's case, and any extra fencepost settings.
CASES = (
    ("ordered", "timing-ordered", {}),
    ("overlap", "timing-overlap", {}),
    ("render", "timing-render", {}),
    ("race", "nobarrier", {}),
    ("disabled", "timing-ordered", {"FENCEPOST_TRACE_GPU": "0"}),
    ("overflow", "timing-overflow", {}),
)


def clean_env():
    return {k: v for k, v in os.environ.items()
            if not k.startswith("FENCEPOST_") and k not in ("DYLD_INSERT_LIBRARIES", "MTL_CAPTURE_ENABLED")}


def check_case(arch, mode, case, knobs, path, env):
    settings = {"DYLD_INSERT_LIBRARIES": OUT / "libfencepost.dylib", "FENCEPOST_TRACE": path, **knobs}
    cmd = ["arch", "-" + arch, *[arg for key, value in settings.items() for arg in ("-e", f"{key}={value}")],
           OUT / "mtl4test", case]
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, (cmd, result.stdout, result.stderr)
    checked = subprocess.run([sys.executable, "-I", ROOT / "tests/trace_check.py", path, "--mtl4-timing", mode],
                           text=True, capture_output=True)
    assert checked.returncode == 0, (cmd, checked.stdout, checked.stderr)
    return checked.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch", choices=("arm64", "x86_64"))
    args = parser.parse_args()
    env = clean_env()
    with tempfile.TemporaryDirectory(prefix="fencepost-mtl4-timing-") as directory:
        for mode, case, knobs in CASES:
            for run in range(RUNS):
                trace = Path(directory) / f"{mode}-{run}.json"
                output = check_case(args.arch, mode, case, knobs, trace, env)
                if not run:
                    print(f"{args.arch} {mode} ({case}): {output}", flush=True)
    print(f"{args.arch}: 6 Metal 4 timing cases {RUNS} times each passed")


if __name__ == "__main__":
    main()
