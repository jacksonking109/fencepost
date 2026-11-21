#!/usr/bin/env python3
"""Measure Metal 4 CPU overhead with and without fencepost: per draw, per commit and per pipeline build.

Each measurement is the median of RUNS runs of out/mtl4bench, on each requested slice.
"""

import argparse
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
RUNS = 3
MEASURES = (
    ("metal4", "CPU us/draw", r"median ([0-9.]+) p90 [0-9.]+ CPU us/draw"),
    ("metal4-commit", "CPU us/commit", r"median ([0-9.]+) p90 [0-9.]+ CPU us/commit"),
    ("pipelines", "ms/pipeline, MTL4Compiler", r"median ([0-9.]+) ms MTL4Compiler"),
    ("pipelines", "ms/pipeline, device", r", ([0-9.]+) ms device"),
)


def bench(arch, mode, injected, env):
    knobs = ["-e", f"DYLD_INSERT_LIBRARIES={OUT / 'libfencepost.dylib'}"] if injected else []
    result = subprocess.run(["arch", "-" + arch, *knobs, OUT / "mtl4bench", mode], env=env,
                            text=True, capture_output=True, timeout=120, check=True)
    if injected:
        if mode == "pipelines":
            assert "fencepost stats: 0 races, 0 possible" in result.stdout.splitlines(), (result.stdout, result.stderr)
        else:
            assert "fencepost: final: 0 race(s) (0 hits), 0 possible (0 hits)" in result.stderr, result.stderr
    return result.stdout


def samples(arch, mode, injected, env):
    if mode == "pipelines":
        bench(arch, mode, injected, env)  # warm the shader cache for this configuration
    return [bench(arch, mode, injected, env) for _ in range(RUNS)]


def trace_timing(arches, env):
    modes = ("metal4", "metal4-commit")
    settings = ("off", "estimates", "samples")
    print("| slice | measure | trace off | estimates | samples |\n|---|---|---|---|---|", flush=True)
    with tempfile.TemporaryDirectory(prefix="fencepost-mtl4-timing-") as directory:
        for arch in arches:
            for mode, unit, pattern in (m for m in MEASURES if m[0] in modes):
                timings = {setting: [] for setting in settings}
                for run in range(RUNS):
                    # Rotate the settings so none always runs with a cold GPU.
                    for setting in settings[run:] + settings[:run]:
                        knobs = dict(env)
                        trace = Path(directory) / f"{arch}-{mode}-{setting}-{run}.json"
                        if setting != "off":
                            knobs.update(FENCEPOST_TRACE=str(trace), FENCEPOST_TRACE_GPU=str(int(setting == "samples")))
                        result = bench(arch, mode, True, knobs)
                        timings[setting].append(float(re.search(pattern, result)[1]))
                        if setting != "off":
                            cmd = [sys.executable, "-I", ROOT / "tests/trace_check.py", trace]
                            if setting == "estimates":
                                cmd.append("--cpu-only")
                            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                values = " | ".join(f"{median(timings[setting]):.6f}" for setting in settings)
                print(f"| {arch} | {mode} {unit} | {values} |", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch", nargs="*", choices=("arm64", "x86_64"), default=["arm64", "x86_64"])
    parser.add_argument("--trace-timing", action="store_true", help="compare tracing off, estimates, and GPU samples")
    args = parser.parse_args()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("FENCEPOST_") and k not in ("DYLD_INSERT_LIBRARIES", "MTL_CAPTURE_ENABLED")}
    if args.trace_timing:
        trace_timing(args.arch, env)
        return
    print("Pipeline timings are measured with a warm shader cache.\n", flush=True)
    print("| slice | measure | without | with fencepost | change |\n|---|---|---|---|---|")
    for arch in args.arch:
        outputs = {(mode, injected): samples(arch, mode, injected, env)
                   for mode in dict.fromkeys(m for m, _, _ in MEASURES) for injected in (False, True)}
        for mode, unit, pattern in MEASURES:
            values = [median(float(re.search(pattern, text)[1]) for text in outputs[mode, injected])
                      for injected in (False, True)]
            print(f"| {arch} | {mode} {unit} | {values[0]:.4f} | {values[1]:.4f} | "
                  f"{100 * (values[1] / values[0] - 1):+.0f}% |", flush=True)


if __name__ == "__main__":
    main()
