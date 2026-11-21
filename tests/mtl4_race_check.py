#!/usr/bin/env python3
"""Check API-stream race verdicts independently of the GPU's racy checksum."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNS = 3   # every verdict must hold on every run
OUT = ROOT / "out"
MIXED_EARLY = "the first's command buffer had not completed when the second's was committed, and no fence or event orders them"
WHY = {
    "nobarrier": "no Metal 4 ordering covers Dispatch -> Dispatch",
    "wrongstage": "queue barrier Blit -> Dispatch does not order Dispatch -> Dispatch",
    "eventlate": "event wait matched a signal before the producer's Dispatch work",
    "fence-missing": "no Metal 4 ordering covers Blit -> Blit",
    "commits": "no Metal 4 ordering covers Blit -> Blit",
    "color-vertex": "queue barrier Vertex -> Dispatch does not order Fragment -> Dispatch",
    "overlap": "no Metal 4 ordering covers Blit -> Dispatch",
    "fragment-vertex": "queue barrier Blit -> Vertex does not order Blit -> Fragment",
    "compiler-race": "no Metal 4 ordering covers Dispatch -> Dispatch",
    "load-late": "queue barrier Fragment -> Fragment does not order Fragment -> Fragment",
    "load-fence-late": "no Metal 4 ordering covers Fragment -> Fragment",
    "load-fence-before": "no Metal 4 ordering covers Fragment -> Fragment",
    "load-fence-late-split": "no Metal 4 ordering covers Fragment -> Fragment",
    "load-producer-before": "queue barrier Fragment -> Fragment does not order Fragment -> Fragment",
    "store-depth": "no Metal 4 ordering covers Fragment -> Fragment",
    "store-resolve-same": "no Metal 4 ordering covers Fragment -> Fragment",
    "stitched-writer": "no Metal 4 ordering covers Fragment -> Fragment",
    "mixed-3to4-early": MIXED_EARLY,
    "mixed-4to3-early": MIXED_EARLY,
    "consumer-intervening": "no Metal 4 ordering covers Dispatch -> Dispatch",
    "chain-empty": "no Metal 4 ordering covers Dispatch -> Dispatch",
    "chain-before-blit": "no Metal 4 ordering covers Dispatch -> Dispatch",
    "copy-compute": "no Metal 4 ordering covers Dispatch -> Dispatch",
    "copy-compute-async": "no Metal 4 ordering covers Dispatch -> Dispatch",
    "copy-render": "no Metal 4 ordering covers Fragment -> Dispatch",
    "copy-render-async": "no Metal 4 ordering covers Fragment -> Dispatch",
    "rw-texture": "no Metal 4 ordering covers Dispatch -> Dispatch",
}


def run(arch, program, mode, env, *, injected=True, **knobs):
    settings = {"DYLD_INSERT_LIBRARIES": OUT / "libfencepost.dylib", **knobs} if injected else knobs
    cmd = ["arch", "-" + arch, *[arg for key, value in settings.items() for arg in ("-e", f"{key}={value}")],
           OUT / program, mode]
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, (cmd, result.stdout, result.stderr)
    return result


def check_case(arch, directory, program, mode, env, **extra):
    baseline = None
    if mode.startswith("copy-"):
        clean = run(arch, program, mode, env, injected=False)
        baseline, = [line for line in clean.stdout.splitlines() if line.startswith("pipeline reflection: ")]
    for run_index in range(RUNS):
        check_run(arch, directory, program, mode, env, run_index, baseline, **extra)


def check_run(arch, directory, program, mode, env, run_index, baseline, **extra):
    path = directory / f"{mode}-{run_index}.json"
    knobs = {"FENCEPOST_TRACE": path, **extra}
    result = run(arch, program, mode, env, **knobs)
    if baseline:
        assert baseline in result.stdout.splitlines(), (mode, baseline, result.stdout)
    races = [line for line in result.stderr.splitlines() if line.startswith(("fencepost: RACE ", "fencepost: possible race "))]
    expected = int(mode in WHY)
    assert len(races) == expected, (mode, result.stderr)
    events = json.loads(path.read_text())
    summary = events[-1]["args"]
    # FENCEPOST_MTL4_RW=possible downgrades only read-write buffers.
    certain = expected and not (mode == "nobarrier" and extra.get("FENCEPOST_MTL4_RW") == "possible")
    assert summary["race_hits"] == certain and summary["possible_hits"] == expected - certain, (mode, summary)
    assert summary["order_hits"] == int(mode == "order") and summary["partial_hits"] == int(mode == "partial"), summary
    assert not summary.get("mtl4_address_misses") and not summary.get("mtl4_resource_id_misses"), summary
    if expected:
        reason, = [line.strip() for line in result.stderr.splitlines() if line.startswith("  why   : ")]
        assert reason == f"why   : {WHY[mode]}", (mode, reason)
        if program == "mtl4test":
            assert "bytes [0,262144)" in races[0], races
        if mode == "overlap":
            assert "bytes [16,64)" in races[0]
        if not run_index:
            print(f"{arch} {mode} {races[0]}\n  {reason}", flush=True)
    if mode == "mutate":
        assert summary["accesses"] == 4, summary
    if mode == "placement":
        assert summary["accesses"] == 3, summary
    if mode in ("load-late", "load-fence-late", "load-fence-before", "load-fence-late-split",
                "load-producer-before"):
        assert "  second: " in result.stderr and "attachment load" in result.stderr, result.stderr
    if mode == "store-resolve-same":
        assert result.stderr.count(": MSAA resolve,") == 2, result.stderr
    if mode == "stitched-writer":
        assert 'MTL4 outside writer' in result.stderr and 'MTL4 resume' in result.stderr, result.stderr
    if mode.startswith("mixed-") and expected:
        assert "read-after-write" in races[0] and "Metal 3 " in result.stderr and "Metal 4 " in result.stderr, result.stderr
    if mode == "order":
        assert "Blit -> Blit is unordered" in result.stderr, result.stderr
    if mode == "partial":
        assert "Blit -> Dispatch|Blit is ordered only before Dispatch" in result.stderr, result.stderr
    subprocess.run([sys.executable, "-I", ROOT / "tests/trace_check.py", path], check=True,
                   stdout=subprocess.DEVNULL)


def check_suite(arch, directory):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("FENCEPOST_") and k not in ("DYLD_INSERT_LIBRARIES", "MTL_CAPTURE_ENABLED")}
    for mode in ("ok", "nobarrier", "wrongstage", "event", "eventlate"):
        check_case(arch, directory, "mtl4test", mode, env)
    for mode in ("producer", "fence", "fence-missing", "commits", "commits-barrier", "color-fragment",
                 "color-vertex", "mutate", "disjoint", "overlap", "placement", "order", "partial",
                 "const-shared", "fragment-only", "fragment-vertex", "reflect-app", "compiler-race",
                 "load-early", "load-late", "load-fence-early", "load-fence-late", "load-fence-drawless",
                 "load-fence-multiple", "load-fence-before", "load-fence-split", "load-fence-late-split",
                 "load-producer", "load-producer-before",
                 "store-depth-discard", "store-depth", "store-resolve-separate", "store-resolve-same",
                 "stitched", "stitched-load", "stitched-writer",
                 "mixed-3to4", "mixed-3to4-early", "mixed-4to3", "mixed-4to3-early",
                 "consumer-buffer", "consumer-buffers", "consumer-commits",
                 "consumer-intervening", "chain-empty", "chain-before-blit", "chain-blit", "chain-fence",
                 "copy-compute", "copy-compute-async", "copy-render", "copy-render-async", "rw-texture"):
        check_case(arch, directory, "mtl4races", mode, env)
    possible = directory / "rw"
    possible.mkdir()
    check_case(arch, possible, "mtl4test", "nobarrier", env, FENCEPOST_MTL4_RW="possible")
    check_case(arch, possible, "mtl4races", "rw-texture", env, FENCEPOST_MTL4_RW="possible")

    run(arch, "mtl4races", "fence-missing", env, FENCEPOST_TRACE=directory / "ring",
        FENCEPOST_RING=3, FENCEPOST_RING_AFTER=0, FENCEPOST_TRIGGER_FILE="")
    dump, = directory.glob("ring-*-f1-race.json")
    subprocess.run([sys.executable, "-I", ROOT / "tests/trace_check.py", dump, "--dump", "--trigger", "1", "race"],
                   check=True)
    result = run(arch, "mtl4bench", "metal4", env)
    assert "pixel: 0,255,0,255" in result.stdout
    assert "0 race(s) (0 hits), 0 possible (0 hits)" in result.stderr, result.stderr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch", choices=("arm64", "x86_64"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="fencepost-mtl4-races-") as directory:
        check_suite(args.arch, Path(directory))
    print(f"{args.arch}: 60 Metal 4 verdicts {RUNS} times each, race ring dump and clean benchmark passed")
