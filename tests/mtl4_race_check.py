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
OUT = ROOT / "out"
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
}


def run(arch, program, mode, env, **knobs):
    settings = {"DYLD_INSERT_LIBRARIES": OUT / "libfencepost.dylib", **knobs}
    cmd = ["arch", "-" + arch, *[arg for key, value in settings.items() for arg in ("-e", f"{key}={value}")],
           OUT / program, mode]
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, (cmd, result.stdout, result.stderr)
    return result


def check_case(arch, directory, program, mode, env, **extra):
    path = directory / (mode + ".json")
    knobs = {"FENCEPOST_TRACE": path, **extra}
    result = run(arch, program, mode, env, **knobs)
    races = [line for line in result.stderr.splitlines() if line.startswith(("fencepost: RACE ", "fencepost: possible race "))]
    expected = int(mode in WHY)
    assert len(races) == expected, (mode, result.stderr)
    events = json.loads(path.read_text())
    summary = events[-1]["args"]
    # Pipeline reflection makes shader writes definite; FENCEPOST_MTL4_RW=possible downgrades read-write slots.
    certain = expected and "FENCEPOST_MTL4_RW" not in extra
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
        print(f"{arch} {mode} {races[0]}\n  {reason}", flush=True)
    if mode == "mutate":
        assert summary["accesses"] == 4, summary
    if mode == "placement":
        assert summary["accesses"] == 3, summary
    if mode == "load-late":
        assert "  second: " in result.stderr and "attachment load" in result.stderr, result.stderr
    if mode == "order":
        assert "Blit -> Blit is unordered" in result.stderr, result.stderr
    if mode == "partial":
        assert "Blit -> Dispatch|Blit is ordered only before Dispatch" in result.stderr, result.stderr
    subprocess.run([sys.executable, "-I", ROOT / "tests/trace_check.py", path], check=True)


def check_suite(arch, directory):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("FENCEPOST_") and k not in ("DYLD_INSERT_LIBRARIES", "MTL_CAPTURE_ENABLED")}
    for mode in ("ok", "nobarrier", "wrongstage", "event", "eventlate"):
        check_case(arch, directory, "mtl4test", mode, env)
    for mode in ("producer", "fence", "fence-missing", "commits", "commits-barrier", "color-fragment",
                 "color-vertex", "mutate", "disjoint", "overlap", "placement", "order", "partial",
                 "const-shared", "fragment-only", "fragment-vertex", "reflect-app", "compiler-race",
                 "load-early", "load-late"):
        check_case(arch, directory, "mtl4races", mode, env)
    possible = directory / "rw"
    possible.mkdir()
    check_case(arch, possible, "mtl4test", "nobarrier", env, FENCEPOST_MTL4_RW="possible")

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
    print(f"{args.arch}: 26 Metal 4 verdicts, race ring dump and clean benchmark passed")
