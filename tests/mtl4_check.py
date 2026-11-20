#!/usr/bin/env python3
"""Check Metal 4 frame boundaries, commit feedback and a CLI-triggered ring dump."""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FEEDBACK = "app_feedback fresh=1 reused=1 added=1 batch=6 errors=0"
CAPTURE_FALLBACK = "fencepost: capture not yet supported for Metal 4"


def command(arch, knobs, *flags):
    return ["arch", "-" + arch, *[arg for k, v in knobs.items() for arg in ("-e", f"{k}={v}")],
            OUT / "mtl4frames", *flags]


def check_trace(path, frames, *, dump=False):
    cmd = [sys.executable, "-I", ROOT / "tests/trace_check.py", path]
    if dump:
        cmd += ["--dump", "--trigger", "3", "mtl4-cli", "--frames", "1:5"]
    subprocess.run(cmd, check=True)
    events = json.loads(path.read_text(), parse_float=Decimal)
    tracks = {e["tid"]: e["args"]["name"] for e in events if e["name"] == "thread_name"}
    commits = {e["args"]["uid"]: e for e in events if e.get("cat") == "command buffer"}
    encoders = [e for e in events if e.get("cat") in ("compute", "render", "blit")]
    frame_slices = [e["args"] for e in events if e.get("cat") == "frame" and e["ph"] == "X"]
    assert sorted(f["frame"] for f in frame_slices) == frames
    assert all(f["encoders"] == 3 for f in frame_slices)
    assert len(commits) == len(frames) * 2
    assert all(c["dur"] > 0 and c["name"].startswith("MTL4 commit") for c in commits.values())
    assert all("MTL4" in tracks[c["tid"]] for c in commits.values())
    assert Counter(e["cat"] for e in encoders) == {"compute": len(frames), "render": len(frames) * 2}
    assert all(e["args"]["estimated"] and e["name"].startswith("MTL4 ") for e in encoders)
    by_commit = defaultdict(list)
    for e in encoders:
        by_commit[e["args"]["cb"]].append(e)
    for encs in by_commit.values():
        encs.sort(key=lambda e: e["ts"])
        assert [e["args"]["cb_label"] for e in encs] in (["producer"], ["batch first", "batch second"])
        assert len({e["args"]["logical_cb"] for e in encs}) == len(encs)
    starts = {e["id"]: e for e in events if e["ph"] == "s"}
    finishes = {e["id"]: e for e in events if e["ph"] == "f"}
    assert all(start["ts"] <= finishes[i]["ts"] for i, start in starts.items())
    assert sum(e["cat"] == "event" for e in starts.values()) == len(frames) * 2
    summary = events[-1]["args"]
    assert summary["markers"] == summary["race_hits"] == summary["order_hits"] == summary["dropped"] == 0
    if dump:
        trigger, = (e for e in events if e["name"].startswith("TRIGGER "))
        assert trigger["args"]["source"] == "file"
    print(f"{len(frames)} frames, {len(commits)} real GPU commit spans, {len(frames) * 2} event flows", flush=True)


def run(arch, directory):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("FENCEPOST_") and k not in ("DYLD_INSERT_LIBRARIES", "MTL_CAPTURE_ENABLED")}
    injection = {"DYLD_INSERT_LIBRARIES": OUT / "libfencepost.dylib"}
    for mode in ("api", "drawable"):
        flags = ("--drawable",) if mode == "drawable" else ()
        path = directory / f"{mode}.json"
        baseline = subprocess.check_output(command(arch, {}, *flags), env=env, text=True, timeout=20)
        injected = subprocess.check_output(command(arch, dict(injection, FENCEPOST_TRACE=path), *flags),
                                           env=env, text=True, timeout=20)
        assert injected == baseline == f"frames=6 boundary={mode} {FEEDBACK}\n", (baseline, injected)
        print(injected.strip(), flush=True)
        check_trace(path, list(range(1, 7)))

    trigger_file = directory / "mtl4.trigger"
    prefix = directory / "ring"
    knobs = dict(injection, FENCEPOST_TRACE=prefix, FENCEPOST_RING=3, FENCEPOST_RING_AFTER=2,
                 FENCEPOST_TRIGGER_FILE=trigger_file, MTL_CAPTURE_ENABLED=1,
                 FENCEPOST_CAPTURE=directory / "capture", FENCEPOST_CAPTURE_FILE=directory / "capture.trigger")
    with subprocess.Popen(command(arch, knobs, "--trigger"), env=env,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as app:
        try:
            assert select.select([app.stdout], [], [], 10)[0], "Metal 4 trigger fixture did not become ready"
            assert app.stdout.readline() == "ready for trigger\n"
            subprocess.run(["arch", "-" + arch, OUT / "fencepost-trigger", "mtl4-cli",
                            "--file", trigger_file, "--pid", str(app.pid)], env=env, check=True, timeout=10)
            time.sleep(0.3)  # Let the 250 ms file-poll interval elapse before frame 3 submits.
            output, log = app.communicate("\n", timeout=15)
            assert app.returncode == 0 and output == f"frames=6 boundary=api {FEEDBACK}\n", (output, log)
            assert log.splitlines().count(CAPTURE_FALLBACK) == 1, log
        finally:
            if app.poll() is None:
                app.kill()
    dumps = list(directory.glob("ring-*.json"))
    assert len(dumps) == 1, (dumps, log)
    dump = dumps[0]
    assert not list(directory.glob("*.gputrace"))
    check_trace(dump, list(range(1, 6)), dump=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch", choices=["arm64", "x86_64"])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="fencepost-mtl4-") as directory:
        run(args.arch, Path(directory))
    print(f"{args.arch}: Metal 4 API, drawable and CLI ring checks passed")
