#!/usr/bin/env python3
"""Exercise ring triggers, then check every standalone dump and off-mode stream."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run_case(directory, arch, mode, ring):
    prefix = directory / f"{mode}-{ring}"
    trigger_file = prefix.with_suffix(".trigger")
    trigger_file.write_text("stale\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("FENCEPOST_")}
    knobs = {
        "DYLD_INSERT_LIBRARIES": ROOT / "out/libfencepost.dylib",
        "FENCEPOST_TRIGGER_FILE": trigger_file,
        "FENCEPOST_TRACE": prefix,
        "FENCEPOST_TRIGGER_COOLDOWN": 3 if mode == "file" else 0,
    }
    if ring:
        knobs["FENCEPOST_RING"] = 10
    if mode == "race":
        knobs.update(FENCEPOST_TRIGGER_RACE="all", FENCEPOST_MAX_AUTO_DUMPS=1, FENCEPOST_MAX_MANUAL_DUMPS=1)
    cmd = ["arch", "-" + arch]
    for k, v in knobs.items():
        cmd.extend(["-e", f"{k}={v}"])
    subprocess.run([*cmd, ROOT / "out/ringtest", mode], env=env, check=True)
    dumps = sorted(directory.glob(prefix.name + "-*.json"))
    checker = [sys.executable, ROOT / "tests/trace_check.py"]
    if not ring:
        assert not dumps
        subprocess.run([*checker, prefix], check=True)
        return
    assert not prefix.exists()
    expected = {
        "api": [(30, "bad-frame", "api")],
        "race": [(30, "race", "race"), (36, "checkpoint", "api")],
        "spike": [(131, "spike", "spike")],
        "file": [(30, "flicker", "file"), (36, "checkpoint", "api")],
    }[mode]
    assert len(dumps) == len(expected), dumps
    for path, (frame, reason, source) in zip(dumps, expected):
        subprocess.run([*checker, path, "--dump", "--trigger", str(frame), reason,
                        "--frames", f"{frame - 9}:{frame + 2}"], check=True)
        events = json.loads(path.read_text())
        encoders = [e for e in events if e.get("cat") in {"blit", "render", "compute"}]
        assert Counter(e["args"]["frame"] for e in encoders) == {
            i: 1 + (mode == "race" and i in (30, 35)) for i in range(frame - 9, frame + 3)
        }
        assert {e["args"]["uid"] for e in encoders} == {
            e["args"]["uid"] for e in events if e.get("cat") == "cpu encode"
        }
        trigger = next(e for e in events if e["name"] == "TRIGGER " + reason)
        assert trigger["args"]["source"] == source
        if mode == "race":
            assert any(e["name"].startswith("RACE ") for e in events)
        if reason == "checkpoint":
            assert events[-1]["args"]["suppressed_triggers"] == 1
    assert trigger_file.read_text() == ("flicker\n" if mode == "file" else "stale\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch", choices=["arm64", "x86_64"])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        for case in ["api", "race", "spike", "file"]:
            for enabled in [True, False]:
                run_case(Path(scratch), args.arch, case, enabled)
    print(f"{args.arch}: all four trigger scenarios passed with ring on and off")
