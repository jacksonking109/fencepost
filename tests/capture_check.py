#!/usr/bin/env python3
"""Exercise GPU trace capture in one architecture without a GPU fault."""
import argparse
import os
from pathlib import Path
import re
import select
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DISABLED = "fencepost: FENCEPOST_CAPTURE needs MTL_CAPTURE_ENABLED=1 at launch; capture disabled"


def run_case(directory, arch, mode, *, enabled=True, min_free=0, frames=1, max_secs=10,
             reason=None, message=None, foreign=None):
    prefix = directory / "capture"
    request = directory / "request"
    if foreign is None:
        request.write_text("stale\n")
    else:
        request.symlink_to(foreign)
        message = f"fencepost: ignoring request file owned by another user: {request}"
    knobs = {
        "DYLD_INSERT_LIBRARIES": ROOT / "out/libfencepost.dylib",
        "FENCEPOST_CAPTURE": prefix,
        "FENCEPOST_CAPTURE_FILE": request,
        "FENCEPOST_CAPTURE_MIN_FREE_GB": min_free,
        "FENCEPOST_CAPTURE_FRAMES": frames,
        "FENCEPOST_CAPTURE_MAX_SECS": max_secs,
        "FENCEPOST_CAPTURE_COOLDOWN": 0,
    }
    if enabled:
        knobs["MTL_CAPTURE_ENABLED"] = 1
    app_trace = directory / "application.gputrace"
    if mode == "app-capture":
        knobs["CAPTURETEST_APP_TRACE"] = app_trace
    env = {key: value for key, value in os.environ.items()
           if key != "MTL_CAPTURE_ENABLED" and not key.startswith(("FENCEPOST_", "DYLD_"))}
    command = ["arch", "-" + arch]
    for key, value in knobs.items():
        command.extend(["-e", f"{key}={value}"])
    with subprocess.Popen([*command, ROOT / "out/capturetest", mode], env=env,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as test:
        try:
            if mode in {"file", "no-frame"}:
                assert select.select([test.stdout], [], [], 15)[0], "capturetest did not become ready"
                ready = test.stdout.readline()
                if not ready:
                    _, log = test.communicate(timeout=5)
                    raise AssertionError((test.returncode, log))
                assert ready == "ready\n", ready
                if foreign is None:
                    stamp = request.stat().st_mtime_ns
                    request.write_text("File Request\nignored second line\n")
                    assert request.stat().st_mtime_ns > stamp
            idle_log = b""
            if mode == "idle":
                deadline = time.monotonic() + 15
                while b" (time limit)\n" not in idle_log:
                    timeout = max(0, deadline - time.monotonic())
                    assert select.select([test.stderr], [], [], timeout)[0], idle_log
                    chunk = os.read(test.stderr.fileno(), 4096)
                    assert chunk, idle_log
                    idle_log += chunk
            _, log = test.communicate(input="\n", timeout=30)
            log = idle_log.decode() + log
            assert test.returncode == 0, (test.returncode, log)
        finally:
            if test.poll() is None:
                test.kill()
                test.wait()

    captures = list(directory.glob("capture-*.gputrace"))
    if mode == "app-capture":
        assert app_trace.is_dir(), log
    starts = [line for line in log.splitlines() if line.startswith("fencepost: capturing frame ")]
    stops = [line for line in log.splitlines() if line.startswith("fencepost: capture written ")]
    if message:
        assert log.splitlines().count(message) == 1, log
    if reason is None:
        assert not captures and not starts and not stops, (captures, log)
        return
    expected = directory / f"capture-{test.pid}-f2-{reason}.gputrace"
    assert captures == [expected] and expected.is_dir(), (captures, log)
    size = sum(path.stat().st_size for path in expected.rglob("*") if path.is_file())
    assert size >= 4096, (size, log)
    assert starts == [f"fencepost: capturing frame 2 ({reason}) -> {expected}"], log
    suffix = r" \(time limit\)" if mode == "idle" else ""
    assert len(stops) == 1 and re.fullmatch(
        rf"fencepost: capture written {re.escape(str(expected))}, \d+\.\d{{2}} MB, \d+\.\d s{suffix}", stops[0]), log


def check_cli(directory, arch):
    request = directory / "capture.request"
    timeline = directory / "timeline.request"
    env = dict(os.environ, FENCEPOST_CAPTURE_FILE=str(request), FENCEPOST_TRIGGER_FILE=str(timeline))
    subprocess.run(["arch", "-" + arch, ROOT / "out/fencepost-trigger", "--capture", "cli"], env=env, check=True)
    assert request.read_bytes() == b"cli\n" and not timeline.exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch", choices=["arm64", "x86_64"])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        cases = [
            ("file", "file request and frame 2 trace", {"reason": "file-request"}),
            ("file", "disabled without MTL_CAPTURE_ENABLED", {"enabled": False, "message": DISABLED}),
            ("file", "disk space refusal", {"min_free": 999999,
             "message": "fencepost: capture refused: insufficient free space on output volume"}),
            ("no-frame", "no capture without a frame boundary", {}),
            ("api", "capture API and active-request suppression", {"reason": "api-request"}),
            ("multi-frame", "capture spans two frame ends", {"reason": "api-request", "frames": 2}),
            ("idle", "capture time limit while submissions are idle",
             {"reason": "api-request", "frames": 2, "max_secs": 1}),
            ("exit-active", "exit during a multi-frame capture", {"reason": "api-request", "frames": 3}),
            ("app-capture", "app-owned capture left running", {
                "message": "fencepost: capture refused: Metal capture already running"}),
        ]
        foreign = Path("/etc/hosts")
        if foreign.is_file() and foreign.stat().st_uid != os.geteuid() and os.access(foreign, os.R_OK):
            cases.append(("file", "foreign-owned request ignored and logged once", {"foreign": foreign}))
        else:
            print(f"{args.arch}: skipped foreign-owned request case (no readable foreign fixture)", flush=True)
        for index, (mode, label, options) in enumerate(cases):
            directory = Path(scratch).resolve() / str(index)
            directory.mkdir()
            run_case(directory, args.arch, mode, **options)
            print(f"{args.arch}: {label} passed", flush=True)
        check_cli(Path(scratch), args.arch)
    print(f"{args.arch}: capture checks passed (including CLI)")


if __name__ == "__main__":
    main()
