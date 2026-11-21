#!/usr/bin/env python3
"""Exercise GPU trace capture in one architecture without a GPU fault."""
import argparse
import os
from pathlib import Path
import platform
import re
import select
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DISABLED = "fencepost: FENCEPOST_CAPTURE needs MTL_CAPTURE_ENABLED=1 at launch; capture disabled"


def run_case(directory, arch, mode, *, enabled=True, min_free=0, frames=1, idle_secs=10, max_secs=120,
             quota=3, cooldown=0, reason=None, message=None, foreign=None, targeted=False, label_frames=1):
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
        "FENCEPOST_CAPTURE_IDLE_SECS": idle_secs,
        "FENCEPOST_CAPTURE_MAX_SECS": max_secs,
        "FENCEPOST_CAPTURE_MAX": quota,
        "FENCEPOST_CAPTURE_COOLDOWN": cooldown,
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
    drawable = mode == "mtl4-drawable"
    fixture = [ROOT / "out/mtl4frames", "--drawable", "--trigger"] if drawable else [ROOT / "out/capturetest", mode]
    with subprocess.Popen([*command, *fixture], env=env,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as test:
        try:
            if drawable or mode in {"file", "no-frame", "empty-file", "wrong-pid"}:
                assert select.select([test.stdout], [], [], 15)[0], "capturetest did not become ready"
                ready = test.stdout.readline()
                if not ready:
                    _, log = test.communicate(timeout=5)
                    raise AssertionError((test.returncode, log))
                assert ready == ("ready for trigger\n" if drawable else "ready\n"), ready
                if mode == "empty-file":
                    stamp = request.stat().st_mtime_ns + 1_000_000_000
                    request.write_text("")
                    os.utime(request, ns=(stamp, stamp))
                    test.stdin.write("\n")
                    test.stdin.flush()
                    assert select.select([test.stdout], [], [], 15)[0], "empty request was not polled"
                    assert test.stdout.readline() == "request polled\n"
                    request.write_text("File Request\n")
                    os.utime(request, ns=(stamp, stamp))
                elif targeted or mode == "wrong-pid":
                    cli = ["arch", "-" + arch, ROOT / "out/fencepost-trigger", "--capture", "--file", request]
                    pid = test.pid if targeted else 1
                    subprocess.run([*cli, "File Request", "--pid", str(pid)], env=env, check=True)
                    if mode == "wrong-pid":
                        stamp = request.stat().st_mtime_ns
                        test.stdin.write("\n")
                        test.stdin.flush()
                        assert select.select([test.stdout], [], [], 15)[0], "targeted request was not polled"
                        assert test.stdout.readline() == "request polled\n"
                        # A broadcast at the ignored PID request's mtime must also be ignored.
                        subprocess.run([*cli, "Same Stamp"], env=env, check=True)
                        os.utime(request, ns=(stamp, stamp))
                        test.stdin.write("\n")
                        test.stdin.flush()
                        assert select.select([test.stdout], [], [], 15)[0], "same-mtime request was not polled"
                        assert test.stdout.readline() == "same mtime polled\n"
                        subprocess.run([*cli, "Later Broadcast"], env=env, check=True)
                elif foreign is None:
                    stamp = request.stat().st_mtime_ns
                    request.write_text("File Request\nignored second line\n")
                    assert request.stat().st_mtime_ns > stamp
            idle_log = b""
            if mode == "idle":
                deadline = time.monotonic() + 15
                while b" (idle limit)\n" not in idle_log:
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
    frame = {"empty-file": 3, "wrong-pid": 4, "mtl4-defer": 3, "mtl4-drawable": 3}.get(mode, 2)
    expected = directory / f"capture-{test.pid}-f{frame}-{reason}.gputrace"
    assert captures == [expected] and expected.is_dir(), (captures, log)
    size = sum(path.stat().st_size for path in expected.rglob("*") if path.is_file())
    assert size >= 4096, (size, log)
    assert starts == [f"fencepost: capturing frame {frame} ({reason}) -> {expected}"], log
    suffix = {"idle": r" \(idle limit\)", "hard-limit": r" \(time limit\)"}.get(mode.removeprefix("mtl4-"), "")
    assert len(stops) == 1 and re.fullmatch(
        rf"fencepost: capture written {re.escape(str(expected))}, \d+\.\d{{2}} MB, \d+\.\d s{suffix}", stops[0]), log
    if mode == "slow-frame":
        assert "fencepost:   frames: 2;" in log, log
    if mode == "api":
        snapshots = [path for path in expected.rglob("MTLBuffer-*") if path.is_file()]
        assert len(snapshots) == 1, (snapshots, log)
        name = re.fullmatch(r"(MTLBuffer-\d+)-\d+", snapshots[0].name)
        assert name and f"fencepost: largest: {name[1]} 8 MB in 1 snapshots" in log.splitlines(), log
    if mode.startswith("mtl4-") and not drawable:
        blobs = [path.read_bytes() for path in expected.rglob("*") if path.is_file()]
        excluded = [*range(1, frame), 99]
        if mode != "mtl4-hard-limit":
            excluded.append(frame + label_frames)
        for index in [*range(frame, frame + label_frames), *excluded]:
            for part in range(2):
                label = f"fp4-capture-frame-{index}-encoder-{part}".encode()
                assert any(label in blob for blob in blobs) == (index not in excluded), (label, log)
        if mode == "mtl4-defer":
            assert log.count("fencepost: capture deferred: Metal 4 command buffers awaiting submission") == 1, log


def run_cases(directory, arch, cases, repeats):
    for run in range(repeats):
        for index, (mode, label, options) in enumerate(cases):
            case = directory / f"{run}-{index}"
            case.mkdir()
            try:
                run_case(case, arch, mode, **options)
            finally:
                documents = list(case.glob("*.gputrace"))
                if documents:
                    subprocess.run(["trash", *documents], check=True)
            print(f"{arch}: {label} passed ({run + 1}/{repeats})", flush=True)


def check_cli(directory, arch):
    request = directory / "capture.request"
    timeline = directory / "timeline.request"
    env = dict(os.environ, FENCEPOST_CAPTURE_FILE=str(request), FENCEPOST_TRIGGER_FILE=str(timeline))
    subprocess.run(["arch", "-" + arch, ROOT / "out/fencepost-trigger", "--capture", "cli"], env=env, check=True)
    assert request.read_bytes() == b"cli\n" and not timeline.exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch", choices=["arm64", "x86_64"])
    parser.add_argument("--metal", choices=["3", "4"], help="run one Metal version's cases")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        cases = [
            ("file", "file request and frame 2 trace", {"reason": "file-request"}),
            ("empty-file", "empty request ignored without acknowledging mtime", {"reason": "file-request"}),
            ("file", "matching PID request captured", {"reason": "file-request", "targeted": True}),
            ("wrong-pid", "wrong PID acknowledged and later broadcast captured", {"reason": "later-broadcast"}),
            ("file", "disabled without MTL_CAPTURE_ENABLED", {"enabled": False, "message": DISABLED}),
            ("file", "disk space refusal", {"min_free": 999999,
             "message": "fencepost: capture refused: insufficient free space on output volume"}),
            ("no-frame", "no capture without a frame boundary", {}),
            ("api", "capture API and active-request suppression", {"reason": "api-request"}),
            ("multi-frame", "capture spans two frame ends", {"reason": "api-request", "frames": 2}),
            ("slow-frame", "slow frame finishes while submissions continue",
             {"reason": "api-request", "idle_secs": 2}),
            ("idle", "idle limit stops capture after submissions stop",
             {"reason": "api-request", "frames": 2, "idle_secs": 1}),
            ("hard-limit", "hard cap stops capture while frames continue",
             {"reason": "api-request", "frames": 1000, "idle_secs": 2, "max_secs": 3}),
            ("exit-active", "exit during a multi-frame capture", {"reason": "api-request", "frames": 3}),
            ("app-capture", "app-owned capture left running", {
                "message": "fencepost: capture refused: Metal capture already running"}),
        ]
        foreign = Path("/etc/hosts")
        if foreign.is_file() and foreign.stat().st_uid != os.geteuid() and os.access(foreign, os.R_OK):
            cases.append(("file", "foreign-owned request ignored and logged once", {"foreign": foreign}))
        else:
            print(f"{args.arch}: skipped foreign-owned request case (no readable foreign fixture)", flush=True)
        root = Path(scratch).resolve()
        if args.metal != "4":
            run_cases(root, args.arch, cases, 1)
            check_cli(root, args.arch)
        if args.metal != "3":
            if int(platform.mac_ver()[0].split(".")[0]) < 26:
                if args.metal == "4":
                    parser.error("Metal 4 capture checks require macOS 26 or later")
                print(f"{args.arch}: skipped Metal 4 capture checks (requires macOS 26 or later)")
            else:
                metal4 = root / "metal4"
                metal4.mkdir()
                api = {"reason": "api-request"}
                cases = [
                    ("mtl4-api", "Metal 4 next-frame encoder labels", api),
                    ("mtl4-drawable", "Metal 4 file trigger after signalDrawable:", {"reason": "file-request"}),
                    ("mtl4-defer", "Metal 4 defers across an ended buffer on another thread and queue", api),
                    ("mtl4-blocked", "Metal 4 gives up after three blocked frame starts", {
                        "message": "fencepost: capture refused: Metal 4 command buffers still awaiting submission after 3 frame starts"}),
                    ("mtl4-mixed", "mixed process starts at Metal 4 begin", api),
                    ("mtl4-quota", "Metal 4 quota refusal", dict(api, quota=1,
                        message="fencepost: capture refused: capture limit reached")),
                    ("mtl4-cooldown", "Metal 4 cooldown refusal", dict(api, cooldown=600,
                        message="fencepost: capture refused: cooldown in effect")),
                    ("mtl4-space", "Metal 4 disk space refusal", {"min_free": 999999,
                        "message": "fencepost: capture refused: insufficient free space on output volume"}),
                    ("mtl4-multi", "Metal 4 two-frame encoder labels", dict(api, frames=2, label_frames=2)),
                    ("mtl4-idle", "Metal 4 idle watchdog", dict(api, frames=2, idle_secs=1)),
                    ("mtl4-hard-limit", "Metal 4 hard cap with continuing submissions",
                     dict(api, frames=1000, idle_secs=2, max_secs=3)),
                ]
                run_cases(metal4, args.arch, cases, 3)
    print(f"{args.arch}: capture checks passed")


if __name__ == "__main__":
    main()
