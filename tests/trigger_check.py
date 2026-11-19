#!/usr/bin/env python3
"""Check the trigger CLI, a standalone ring dump, and the menu bar app's startup."""
import argparse
import json
import os
import platform
from pathlib import Path
import plistlib
import select
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = "io.github.jacksonking109.fencepost-trigger"


def check_cli(directory, cli, env):
    path = directory / "cli.trigger"
    explicit = directory / "explicit.trigger"
    cli_env = dict(env, FENCEPOST_TRIGGER_FILE=str(path))
    subprocess.run(cli, env=cli_env, check=True)
    assert path.read_bytes() == b"manual\n"
    first = path.stat().st_mtime_ns
    subprocess.run([*cli, "second"], env=cli_env, check=True)
    assert path.read_bytes() == b"second\n"
    assert path.stat().st_mtime_ns > first

    # Force a same-second advance even if the two processes above straddle a second.
    second = 1_000_000_000
    future = (time.time_ns() // second + 60) * second
    os.utime(path, ns=(future, future))
    subprocess.run([*cli, "same-second"], env=cli_env, check=True)
    stamp = path.stat().st_mtime_ns
    assert future < stamp < future + second, stamp

    subprocess.run([*cli, "explicit", "--file", explicit], env=cli_env, check=True)
    assert explicit.read_bytes() == b"explicit\n"
    assert path.read_bytes() == b"same-second\n"
    unwritable = directory / "unwritable"
    unwritable.mkdir(mode=0o555)
    try:
        # Root can write through mode bits but cannot replace a directory with a file.
        destination = unwritable if os.geteuid() == 0 else unwritable / "trigger"
        failed = subprocess.run([*cli, "--file", destination], env=cli_env, capture_output=True)
        assert failed.returncode != 0 and b"fencepost-trigger:" in failed.stderr, failed
        assert not list(directory.glob(".fencepost-trigger.*"))
    finally:
        unwritable.chmod(0o755)


def check_ring(directory, arch, cli, env):
    path = directory / "ring.trigger"
    prefix = directory / "external"
    knobs = {
        "DYLD_INSERT_LIBRARIES": ROOT / "out/libfencepost.dylib",
        "FENCEPOST_RING": 10,
        "FENCEPOST_TRACE": prefix,
        "FENCEPOST_TRIGGER_FILE": path,
    }
    command = ["arch", "-" + arch]
    for key, value in knobs.items():
        command.extend(["-e", f"{key}={value}"])
    with subprocess.Popen([*command, ROOT / "out/ringtest", "external"], env=env,
                          stdout=subprocess.PIPE, text=True) as ring:
        try:
            assert select.select([ring.stdout], [], [], 10)[0], "ringtest did not become ready"
            assert ring.stdout.readline() == "ready\n"
            assert not path.exists()
            subprocess.run([*cli, "external-cli", "--file", path], env=env, check=True)
            assert ring.wait(timeout=12) == 0
        finally:
            if ring.poll() is None:
                ring.kill()

    dumps = list(directory.glob("external-*.json"))
    assert len(dumps) == 1, dumps
    dump = dumps[0]
    subprocess.run([sys.executable, "-I", ROOT / "tests/trace_check.py", "--dump", dump], check=True)
    trigger, = (event for event in json.loads(dump.read_text()) if event["name"].startswith("TRIGGER "))
    assert (trigger["args"]["reason"], trigger["args"]["source"]) == ("external-cli", "file")
    assert path.read_bytes() == b"external-cli\n"


def read_preferences():
    return plistlib.loads(subprocess.check_output(["defaults", "export", DOMAIN, "-"]))


def smoke_pid(executable, token):
    args = subprocess.check_output(["ps", "-ww", "-ax", "-o", "pid=,command="], text=True)
    for line in args.splitlines():
        fields = line.split(None, 1)
        if len(fields) == 2 and fields[1].startswith(f"{executable} ") and token in fields[1]:
            return int(fields[0])
    return None


def check_app(directory, arch):
    app = ROOT / "out/Fencepost Trigger.app"
    executable = app / "Contents/MacOS/Fencepost Trigger"
    path = str(directory / "app.trigger")
    token = f"-FencepostSmokeTest {path}"
    original = read_preferences()
    subprocess.run(["defaults", "write", DOMAIN, "triggerFile", "-string", path], check=True)
    try:
        subprocess.run(["open", "-n", "-g", "--arch", arch, "-a", app,
                        "--args", "-FencepostSmokeTest", path], check=True, timeout=10)
        deadline = time.monotonic() + 5
        while smoke_pid(executable, token) is None and time.monotonic() < deadline:
            time.sleep(0.1)
        time.sleep(0.5)
        assert smoke_pid(executable, token) is not None, "Fencepost Trigger did not stay running"
    finally:
        try:
            pid = smoke_pid(executable, token)
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 5
                while smoke_pid(executable, token) is not None and time.monotonic() < deadline:
                    time.sleep(0.1)
                assert smoke_pid(executable, token) is None, "Fencepost Trigger did not quit"
        finally:
            if "triggerFile" in original:
                current = read_preferences()
                current["triggerFile"] = original["triggerFile"]
                subprocess.run(["defaults", "import", DOMAIN, "-"], input=plistlib.dumps(current), check=True)
            else:
                subprocess.run(["defaults", "delete", DOMAIN, "triggerFile"], check=True)
    assert read_preferences().get("triggerFile") == original.get("triggerFile")
    assert not Path(path).exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch", choices=["arm64", "x86_64"])
    args = parser.parse_args()
    cli = ["arch", "-" + args.arch, ROOT / "out/fencepost-trigger"]
    env = {key: value for key, value in os.environ.items() if not key.startswith(("FENCEPOST_", "DYLD_"))}
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        check_cli(directory, cli, env)
        print(f"{args.arch}: CLI content, mtime, overrides and write failure passed", flush=True)
        check_ring(directory, args.arch, cli, env)
        print(f"{args.arch}: exactly one external-cli ring dump passed", flush=True)
        # The menu-bar app is only launched as the machine's native slice: an Intel launch on Apple silicon
        # goes through Rosetta and makes macOS show its "Support Ending for Intel-based Apps" notice.
        if args.arch == platform.machine():
            check_app(directory, args.arch)
            print(f"{args.arch}: app launch and cleanup passed (global hotkey is manual-only)")
        else:
            print(f"{args.arch}: app launch skipped (not the native slice)")


if __name__ == "__main__":
    main()
