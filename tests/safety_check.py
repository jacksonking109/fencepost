#!/usr/bin/env python3
"""Regression tests for log backpressure, detector ownership and trace retention."""
import argparse
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FINAL = "fencepost: final: 0 race(s) (0 hits), 0 possible (0 hits); 1 encoders analysed"


def command(arch, mode, **knobs):
    cmd = ["arch", "-" + arch, "-e", f"DYLD_INSERT_LIBRARIES={OUT / 'libfencepost-testhooks.dylib'}"]
    for key, value in knobs.items():
        cmd.extend(["-e", f"{key}={value}"])
    return [*cmd, OUT / "safetytest", mode]


def fill_pipe(fd):
    os.set_blocking(fd, False)
    for size in (4096, 1):
        while True:
            try:
                os.write(fd, b"X" * size)
            except BlockingIOError:
                break
    os.set_blocking(fd, True)


def check_log(arch, env, drain):
    read, write = os.pipe()
    try:
        fill_pipe(write)
        with subprocess.Popen(command(arch, "log"), env=env, stdout=subprocess.PIPE,
                              stderr=write) as child:
            os.close(write)
            write = None
            try:
                assert select.select([child.stdout], [], [], 5)[0], "detector blocked on stderr"
                assert child.stdout.readline() == b"detector returned with blocked pipe\n"
                if drain:
                    chunks = []
                    while True:
                        assert select.select([read], [], [], 5)[0], "log drain stalled"
                        chunk = os.read(read, 65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    log = b"".join(chunks).lstrip(b"X").decode()
                    notices = list(re.finditer(r"fencepost: (\d+) log lines dropped", log))
                    written = re.findall(r"fencepost probe: (\d+)", log)
                    dropped = sum(int(notice[1]) for notice in notices)
                    assert notices and len(written) + dropped == 16000, (len(written), dropped)
                    assert len(re.findall(r"fencepost probe: ", log[:notices[0].start()])) <= 1
                    assert written == sorted(set(written)) and FINAL in log
                assert child.wait(timeout=5) == 0
                if not drain:
                    saved, = Path("/tmp").glob(f"fencepost-final-{child.pid}-*")
                    assert FINAL in saved.read_text()
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                for saved in Path("/tmp").glob(f"fencepost-final-{child.pid}-*"):
                    subprocess.run(["trash", saved], check=True)
    finally:
        os.close(read)
        if write is not None:
            os.close(write)


def check(arch, mode, env, **knobs):
    result = subprocess.run(command(arch, mode, **knobs), env=env, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, (mode, result.returncode, result.stdout, result.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch", choices=["arm64", "x86_64"])
    args = parser.parse_args()
    arch = args.arch
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("FENCEPOST_", "DYLD_")) and key != "MTL_CAPTURE_ENABLED"}
    check_log(arch, env, drain=False)
    check_log(arch, env, drain=True)
    assert "node pool exhausted: 11 nodes skipped" in check(arch, "pool", env).stderr
    resource = check(arch, "resource", env, NSZombieEnabled="YES").stderr
    assert re.search(r'RACE write-after-write on buffer "released safety buffer" 16 bytes .*bytes \[4, 12\)', resource), resource
    assert "1 encoders released without endEncoding" in check(arch, "abandoned", env).stderr
    with tempfile.TemporaryDirectory() as directory:
        stream = Path(directory) / "stream.json"
        result = check(arch, "stream", env, FENCEPOST_TRACE=stream)
        assert "20000 completed, 0 live CB records, 0 live endpoints" in result.stdout
        summary = json.loads(stream.read_text())[-1]["args"]
        assert summary["encoders_traced"] == 20000 and summary["dropped"] == 0, summary
        subprocess.run([sys.executable, "-I", ROOT / "tests/trace_check.py", stream], check=True)
        recycled = Path(directory) / "recycled.json"
        check(arch, "trace_recycled", env, FENCEPOST_TRACE=recycled, FENCEPOST_TRACE_GPU=0)
        summary = json.loads(recycled.read_text())[-1]["args"]
        assert summary["encoders_traced"] == 1 and summary["dropped"] == 0, summary
    print(f"{arch}: safety checks passed")


if __name__ == "__main__":
    main()
