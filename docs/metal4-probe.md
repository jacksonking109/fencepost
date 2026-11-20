# Metal 4 probe

Measured on an Apple M3 Pro running macOS 26.6.2 (25G83), built with the active Xcode macOS 27.2 SDK. Run the commands from the Fencepost repo in zsh. GPU tests used the host session because the sandbox returned no default Metal device.

## Build and Rosetta

```zsh
pgrep -f wineserver
```

`pgrep` exited 1 with no output, so the build could use the repo’s `out/` directory.

```zsh
./build.sh
lipo -archs out/mtl4test
lipo -archs out/mtl4bench
codesign --verify --verbose=1 out/mtl4test out/mtl4bench
probe_scratch=$(mktemp -d /tmp/fencepost-metal4-followup.XXXXXX)
set -e -o pipefail
```

The build completed without warnings. Both probes reported `x86_64 arm64` and passed signature verification. Both slices printed the same startup lines in the compute and coexistence runs:

```text
device: Apple M3 Pro
supportsFamily:MTLGPUFamilyMetal4: yes
newMTL4CommandQueue: ok
newCommandAllocator: ok
newCommandBuffer: ok
```

Rosetta ran both synchronized compute cases, all three planted races, and the Metal 4 render benchmark. `ok` and `event` matched the expected checksum, and both render paths produced the expected pixel. Every run in the final matrix exited successfully on both architectures.

## Cross-queue feedback timeout

The first `event` attempt used one `MTL4CommitOptions` for both queues and timed out after one second. `MTL4CommandQueue.h` defines queue signals to cover prior work and queue waits to gate future work. The probe commits the producer, schedules its signal, schedules the consumer wait, then commits the consumer. In `eventlate` it schedules an additional earlier signal before committing the producer, so both values are scheduled before the consumer wait. [MTL4CommandQueue](https://developer.apple.com/documentation/metal/mtl4commandqueue)

The follow-up used distinct options, callbacks, and semaphores for each commit, with a fresh five-second deadline for each CPU wait. It was first built in scratch and run as follows:

```zsh
clang -arch arm64 -arch x86_64 -O2 -g -Wall -Wextra -Wno-unused-parameter \
    -fobjc-arc -mmacosx-version-min=13.0 -o "$probe_scratch/mtl4test" tests/mtl4test.m \
    -framework Metal -framework Foundation
codesign -f -s - "$probe_scratch/mtl4test"
for debug in 0 1; do
    for attempt in 1 2 3; do
        /usr/bin/time -p arch -arm64 -e MTL_DEBUG_LAYER="$debug" -e MTL_SHADER_VALIDATION=0 \
            "$probe_scratch/mtl4test" event
    done
done
```

Each run printed `producer commit feedback received`, `consumer commit feedback received`, and `case event: expected 2147516416 got 2147516416`. The wall times in seconds were `0.61`, `0.03`, `0.03` without validation and `0.04`, `0.03`, `0.04` with API Validation.

An arm64 scratch control reused `writeCommit` for the consumer, awaited two posts to `writeDone` from its single handler, and scheduled a signal to value `3` immediately after the consumer commit. Running `/usr/bin/time -p arch -arm64 -e MTL_DEBUG_LAYER=1 -e MTL_SHADER_VALIDATION=0 "$probe_scratch/shared-options-control" event` produced:

```text
shared commit feedback received
mtl4test: timed out waiting for second shared commit feedback
event.signaledValue at timeout: 3
event.signaledValue 5 s later: 3
real 10.28
```

The consumer’s completion signal proved both workloads had finished. The shared options delivered only one callback. The probes use fresh options per commit.

## Apple validation

`man MetalValidation` and Xcode’s API and Shader Validation documentation describe warnings, residency checks, and shader memory checks. I found no documented setting for missing Metal 4 barriers or early event waits. [API Validation](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-api-usage), [Shader Validation](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage)

```zsh
for mode in api shader; do
    validation_env=(-e MTL_DEBUG_LAYER=1)
    if [[ $mode == api ]]; then
        validation_env+=(-e MTL_SHADER_VALIDATION=0)
    else
        validation_env+=(-e MTL_DEBUG_LAYER_WARNING_MODE=nslog -e MTL_SHADER_VALIDATION=1
                         -e MTL_SHADER_VALIDATION_GLOBAL_MEMORY=1 -e MTL_SHADER_VALIDATION_RESOURCE_USAGE=1
                         -e MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1)
    fi
    for a in arm64 x86_64; do
        for c in ok nobarrier wrongstage event eventlate; do
            /usr/bin/time -p arch "-$a" "${validation_env[@]}" ./out/mtl4test "$c"
        done
    done
done
```

`Metal API Validation Enabled` appeared in all 20 processes. `Metal GPU Validation Enabled` also appeared in the 10 combined runs. There were no other validation messages. Wall time per process was 0.04–0.48 seconds. Each case printed an expected checksum of `2147516416`. The observed `got` values were:

| Case | arm64 API | x86_64 API | arm64 API + Shader | x86_64 API + Shader |
|---|---:|---:|---:|---:|
| `ok` | 2147516416 | 2147516416 | 2147516416 | 2147516416 |
| `nobarrier` | 2147516416 | 2147515240 | 2147516416 | 2147516416 |
| `wrongstage` | 2147505976 | 2147505976 | 2147516416 | 2147516416 |
| `event` | 2147516416 | 2147516416 | 2147516416 | 2147516416 |
| `eventlate` | 2147508160 | 2147510088 | 2147516280 | 2147516280 |

Apple validation was silent for all three planted hazards. Each hazard produced a wrong checksum in at least one run while validation was active. This leaves a gap that Fencepost’s Metal 4 front end could cover.

## CPU cost per draw

```zsh
for attempt in 1 2 3; do
    for a in arm64 x86_64; do
        for path_mode in metal3 metal4; do
            arch "-$a" -e MTL_DEBUG_LAYER=0 -e MTL_SHADER_VALIDATION=0 ./out/mtl4bench "$path_mode"
        done
    done
done
```

Each run encoded 10,000 draws per frame for 60 frames, with 5 warmups. All runs printed `pixel: 0,255,0,255`. The draw loop changes a vertex buffer binding and a fragment texture binding on each draw and shades one pixel in a 256 × 256 offscreen target. The CPU interval includes encoding and commit. GPU completion waits, the Metal 4 allocator reset, and registration of feedback for the test harness are outside that interval. Values below are the reported median / p90 in CPU µs/draw.

| Slice | Path | Run 1 | Run 2 | Run 3 |
|---|---|---:|---:|---:|
| arm64 | Metal 3 | 0.100400 / 0.117342 | 0.092083 / 0.096192 | 0.090117 / 0.093187 |
| arm64 | Metal 4 | 0.085062 / 0.088946 | 0.085408 / 0.088633 | 0.085396 / 0.090125 |
| x86_64 | Metal 3 | 0.140204 / 0.152762 | 0.145846 / 0.151046 | 0.141433 / 0.145517 |
| x86_64 | Metal 4 | 0.147050 / 0.158108 | 0.137729 / 0.141196 | 0.139417 / 0.142429 |

The median of the three per-run medians fell by 7.3% on arm64 (0.092083 to 0.085396 µs/draw) and 1.4% under Rosetta (0.141433 to 0.139417 µs/draw). Metal 4 was slower in one of the three paired Rosetta runs. Under Rosetta that is within the run-to-run spread (4–7%), so there is no measurable difference. On arm64 the gain is about 6–7%.

Two limits apply. The workload binds only three resources, so it can't show the per-resource tracking cost that Metal 4 removes. A translation layer with thousands of resources may gain more. The final-pixel check sees only the last draw. A scratch variant that gave each draw its own pixel (1,000 draws, both paths, both slices) found all 256 pixels correct, so Metal 4 captures argument-table bindings per draw and the two paths do the same work.

**For wine-lab:** the Metal calls themselves cost about 0.14 µs per draw under Rosetta. PEAK's D3D12 replay, measured on 2026-10-09, cost about 2.4 µs per draw. So the Metal API is roughly 6% of replay cost. Even a free Metal 4 couldn't remove most of the CPU limit, which sits in the translation layer's own work. A D3D12 port's CPU cost and frame rate on Metal 4 remain unmeasured.

## Fencepost coexistence

```zsh
unset DYLD_INSERT_LIBRARIES FENCEPOST_TRACE FENCEPOST_LOG FENCEPOST_RING FENCEPOST_CAPTURE
for a in arm64 x86_64; do
    for executable in mtl4test mtl4bench; do
        arg=ok
        [[ $executable == mtl4bench ]] && arg=metal4
        dir="$probe_scratch/$a/$executable"
        mkdir -p "$dir"
        arch "-$a" -e MTL_DEBUG_LAYER=0 -e MTL_SHADER_VALIDATION=0 "./out/$executable" "$arg" |
            tee "$dir/baseline.stdout"
        arch "-$a" -e MTL_DEBUG_LAYER=0 -e MTL_SHADER_VALIDATION=0 \
            -e DYLD_INSERT_LIBRARIES="$PWD/out/libfencepost.dylib" -e FENCEPOST_TRACE="$dir/m4.json" \
            -e FENCEPOST_LOG="$dir/fencepost.log" "./out/$executable" "$arg" |
            tee "$dir/injected.stdout"
        if [[ $executable == mtl4test ]]; then
            diff -u "$dir/baseline.stdout" "$dir/injected.stdout"
        else
            diff -u <(rg '^pixel:' "$dir/baseline.stdout") <(rg '^pixel:' "$dir/injected.stdout")
        fi
        cat "$dir/fencepost.log"
    done
done
python3 -I - "$probe_scratch" <<'PY'
import json
from pathlib import Path
import sys

for path in sorted(Path(sys.argv[1]).glob("*/*/m4.json")):
    records = json.loads(path.read_text())
    summary, = (record["args"] for record in records if record["name"] == "fencepost summary")
    print(f"{path.parts[-3]}/{path.parts[-2]}: {[(r['ph'], r['name']) for r in records]}")
    keys = ("encoders_traced", "cbs", "nodes", "accesses", "flows", "markers")
    print(" ".join(f"{key}={summary[key]}" for key in keys))
    assert all(summary[key] == 0 for key in keys)
PY
```

Both slices printed `case ok: expected 2147516416 got 2147516416` and `pixel: 0,255,0,255` with and without injection. These are the paired benchmark timings, in CPU µs/draw:

| Slice | Without Fencepost median / p90 | With Fencepost median / p90 |
|---|---:|---:|
| arm64 | 0.092567 / 0.094617 | 0.088204 / 0.092575 |
| x86_64 | 0.152104 / 0.161288 | 0.150200 / 0.157842 |

Every injected log contained just the `fencepost: watching Metal in pid …; reports go to FENCEPOST_LOG` startup line. Each `m4.json` parsed as a JSON array whose sole record was `fencepost summary`. For all four runs the summary printed:

```text
encoders_traced=0 cbs=0 nodes=0 accesses=0 flows=0 markers=0
```

Fencepost coexisted with both Metal 4 programs on both slices. It intercepted device creation and recorded zero Metal 4 work. The paired run is a functional smoke test with timing variation between processes.

## Regression verification

```zsh
export MTL_DEBUG_LAYER=0 MTL_SHADER_VALIDATION=0
for a in arm64 x86_64; do
    arch "-$a" -e DYLD_INSERT_LIBRARIES="$PWD/out/libfencepost.dylib" ./out/fptest
    for t in ring trigger capture; do
        python3 -I "tests/${t}_check.py" "$a"
    done
done
git diff main --check
git diff main --stat
```

`fptest: 34/34 passed` on each slice, along with the ring, trigger, and capture checks. The trigger script ran the app smoke test on arm64 and skipped the Intel app launch by design. The diff covers `build.sh`, `tests/mtl4test.m`, `tests/mtl4bench.m`, and this report.
