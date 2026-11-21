# fencepost

A race detector for Metal GPU synchronisation. Inject it into a Metal 3 or Metal 4 program to find encoders
that touch the same memory without ordering between their accesses.

## Quick start

From a source checkout on a Mac with the macOS 26 SDK, run the bundled detector checks or trace your own executable.

```sh
./build.sh
export FP_LIB="$PWD/out/libfencepost.dylib"
DYLD_INSERT_LIBRARIES="$FP_LIB" out/fptest
DYLD_INSERT_LIBRARIES="$FP_LIB" /path/to/your-metal-program
FENCEPOST_TRACE=/tmp/fencepost.json DYLD_INSERT_LIBRARIES="$FP_LIB" /path/to/your-metal-program
```

Reports go to stderr. Open the JSON in Perfetto to inspect the timeline. Prebuilt binaries are available under [Download](#download).

```
fencepost: RACE read-after-write on texture 64x64, pixel format 70, 1 levels 0x7ff6cb75e1c0, level 0, slice 0
  first : blit encoder #18 (command buffer #8, encoder 1, queue 1) writes it: blit destination
  second: render encoder #19 (command buffer #8, encoder 2, queue 1) reads it: useResource, vertex stage
  why   : both are in command buffer #8 and no fence orders them; untracked resources need updateFence in the
          first and waitForFence in the second
```

## Why

Metal 3 can order GPU work for resources it tracks. Metal 4 queues require explicit synchronisation. Translation layers (DXMT, MoltenVK, D3D12 runtimes) and
many engines create resources with `MTLHazardTrackingModeUntracked`, or reach them through argument buffers Metal does
not track, because tracking costs CPU time. Then ordering is the program's job, through `updateFence`/`waitForFence`
and `MTLEvent` signals and waits. Forget one and two passes overlap on the GPU: a read sees half-written data, and a
frame flickers. The bug comes and goes with timing, it often disappears under a frame capture (captures serialise the
GPU), and Metal's API validation does not report it. Vulkan has a synchronisation validation layer for this class of
bug. As far as we could find, Metal has nothing like it.

fencepost judges ordering the way ThreadSanitizer does for CPU threads: from the synchronisation the program
*encoded*, not from when the GPU happened to run things. So it reports a race even on a run where the race did not
show.

## What it checks

**Resource races.** On Metal 3, for each untracked texture or buffer, fencepost records which encoder reads or writes it, through:
- render pass attachments (load, store, MSAA resolve, visibility buffer);
- `useResource` (with its usage and stages);
- vertex/fragment/compute bindings;
- index and indirect buffers;
- blits, with exact byte ranges, mip levels and slices.

Texture views are mapped onto the storage they share, so two views of different mip levels do not conflict. Two
conflicting accesses (at least one a write) whose encoders have no ordering path between them are a race.

On Metal 3 and Metal 4, a `DontCare` store discards the attachment. A resolve-only MSAA store writes the resolve
target alone. Metal 3 also uses the final store action set on the encoder when the descriptor defers that choice.

**Declared barriers.** On Metal 3, a runtime that reaches resources only through argument buffers or residency sets (bindless
D3D12, for one) gives fencepost nothing to see. Such a runtime can still say what ordering it *means*:

```c
void (*fp_barrier)(id) = dlsym(RTLD_DEFAULT, "fencepost_barrier");   /* NULL when fencepost is not injected */
if (fp_barrier) fp_barrier(commandBuffer);   /* at a D3D12 ResourceBarrier, a Vulkan pipeline barrier, ... */
```

Every encoder submitted to that queue before the call must then be ordered before every encoder after it. fencepost
checks that ordering against the encoded fences and events, and against stage barriers on Metal 4. This check also works when resource accesses are invisible. A render encoder that waits
only before its fragment stage is reported separately: its vertex stage can still overlap.

An encoder the runtime orders by other means (a present blit that touches only tracked resources, say) can be left out
of these checks as an earlier encoder with `fencepost_exempt(encoder)`, called while it is open. Its resource accesses
are still checked.

**How parallel each frame is.** Too much synchronisation costs as well: encoders chained one after another leave the
GPU nothing to overlap. For each Metal 3 frame, fencepost counts the encoders and finds the longest chain ordered by
fences and events. The running summary prints the medians. Metal 4 has [different chain limits](#limits).

```
fencepost:   frames: 833; median of the last 833: 4 encoders, longest ordered chain 2 (parallelism 2.00); the GPU had run out of work at 1.3 commits per frame
```

Parallelism 1.00 means every encoder waits for the one before it. "Run out of work" counts commits made when every
earlier command buffer had already completed: the GPU sat idle, waiting for the CPU (or for vsync). `FENCEPOST_CHAIN=N`
lists frame N's chain, with runs of the same pass folded (labels compared without `#<number>`):

```
fencepost: CHAIN frame 300: 4 encoders; the longest chain the synchronisation forces to run one after another has 2 (parallelism 2.00)
  1           render  "R#301 MainVS|MainPS 640x480"
  2           blit    "B f6 join"
```

A program that does not present, or wants its own frame boundaries, calls `fencepost_frame(commandBuffer)` (found with
`dlsym` like `fencepost_barrier`); from its first call, presents no longer end frames.

## Timeline (FENCEPOST_TRACE)

```
FENCEPOST_TRACE=/tmp/fencepost-%p.json DYLD_INSERT_LIBRARIES=$PWD/out/libfencepost.dylib your-metal-program
```

In [Perfetto](https://ui.perfetto.dev/), choose **Open trace file** and select the JSON for the process that encoded
the work. Chrome's `chrome://tracing` also reads it. The **Frames** track spans present to present (or the
program's explicit frame boundaries) and carries the chain length, parallelism and idle-commit count. The first frame
starts when fencepost sees the first command buffer. Each queue has command buffer and encoder tracks, with extra
lanes for overlapping slices. In a GPU-timed command buffer, an encoder without GPU samples gets a short estimated
slice after the preceding encoder or at the command buffer start. It has `gpu_time: false` and `estimated: true`.
When the command buffer has no GPU span, its encoders use a separate CPU encode time queue track.
The **CPU encode** tracks show the work of creating and ending encoders on each CPU thread.

Arrows show fence waits, event waits and work that completed before the next commit. Event arrows start on the
signalling command buffer. Each command buffer can receive one completed arrow from the latest observed completion
per queue at commit. Race and declared-barrier markers carry the same detail as the log, including hits that repeat
or exceed its report limit. The `overlapped_on_gpu` argument is `true` if the two sampled GPU spans intersect,
`false` if both were sampled and did not intersect, and `null` if either sampled span is unavailable. `gpu_order`
says which way round they ran on this run: `"overlapped"`, `"reversed"` (the second finished before the first
started: the ordering the program needed was inverted, not just missing) or `"in order"` (the race did not show on
this run), `null` without both spans.
The streaming trace keeps up to 20,000 of these markers. Perfetto can omit arrows whose destination ran earlier than the
source finished, or attach an arrow to a diagnostic marker at the destination's start. The JSON retains both
endpoints. The **fencepost summary** gives the detector counters and trace counts.

On Metal 3, GPU timestamps require stage-boundary sampling and measure whole encoders. Render spans run from
vertex start to fragment end. Empty passes, parallel render encoders and systems without usable counter samples
have no per-encoder GPU timing. For timestamped passes fencepost passes Metal a descriptor copy (or makes a
descriptor for the descriptorless calls). A program that attaches its own counter sample buffers at the same stage
boundaries, such as Madeira with `gpu-pass-profile`, keeps its samples. Fencepost skips sampling those encoders
and also skips sampling if all attachment slots are occupied.

Set `FENCEPOST_TRACE_GPU=0` to disable per-encoder sampling and keep command buffer GPU timing.

The trace is flushed with running summaries and on exit. Perfetto can read the complete records in a file whose
process stopped before writing the closing bracket.

## Metal 4

On macOS 26 and later with a Metal 4 device, fencepost records queues, command buffers, render and compute
encoders (including blit commands), and queue event signals and waits. A commit that batches several command
buffers has one GPU span from commit feedback. Sampled encoders have individual spans within it. The trace
puts overlapping encoders in separate lanes and uses their samples to annotate race markers.

### Resources and pipeline reflection

Race analysis tracks render attachments, copies, fills, mipmap generation, and the argument-table bindings
that each draw or dispatch uses. When Metal supplies a GPU address inside a known buffer without a length,
the access covers the rest of that buffer. Texture views are checked against their underlying storage.
The summary counts unresolved addresses and resource IDs.

Argument-table slots take their read/write access from the bound pipeline's binding reflection, per stage. A
slot that a stage's shader does not declare or use has no access in that stage, so a table bound at
Vertex|Fragment counts a buffer only in the stage that reads it. `const` buffers and read textures are reads,
write-only textures are writes, and read-write bindings are definite writes. Metal reflects every non-`const`
`device` pointer as read-write, even when the shader only reads it. `FENCEPOST_MTL4_RW=possible` reports those
buffer conflicts as possible races. Read-write textures remain definite writes. Fencepost requests reflection
when pipelines are built:
- through the device (`newComputePipelineStateWith…`, `newRenderPipelineStateWith…`, tile and mesh variants,
  synchronous and asynchronous), once the process has made a Metal 4 queue, command buffer, argument table or
  compiler, so programs that use only Metal 3 build their pipelines unchanged. It adds
  `MTLPipelineOptionBindingInfo`, except with `MTLPipelineOptionFailOnBinaryArchiveMiss`;
- through `MTL4Compiler` (compute, render and specialization, with and without dynamic linking, synchronous and
  `MTL4CompilerTask`), from a copy of the app's descriptor with `shaderReflection` including `BindingInfo`.

An app that requests no Metal 4 compiler reflection still sees nil from `pipeline.reflection`.
An app that requests reflection through the device API also gets the bindings, even when it omits the
binding information option. If the added reflection causes a build to fail, fencepost retries
the original request for synchronous and asynchronous calls. Pipelines fencepost did not see built (binary
archives, data sets, device pipelines made before the first Metal 4 object) use their own reflection when Metal kept one, and
are otherwise counted as "access unknown" in the summary. Their slots fall back to reads, or to possible writes
with `FENCEPOST_BIND_MAYWRITE=1`.

### Ordering between encoders

Queue submission order alone does not order resource accesses. The source stage must cover the producer's
access, and the destination stage must cover the consumer's. For example, a Blit → Dispatch barrier leaves an
earlier Dispatch write unordered.

A consumer barrier orders the specified stages of earlier encoders before matching work after the barrier in
its own encoder and in later encoders on the queue. The SDK header describes only the current encoder.
Fencepost applies the broader rule in Apple's online documentation. The race suite checks it across command
buffers and commits. [Apple's consumer barrier reference](https://developer.apple.com/documentation/metal/mtl4commandencoder/barrierafterqueuestages:beforestages:visibilityoptions:)

A producer barrier orders earlier commands in its own encoder before work at the specified stages in later
encoders. Two barriers chain through a shared stage only if both cover an operation in that stage, which can
be a fence update or wait. Fences order covered work on the same queue. Queue event signals and waits order
work across queues, including between Metal 3 and Metal 4. Commit feedback orders completed work before a
later submission. The trace summary counts waits with missing event records in `event_wait_lookup_misses`
and waits whose signal history has become unavailable in `event_wait_unknown_signals`. These waits resolve
without a dependency edge. Signal records follow the event object’s lifetime. If an evicted signal value could
satisfy a wait, fencepost counts its source as unknown.

Render attachments load at the start of the pass and store at its end, both in the Fragment stage. Apple's
documentation leaves the effect of an in-encoder wait on the load unspecified. Fencepost treats a consumer
barrier or fence wait covering Fragment before the first draw as covering both loads and stores. After a draw,
it covers stores only. A Fragment fence update or producer barrier covers stores only if it follows the last
draw. One before or between draws leaves the stores and subsequent draws uncovered. A pass stitched across
command buffers in one commit loads in its first segment and stores in its last. Fragment work in those
segments follows submission order.

In the race suite’s stitched pass, a separate writer conflicts with the resumed segment’s store.

```text
fencepost: RACE write-after-write on Metal 4 texture 0x1, levels [0,1), slices [0,1)
  first : render encoder #3 "MTL4 outside writer" (command buffer #3, encoder 1, queue 0) writes it: attachment, Fragment stage
  second: render encoder #1 "MTL4 resume" (command buffer #1, encoder 1, queue 0) writes it: attachment, Fragment stage
  why   : no Metal 4 ordering covers Fragment -> Fragment
```

### Timing, frames and capture

Fencepost submits `commit:count:` as `commit:count:options:` with fresh options for its own feedback. For
app options it tracks, it transfers the registered handlers to fresh options for that submission. Each handler
registration is consumed once, including when the app reuses its options.

With GPU tracing on, fencepost writes two Relaxed timestamps per sampled render or compute encoder. Render
samples run from Vertex before the first draw to Fragment after the last. Fencepost resolves the timestamps on
commit feedback. See [Timeline](#timeline-fencepost_trace) for the sampling switch.

Sampled spans measure elapsed GPU time. In a fixed-workload probe, the GPU clock varied by up to about 9×
within and between commits, so the same work took different amounts of time. In about 0.1–1% of encoders,
a Relaxed start timestamp landed late, shortening the span and leaving a gap before it. Use spans to compare
ordering, overlap and relative cost. A span of a given length can represent different amounts of work.

Each device has 4,096 timestamp entries, enough for 2,048 encoder pairs across outstanding submissions.
When the heap fills, the encoder type is unsupported (machine learning or suspend/resume render segments),
or the timestamp pair is invalid, the trace estimates the encoder's position. With GPU sampling enabled,
these fallbacks count toward `sampling_skipped`.

`signalDrawable:` marks the end of a frame. Programs can use `fencepost_frame` to supply their own frame
boundaries. See [Catching a bad frame](#catching-a-bad-frame-fencepost_ring) for ring triggers and dumps.
`FENCEPOST_CAPTURE` logs that Xcode GPU capture is unavailable for Metal 4
because fencepost learns the submitting queue after encoding.

### Limits

Fencepost records a queue's events in the order its hooks see them. When two threads submit to the same
`MTL4CommandQueue` without the app serializing those calls, Metal can see them in a different order.

Metal 4 analysis covers races between encoders. It skips within-encoder races, argument-buffer and indirect
resource contents, and `MTL4VisibilityOptions` correctness. Residency-set membership alone supplies no
accesses. The summary counts encoders with residency sets and no visible accesses, machine-learning encoders,
and acceleration-structure builds. Frame chains follow queue event ordering. Stage barriers and fence waits
can serialize work without extending the reported chain, so parallelism can read higher than the GPU achieved.

## Catching a bad frame (FENCEPOST_RING)

```
FENCEPOST_RING=10 FENCEPOST_TRACE=/tmp/fencepost DYLD_INSERT_LIBRARIES=$PWD/out/libfencepost.dylib your-metal-program
```

Fencepost keeps the last 10 frames of timeline in memory. A trigger on frame 30 keeps frames 21 through 30,
then records two more by default. Once their analysis and GPU timings are available, fencepost logs the path
to `<prefix>-<pid>-f<frame>-<reason>.json`. Open that file in Perfetto and find **TRIGGER** on the **Frames**
track. Each dump includes its own track names and summary. On normal exit a pending dump contains the work
available so far.

A new race or ORDER signature that gets a detailed report triggers `race`. A command buffer error triggers
`gpu-error` and includes the error description. A long CPU frame triggers `spike` when its boundary-to-boundary
duration exceeds both the previous 120 frames' median times the factor and that median plus the minimum increase.
With `MTL_CAPTURE_ENABLED=1`, timeline spikes default to off because capture instrumentation changes frame timing.
Set `FENCEPOST_TRIGGER_SPIKE=4:50` or `FENCEPOST_SPIKE=1` to enable the usual threshold.
A program can also call `void fencepost_trigger(const char *reason)` through `dlsym`. Only processes that have
encoded work accept triggers.

To mark a glitch yourself, run `printf 'flicker\n' > /tmp/fencepost-trigger-$(id -u)`. Fencepost checks for a
newer modification time at commit, at most once every 250 ms. The request file must belong to the effective
user of the traced process. An existing file owned by that user establishes the starting time.
The first line supplies the reason (up to 32 characters, sanitised to `[a-z0-9-]`), or `manual` for an empty
line. An empty file is ignored until it contains a line. Fencepost only reads the file, so several processes
can observe the same update. Set
`FENCEPOST_TRIGGER_FILE` to use a different path.

The memory cap evicts the oldest frames first and can shorten either side of a capture. A command buffer belongs
to its first encoder's frame. Arrows and diagnostic markers belong to their destination frame. The summary counts
arrows and markers dropped because their source is outside the dump, frame evictions, and triggers suppressed
while another dump was pending, during cooldown, or after the dump limit. Detector counters cover the process
up to the dump.

Manual requests (trigger file, hotkey, CLI, `fencepost_trigger`) and automatic ones (race, GPU error, spike) have
separate cooldowns, so an automatic dump never holds off one you asked for. A manual request that arrives while
another dump is still collecting frames is queued and saved as the next dump; a second one is refused. Fencepost
logs a line for every manual request it queues or refuses, with the reason.

### Hotkey helper

Run `./build.sh`, then `open "out/Fencepost Trigger.app"`. Choose **Save timeline now** in the **FP** menu
or press **Control–Option–Command–F**. Both target the frontmost app's process. If the helper itself is
frontmost or there is no frontmost app, the request goes to all traced processes. The global hotkey takes
that combination away from the game while the app runs. If macOS can't register it, the menu shows
**Global hotkey unavailable**.

To change the hotkey or the trigger path, run these commands and restart the app. Use `ctrl`, `opt`, `cmd`
and `shift` with a key from `f1`–`f19`, `a`–`z` or `0`–`9`.

```sh
defaults write io.github.jacksonking109.fencepost-trigger hotkey 'ctrl+opt+cmd+f10'
defaults write io.github.jacksonking109.fencepost-trigger triggerFile -string /tmp/my-fencepost-trigger
```

Set `FENCEPOST_TRIGGER_FILE` to the same path in the traced process. The default is `/tmp/fencepost-trigger-<uid>`.
For scripts, use `out/fencepost-trigger flicker` or pass `--file /tmp/my-fencepost-trigger`. The CLI uses
`FENCEPOST_TRIGGER_FILE` when `--file` is omitted and uses `manual` when no reason is given. It targets
all traced processes unless you pass `--pid N`. The request file's first line is the reason. An optional
second line `pid=N` targets only that process. Both helpers work with traced Wine and native programs.

To check the hotkey by hand, run a traced program with `FENCEPOST_RING` enabled and press the combination while
the program is rendering. A function-key hotkey also needs Fn on keyboards whose
top row sends media keys. The **FP** item should
briefly show a checkmark, and fencepost should log a dump whose filename ends in `-hotkey.json`.

## Xcode capture

Launch a Metal 3 program with `MTL_CAPTURE_ENABLED=1` and `FENCEPOST_CAPTURE` set to a filename prefix in an
existing directory. Capture is opt-in and works independently of `FENCEPOST_TRACE` and `FENCEPOST_RING`.
In our submission probe, `MTL_CAPTURE_ENABLED=1` added about 67% CPU overhead even with no capture running.

```sh
MTL_CAPTURE_ENABLED=1 FENCEPOST_CAPTURE=/tmp/fencepost \
    DYLD_INSERT_LIBRARIES=$PWD/out/libfencepost.dylib your-metal-program
out/fencepost-trigger --capture flicker
```

Choose **Capture next frame** in the **FP** menu or press **Control–Option–Command–C** to target the
frontmost app's process, with the same fallback as the timeline hotkey. The CLI sends to all traced
processes unless you pass `--pid N`. You can also call `void fencepost_capture(const char *reason)` through
`dlsym`. To customise the app, set `captureHotkey` and
`captureFile` in the same defaults domain as the timeline helper and restart it. Set the matching
`FENCEPOST_CAPTURE_FILE` in the Metal program. Like the timeline trigger file, it uses its modification
time, a reason on line one, and an optional `pid=N` on line two.

After the program has ended at least one frame, fencepost starts capture before allocating the next frame's
first command buffer. It ignores requests while a capture is active. Open the logged
`<prefix>-<pid>-f<frame>-<reason>.gputrace` in Xcode with **File → Open**.

## How Metal 3 analysis works

- **Nodes:** every encoder is one. A render encoder has two stages, vertex and fragment.
- **Edges** order nodes:
  - `updateFence:afterStages:` paired with `waitForFence:beforeStages:`. Pairs are resolved in submission order, so a
    wait meets the latest update `enqueue`d before it.
  - An event wait paired with a signal whose value reaches the wait threshold.
  - A command buffer that completed before another was committed.
- **Ancestor sets:** for each node and stage, fencepost keeps a bitset of everything ordered before it, over a sliding
  window of 4096 nodes. A query is one bit test.
- **When analysis runs:** at `commit`. A node whose event wait is not yet signalled waits until it is, or until the CPU
  sets the value, or for 256 commits.
- **Tracked resources** are skipped: Metal orders those itself.

## Metal 3 limits

The summary reports blind spots where it can identify them.
- **Argument buffers and residency sets:** resources reached only through them are invisible. Declared barriers cover
  the ordering, not the data.
- **`useHeap`:** the resources inside a heap are not seen. Aliasing between resources placed in the same heap memory
  is not modelled.
- **Direct compute bindings** count as reads: whether a kernel writes them needs its pipeline's reflection, and
  translation layers declare their writes with `useResource` (`FENCEPOST_BIND_MAYWRITE=1` changes this).
- **Read+write `useResource`:** a kernel may or may not write the resource. Conflicts that rest only on such a "may
  write" are reported as *possible*, not as races.
- **Parallel render encoders** are outside Metal 3 race analysis.
- **Event signals** are taken to follow all earlier work on their queue.
- **Retired work:** a command buffer that completed before the next commit is treated as ordered. A race whose window
  closed that way on this run is missed.
- **Frame chains** follow fences and events only. Ordering Metal adds itself between encoders using tracked resources
  is not counted, so parallelism can read higher than the GPU really gets. "Run out of work" is a lower bound:
  completion handlers run a little after the GPU finishes.

## Knobs (environment)

| Variable | Default | Meaning |
|---|---|---|
| `FENCEPOST_LOG` | stderr | append reports to this file; `%p` in the name becomes the process id |
| `FENCEPOST_MAX_REPORTS` | 200 | detailed reports printed; the rest are counted |
| `FENCEPOST_POSSIBLE` | 1 | 0 hides "possible" reports |
| `FENCEPOST_SUMMARY_SECS` | 10 | print a running summary at most this often when the counts change; 0 = only at exit |
| `FENCEPOST_CHAIN` | 0 | list the longest ordered chain of this frame (counted from 1) |
| `FENCEPOST_BIND_MAYWRITE` | 0 | 1 counts unknown writes as possible in Metal 3 compute bindings and in Metal 4 argument tables bound with an access-unknown pipeline |
| `FENCEPOST_MTL4_RW` | definite (any value other than `possible`) | `possible` reports conflicts through read-write Metal 4 buffer bindings (non-`const` `device` pointers) as possible races |
| `FENCEPOST_TRACE` | unset | stream to this JSON file (`%p` expands to the process id), or use this literal prefix for ring dumps |
| `FENCEPOST_TRACE_FRAMES` | all | capture frames `a:b`, inclusive and counted from 1 |
| `FENCEPOST_TRACE_MB` | 512 | stop streaming when the file reaches this size in MiB |
| `FENCEPOST_TRACE_GPU` | 1 | 0 disables encoder counter sampling and leaves command buffer GPU timing on |
| `FENCEPOST_RING` | 0 | frames to retain through the trigger frame (inclusive), requires `FENCEPOST_TRACE` |
| `FENCEPOST_RING_AFTER` | 2 | additional frames to record after a trigger |
| `FENCEPOST_RING_MB` | 64 | memory budget in MiB for retained timeline records and metadata |
| `FENCEPOST_TRIGGER_RACE` | new | `new` fires on detailed reports, `all` on every race or ORDER hit, `0` disables |
| `FENCEPOST_TRIGGER_GPUERR` | 1 | trigger on command buffer errors, `0` disables |
| `FENCEPOST_SPIKE` | 1 | `0` disables spike triggers for both timeline dumps and GPU captures |
| `FENCEPOST_TRIGGER_SPIKE` | 4:50 (off with `MTL_CAPTURE_ENABLED=1` unless explicitly enabled) | timeline spike median multiplier and minimum increase in ms (`factor:min_ms`), `0` disables timeline spikes |
| `FENCEPOST_TRIGGER_FILE` | `/tmp/fencepost-trigger-<uid>` | watch this file for a newer modification time, empty disables |
| `FENCEPOST_TRIGGER_COOLDOWN` | 5 | seconds between accepted triggers of one kind (manual or automatic) in one process |
| `FENCEPOST_MAX_AUTO_DUMPS` | 20 | timeline dumps per process from race, GPU-error and spike triggers |
| `FENCEPOST_MAX_MANUAL_DUMPS` | 20 | timeline dumps per process from file, hotkey, CLI and API requests |
| `FENCEPOST_MAX_DUMPS` | 20 | fallback for either timeline dump quota when its own knob is unset |
| `FENCEPOST_CAPTURE` | unset | filename prefix for Metal 3 Xcode captures, requires `MTL_CAPTURE_ENABLED=1` |
| `FENCEPOST_CAPTURE_FILE` | `/tmp/fencepost-capture-<uid>` | request file, empty disables polling |
| `FENCEPOST_CAPTURE_ON` | empty | comma-separated `race`, `gpu-error`, `spike` auto triggers |
| `FENCEPOST_CAPTURE_FRAMES` | 1 | frame ends to record |
| `FENCEPOST_CAPTURE_IDLE_SECS` | 10 | seconds without command buffer commits before stopping, minimum 1 |
| `FENCEPOST_CAPTURE_MAX_SECS` | 120 | hard cap on recording time in seconds, minimum 1 |
| `FENCEPOST_CAPTURE_MAX` | 3 | captures allowed per process |
| `FENCEPOST_CAPTURE_MIN_FREE_GB` | 10 | minimum free GiB on the output volume |
| `FENCEPOST_CAPTURE_COOLDOWN` | 10 | seconds after a capture before another can start |
| `FENCEPOST_MTL4_TEST_ERROR_COMMIT` | 0 | test-hooks dylib only, simulate a feedback error on this one-based submission |
| `FENCEPOST_MTL4_TEST_REFLECTION_FAIL` | 0 | test-hooks dylib only, force this many added-reflection requests to fail to exercise retries |

Each distinct race is printed once and counted after that.

macOS only injects into programs without the hardened runtime (or with the `com.apple.security.cs.allow-dyld-environment-variables`
entitlement): your own builds, command-line tools and Wine, not most notarised or App Store apps. It also strips `DYLD_*` variables when a system binary (`/bin/sh`,
`/usr/bin/time`, `arch`) starts the program; use `arch -e DYLD_INSERT_LIBRARIES=...` or export the variable inside the
script that starts the program.

## Download

Prebuilt (universal: Apple Silicon and x86_64/Rosetta, macOS 13+) on the
[Releases](https://github.com/jacksonking109/fencepost/releases) page. A browser marks downloads as quarantined and
macOS then refuses to load the library ("library load disallowed by system policy"); clear the mark after extracting:

```
tar -xzf fencepost-0.3.0-macos.tar.gz
xattr -dr com.apple.quarantine fencepost-0.3.0-macos
DYLD_INSERT_LIBRARIES=$PWD/fencepost-0.3.0-macos/libfencepost.dylib  your-metal-program
```

The archive also holds the trigger tools: `fencepost-trigger` (command line) and `Fencepost Trigger.app` (menu-bar
hotkeys). The `xattr` line above clears them too.

The library is signed ad hoc, not by an Apple developer ID; building it yourself (below) takes seconds.

## Build and test

Build with Apple's command-line tools and the macOS 26 SDK for the Metal 4 APIs. The universal binaries target
macOS 13 and later. Metal 4 tests require a Metal 4 device and macOS 26 or later. The two-slice test commands
below assume Apple silicon with Rosetta installed.

```sh
git clone https://github.com/jacksonking109/fencepost && cd fencepost
./build.sh
./build.sh release 0.3.0  # also build a stripped release archive
```

Run the Metal 3 detector, trace, ring, trigger and capture checks on each slice:

```sh
for slice in arm64 x86_64; do
    arch "-$slice" -e DYLD_INSERT_LIBRARIES="$PWD/out/libfencepost.dylib" out/fptest
    arch "-$slice" -e DYLD_INSERT_LIBRARIES="$PWD/out/libfencepost.dylib" \
        -e FENCEPOST_TRACE="out/fptest-$slice.json" out/fptest
    python3 -I tests/trace_check.py "out/fptest-$slice.json"
    arch "-$slice" -e DYLD_INSERT_LIBRARIES="$PWD/out/libfencepost.dylib" \
        -e FENCEPOST_TRACE="out/tracetest-$slice.json" -e FENCEPOST_MAX_REPORTS=0 out/tracetest
    python3 -I tests/trace_check.py --shape "out/tracetest-$slice.json"
    for check in ring trigger capture; do python3 -I "tests/${check}_check.py" "$slice"; done
done
```

Run the Metal 4 checks and benchmarks:

```sh
for slice in arm64 x86_64; do
    for check in mtl4 mtl4_race mtl4_timing; do python3 -I "tests/${check}_check.py" "$slice"; done
    python3 -I tests/mtl4_check.py "$slice" --threads
    arch "-$slice" -e DYLD_INSERT_LIBRARIES="$PWD/out/libfencepost.dylib" out/mtl4frames
    arch "-$slice" -e DYLD_INSERT_LIBRARIES="$PWD/out/libfencepost.dylib" out/mtl4bench metal4
done
python3 -I tests/mtl4_overhead.py arm64 x86_64
python3 -I tests/mtl4_overhead.py --trace-timing arm64 x86_64
```

`tests/fptest.m` runs 43 checks in one Metal 3 program, covering race and barrier verdicts, frame chain lengths
and idle commits. Cases include

- a fence updated after the vertex stage while the write happens in the fragment stage;
- a fence re-updated by an unrelated encoder before the wait;
- a wait committed before its signal on another queue;
- a fence that must be resolved in enqueue order, not commit order;
- mip levels written and read through views;
- a fence wait that crosses a frame boundary, which must not lengthen the new frame's chain.

`tests/tracetest.m` encodes a fence pair, a planted race and a cross-queue event in two frames. The checker
verifies their exact arrows and marker, track lanes, and the trace's summary. Use `out/tracetest --app-counters`
to check that the app still gets its own timestamps when tracing is on.

`tests/trigger_check.py` checks trigger file writes, broadcast and PID-targeted CLI ring dumps and app startup in the logged-in
macOS session. Check the global hotkey by hand as described above.

On a Metal 4 device, `mtl4bench metal4-commit` measures CPU time for small submissions with up to eight in
flight, and `mtl4bench pipelines` times compute pipeline builds through `MTL4Compiler` and the device.
`tests/mtl4_overhead.py` runs each with and without fencepost, three times per slice, and prints the medians.
The debug build includes `out/libfencepost-testhooks.dylib` for testing trace records, retired queue signals,
pipeline retries and simulated feedback errors. `mtl4frames --long` exercises frame chains over 512 frames.

## Results so far

**Metal 4 on arm64 and Rosetta (0.3.0).** All 61 race verdicts passed three times per slice.
The detector caught the missing barrier, wrong-stage barrier and premature event signal.
The frame workload and the 10,000-draw benchmark produced zero race reports.

### Overhead

Median CPU times from three runs per configuration on Apple silicon. The draw workload changes a buffer address and
a texture binding on every draw. The commit workload encodes four draws per submission. Pipeline builds use a warm
shader cache. The x86_64 slice runs under Rosetta.

| Slice | CPU measure | Without fencepost | With fencepost | Overhead |
|---|---|---:|---:|---:|
| arm64 | µs/draw | 0.0861 | 0.1057 | +23% |
| arm64 | µs/commit | 4.9583 | 8.3750 | +69% |
| arm64 | ms/pipeline (MTL4Compiler) | 0.0192 | 0.0219 | +14% |
| arm64 | ms/pipeline (device) | 0.0132 | 0.0152 | +15% |
| x86_64 | µs/draw | 0.1391 | 0.1812 | +30% |
| x86_64 | µs/commit | 8.0830 | 13.0840 | +62% |
| x86_64 | ms/pipeline (MTL4Compiler) | 0.0390 | 0.0444 | +14% |
| x86_64 | ms/pipeline (device) | 0.0319 | 0.0351 | +10% |

### Earlier Metal 3 results

- **Our D3D12 runtime, 14 test programs:** 5,300 declared barriers checked, 0 reports, the same check results with and
  without fencepost.
- **The ROTTR flashing-polygon bug reintroduced** (`f6-root-wait = 0`, "encoders after a sync point wait for
  nothing"): `d3d12-parallel` still passes all 32 of its own checks, while fencepost reports the barrier violation (16
  encoders, "waits: nothing"). `d3d12-m2` shows a blit-to-render read-after-write race that the fix removes. The bug
  took a night of frame captures to find in the game; here it shows in a 0.3-second test run.
- **DXMT (D3D11) cube demo:** 1,127 encoders and 2,787 resource accesses checked, 0 reports.
- **The Last of Us Part I through our D3D12 runtime:** 2.1 million encoders and 3.6 million declared barriers in one
  session, 0 reports.
- **A new bug caught in a game:** an experiment let command lists in one D3D12 submission overlap on the GPU. The run
  was faster, black blotches appeared on a wall, and fencepost reported 23 encoders not ordered after their barrier,
  naming the passes (ambient shadows, then their blur). A barrier at the end of one list must still hold for the next
  list, and the change had dropped it. Fixed the same evening; once fixed, most of the speed-up went away.


## Human Input
This was originally made in my testing suite for porting madeira over to macos as I do like their d3d12 implementation.  Ran into a few snags of course and in turn this tool plus some others help me get past those while picking up significant performance gains along the way.  Thought it would help the vibecoders out a little bit.
  
## Licence

MIT, see [LICENSE](LICENSE).

Written with AI assistance (Claude); the commits say so too.
