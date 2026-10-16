# fencepost

A race detector for Metal GPU synchronisation. Inject it into any Metal program and it reports two encoders that
touch the same memory with nothing ordering them: the missing `MTLFence`.

```
DYLD_INSERT_LIBRARIES=/path/to/libfencepost.dylib  your-metal-program
```

```
fencepost: RACE read-after-write on texture 64x64, pixel format 70, 1 levels 0x7ff6cb75e1c0, level 0, slice 0
  first : blit encoder #18 (command buffer #8, encoder 1, queue 1) writes it: blit destination
  second: render encoder #19 (command buffer #8, encoder 2, queue 1) reads it: useResource, vertex stage
  why   : both are in command buffer #8 and no fence orders them; untracked resources need updateFence in the
          first and waitForFence in the second
```

## Why

Metal orders GPU work for you only on resources it tracks. Translation layers (DXMT, MoltenVK, D3D12 runtimes) and
most engines create resources with `MTLHazardTrackingModeUntracked`, or reach them through argument buffers Metal does
not track, because tracking costs CPU time. Then ordering is the program's job, through `updateFence`/`waitForFence`
and `MTLEvent` signals and waits. Forget one and two passes overlap on the GPU: a read sees half-written data, and a
frame flickers. The bug comes and goes with timing, it often disappears under a frame capture (captures serialise the
GPU), and Metal's API validation does not report it. Vulkan has a synchronisation validation layer for this class of
bug. As far as we could find, Metal has nothing like it.

fencepost judges ordering the way ThreadSanitizer does for CPU threads: from the synchronisation the program
*encoded*, not from when the GPU happened to run things. So it reports a race even on a run where the race did not
show.

## What it checks

**Resource races.** For each untracked texture or buffer, fencepost records which encoder reads or writes it, through:
- render pass attachments (load, store, MSAA resolve, visibility buffer);
- `useResource` (with its usage and stages);
- vertex/fragment/compute bindings;
- index and indirect buffers;
- blits, with exact byte ranges, mip levels and slices.

Texture views are mapped onto the storage they share, so two views of different mip levels do not conflict. Two
conflicting accesses (at least one a write) whose encoders have no ordering path between them are a race.

**Declared barriers.** A runtime that reaches resources only through argument buffers or residency sets (bindless
D3D12, for one) gives fencepost nothing to see. Such a runtime can still say what ordering it *means*:

```c
void (*fp_barrier)(id) = dlsym(RTLD_DEFAULT, "fencepost_barrier");   /* NULL when fencepost is not injected */
if (fp_barrier) fp_barrier(commandBuffer);   /* at a D3D12 ResourceBarrier, a Vulkan pipeline barrier, ... */
```

Every encoder submitted to that queue before the call must then be ordered before every encoder after it. fencepost
checks the encoded fences and events against that, resource by resource or not at all. A render encoder that waits
only before its fragment stage is reported separately: its vertex stage can still overlap.

An encoder the runtime orders by other means (a present blit that touches only tracked resources, say) can be left out
of these checks as an earlier encoder with `fencepost_exempt(encoder)`, called while it is open. Its resource accesses
are still checked.

**How parallel each frame is.** Too much synchronisation costs as well: encoders chained one after another leave the
GPU nothing to overlap. For every frame (a `presentDrawable`), fencepost counts the encoders and the longest chain of
them the encoded fences and events force to run in order, and prints the medians with the running summary:

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

## How it works

- **Nodes:** every encoder is one. A render encoder has two stages, vertex and fragment.
- **Edges** order nodes:
  - `updateFence:afterStages:` paired with `waitForFence:beforeStages:`. Pairs are resolved in submission order, so a
    wait meets the latest update `enqueue`d before it.
  - An event signal paired with a wait for that value or less.
  - A command buffer that completed before another was committed.
- **Ancestor sets:** for each node and stage, fencepost keeps a bitset of everything ordered before it, over a sliding
  window of 4096 nodes. A query is one bit test.
- **When analysis runs:** at `commit`. A node whose event wait is not yet signalled waits until it is, or until the CPU
  sets the value, or for 256 commits.
- **Tracked resources** are skipped: Metal orders those itself.

## Limits

These are reported in the summary as blind spots where they occur.
- **Argument buffers and residency sets:** resources reached only through them are invisible. Declared barriers cover
  the ordering, not the data.
- **`useHeap`:** the resources inside a heap are not seen. Aliasing between resources placed in the same heap memory
  is not modelled.
- **Direct compute bindings** count as reads: whether a kernel writes them needs its pipeline's reflection, and
  translation layers declare their writes with `useResource` (`FENCEPOST_BIND_MAYWRITE=1` changes this).
- **Read+write `useResource`:** a kernel may or may not write the resource. Conflicts that rest only on such a "may
  write" are reported as *possible*, not as races.
- **Not followed:** parallel render encoders and Metal 4 (`MTL4CommandQueue`) encoders.
- **Event signals** are taken to follow all earlier work on their queue.
- **Retired work:** a command buffer that completed before the next commit is treated as ordered. A race whose window
  closed that way on this run is missed.
- **Frame chains** follow fences and events only. Ordering Metal adds itself between encoders using tracked resources
  is not counted, so parallelism can read higher than the GPU really gets. "Run out of work" is a lower bound:
  completion handlers run a little after the GPU finishes.

## Knobs (environment)

| Variable | Default | |
|---|---|---|
| `FENCEPOST_LOG` | stderr | append reports to this file; `%p` in the name becomes the process id |
| `FENCEPOST_MAX_REPORTS` | 200 | detailed reports printed; the rest are counted |
| `FENCEPOST_POSSIBLE` | 1 | 0 hides "possible" reports |
| `FENCEPOST_SUMMARY_SECS` | 10 | print a running summary at most this often when the counts change; 0 = only at exit |
| `FENCEPOST_CHAIN` | 0 | list the longest ordered chain of this frame (counted from 1) |
| `FENCEPOST_BIND_MAYWRITE` | 0 | 1: direct compute bindings of writable resources count as "may write", not reads |

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
tar -xzf fencepost-0.1.1-macos.tar.gz
xattr -dr com.apple.quarantine fencepost-0.1.1-macos
DYLD_INSERT_LIBRARIES=$PWD/fencepost-0.1.1-macos/libfencepost.dylib  your-metal-program
```

The library is signed ad hoc, not by an Apple developer ID; building it yourself (below) takes seconds.

## Build and test

Needs macOS 13 or later and Apple's command-line tools (`xcode-select --install`; full Xcode is not required).

```
git clone https://github.com/jacksonking109/fencepost && cd fencepost
./build.sh                                                   # universal arm64 + x86_64, signed ad hoc
./build.sh release 0.1.0                                     # also a release archive, without debug information
DYLD_INSERT_LIBRARIES=$PWD/out/libfencepost.dylib out/fptest
arch -x86_64 -e DYLD_INSERT_LIBRARIES=$PWD/out/libfencepost.dylib out/fptest   # the Rosetta slice
```

`tests/fptest.m` holds 34 small Metal programs with and without the synchronisation they need, each with the number of
races, barrier violations or frame chain lengths fencepost must report. Examples:
- a fence updated after the vertex stage while the write happens in the fragment stage;
- a fence re-updated by an unrelated encoder before the wait;
- a wait committed before its signal on another queue;
- a fence that must be resolved in enqueue order, not commit order;
- mip levels written and read through views;
- a fence wait that crosses a frame boundary, which must not lengthen the new frame's chain.

## Results so far

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
