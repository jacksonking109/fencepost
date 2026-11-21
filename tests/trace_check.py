#!/usr/bin/env python3
"""Validate fencepost's Chrome JSON, with optional exact checks for tracetest."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path


ENCODERS = {"blit", "compute", "render", "machine learning"}
FLOW_CATEGORIES = {"fence", "event", "completed", "race", "order"}
MARKER_PREFIXES = {
    "RACE ": "race",
    "possible race ": "possible",
    "ORDER barrier not honoured": "order",
    "ORDER fragment-only": "partial",
    "ORDER stage partial": "partial",
}


def load_trace(path):
    body = path.read_text(encoding="utf-8").rstrip()
    # The writer prefixes each event with its separator. EOF between records
    # only needs the closing bracket, and the file must parse both ways.
    prefix = body[:-1] if body.endswith("]") else body
    before_close = json.loads(prefix + "\n]", parse_float=Decimal)
    if body.endswith("]"):
        assert before_close == json.loads(body, parse_float=Decimal)
    assert isinstance(before_close, list)
    return before_close


def marker_kind(event):
    return next((kind for prefix, kind in MARKER_PREFIXES.items() if event["name"].startswith(prefix)), None)


def slice_end(event):
    return event["ts"] + event["dur"]


def check_shape(encoders, slices, flows, markers, frames, selected, summary):
    candidates = {
        "fence update": 1, "empty blit": 1, "fence wait": 1, "racing write": 1, "event wait": 2,
    }
    want = {name: frame for name, frame in candidates.items() if frame in selected}
    by_name = {event["name"]: event for event in encoders.values()}
    assert {name: event["args"]["frame"] for name, event in by_name.items()} == want
    assert {frame["args"]["frame"]: frame["args"]["encoders"] for frame in frames} == {
        frame: sum(f == frame for f in want.values()) for frame in selected
    }
    assert {frame["args"]["frame"]: frame["args"]["longest_chain"] for frame in frames} == {
        frame: {1: 2, 2: 1}[frame] for frame in selected
    }
    command_buffers = [event for event in slices if event["cat"] == "command buffer"]
    assert {event["name"] for event in command_buffers} == {
        {1: "CB #1 produce", 2: "CB #2 consume"}[frame] for frame in selected
    }

    def uid(name):
        return by_name[name]["args"]["uid"]

    want_flows = []
    if 1 in selected:
        want_flows.extend((
            ("fence", uid("fence update"), uid("fence wait")),
            ("race", uid("fence wait"), uid("racing write")),
        ))
        assert by_name["fence wait"]["args"]["fence_waits"][0]["last_updater_uid"] == uid("fence update")
        produce = next(event for event in command_buffers if event["name"] == "CB #1 produce")
        empty = by_name["empty blit"]
        before = by_name["fence update"]
        assert empty["args"]["estimated"] is True and empty["args"]["cb"] == produce["args"]["uid"]
        assert empty["ts"] == max(produce["ts"], min(slice_end(before), slice_end(produce)))
        following = (by_name[name] for name in ("fence wait", "racing write"))
        next_start = next((event["ts"] for event in following if event["args"]["gpu_time"]), slice_end(produce))
        limit = min(next_start, slice_end(produce))
        assert slice_end(empty) == max(empty["ts"], min(empty["ts"] + 1, limit))
        assert len(markers) == 1
        marker = markers[0]
        assert marker["name"] == "RACE write-after-write"
        assert marker["args"]["to_uid"] == uid("racing write")
        assert '"raced \\ buffer"' in marker["args"]["resource"]
        assert '"fence wait"' in marker["args"]["first"] and '"racing write"' in marker["args"]["second"]
        assert "no fence orders them" in marker["args"]["why"]
    else:
        assert not markers
    if selected == {1, 2}:
        signals = [start for start, _ in flows if start["cat"] == "event"]
        assert len(signals) == 1
        want_flows.append(("event", signals[0]["args"]["from_uid"], uid("event wait")))
        produce = next(event for event in command_buffers if event["name"] == "CB #1 produce")
        assert signals[0]["tid"] == produce["tid"]
    got_flows = [(start["cat"], start["args"]["from_uid"], start["args"]["to_uid"]) for start, _ in flows]
    assert sorted(got_flows) == sorted(want_flows), (got_flows, want_flows)
    if summary["sampling_supported"] and selected == {1, 2}:
        assert summary["sampled"] > len(encoders) / 2, "expected GPU timing for most workload encoders"


def check_mtl4_timing(case, encoders, command_buffers, markers, summary):
    spans = list(encoders.values())
    if case in ("ordered", "overlap", "disabled"):
        spans = sorted((e for e in spans if e["args"]["cb_label"] == "timing dispatches"),
                       key=lambda e: e["args"]["uid"])
        assert [e["name"] for e in spans] == ["MTL4 timing 1x", "MTL4 timing 4x", "MTL4 timing 16x"]
        commit, = {e["args"]["cb"] for e in spans}
        assert command_buffers[commit]["dur"] > 0
        if case == "disabled":
            assert all(e["args"].get("estimated") for e in spans)
            assert summary["sampled"] == summary["sampling_skipped"] == summary["pool_exhausted"] == 0
            return
        assert all(e["args"]["gpu_time"] for e in spans), spans
        if case == "ordered":
            first, middle, last = spans
            assert slice_end(first) <= middle["ts"] and slice_end(middle) <= last["ts"], spans
            # Spans measure elapsed GPU time. The clock can vary about 9× even within
            # a commit, and a rare late Relaxed start shortens a span. Check the
            # longest workload's rank without assuming fixed duration ratios.
            assert all(e["dur"] > 0 for e in spans), spans
            assert last["dur"] > max(first["dur"], middle["dur"]), spans
        else:
            by_time = sorted(spans, key=lambda e: e["ts"])
            overlapping = [(a, b) for i, a in enumerate(by_time) for b in by_time[i + 1:] if b["ts"] < slice_end(a)]
            assert overlapping and all(a["tid"] != b["tid"] for a, b in overlapping), spans
        print("encoder durations in µs", *(e["dur"] for e in spans))
    elif case == "render":
        render, = (e for e in spans if e["cat"] == "render")
        assert render["args"]["gpu_time"] and render["args"]["cb"] in command_buffers, render
        print(f"render duration {render['dur']} µs")
    elif case == "race":
        marker, = markers
        assert type(marker["args"]["overlapped_on_gpu"]) is bool
        assert marker["args"]["gpu_order"] in ("overlapped", "in order", "reversed"), marker
        print(f"race GPU order {marker['args']['gpu_order']}")
    else:
        batch = sorted((e for e in spans if e["args"]["cb_label"] == "timing overflow"),
                       key=lambda e: e["args"]["uid"])
        assert len(batch) == 2052 and len({e["args"]["cb"] for e in batch}) == 1
        assert len({e["args"]["logical_cb"] for e in batch}) == 4
        assert all(e["args"].get("estimated") for e in batch[2048:])
        assert all(e["args"]["gpu_time"] for e in batch[:2048])
        assert summary["pool_exhausted"] == 4 and summary["sampling_skipped"] == 5, summary
        empty, = (e for e in spans if e["name"] == "MTL4 timing empty")
        reused, = (e for e in spans if e["name"] == "MTL4 timing reused pair")
        assert empty["args"].get("estimated") and reused["args"]["gpu_time"]


def check_trace(events, args):
    assert len({event["pid"] for event in events}) == 1
    summaries = [event for event in events if event["name"] == "fencepost summary"]
    assert len(summaries) == 1 and events[-1] is summaries[0]
    summary = summaries[0]["args"]
    timestamps = [event["ts"] for event in events if "ts" in event]
    assert timestamps[0] == 0 and all(isinstance(ts, Decimal) and ts >= 0 for ts in timestamps)

    track_names = {event["tid"]: event["args"]["name"] for event in events if event["name"] == "thread_name"}
    slices = [event for event in events if event["ph"] == "X"]
    command_buffers = {event["args"]["uid"]: event for event in slices if event["cat"] == "command buffer"}
    standalone_signals = {event["args"]["uid"]: event for event in slices if event["cat"] == "event signal"}

    def cpu_domain(event):
        return "(CPU encode time)" in track_names[event["tid"]]

    intervals = defaultdict(list)
    encoders = {}
    frames = []
    for event in slices:
        assert isinstance(event["dur"], Decimal) and event["dur"] >= 0
        tid, start = event["tid"], event["ts"]
        intervals[tid].append((start, start + event["dur"]))
        assert tid in track_names
        if event["cat"] in ENCODERS:
            data = event["args"]
            assert {"uid", "kind", "label", "cb", "cb_label", "queue", "frame", "chain_depth", "accesses",
                    "fence_waits", "fence_updates", "event_waits", "after_declared_barrier", "gpu_time"} <= data.keys()
            assert data["kind"] == event["cat"] and data["uid"] not in encoders
            assert type(data["gpu_time"]) is bool
            estimated = data.get("estimated", False)
            assert type(estimated) is bool and not (estimated and data["gpu_time"])
            assert cpu_domain(event) == (not (estimated or data["gpu_time"]))
            if data["cb"] in command_buffers:
                assert estimated or data["gpu_time"]
                if data["gpu_time"] and "logical_cb" in data:
                    cb = command_buffers[data["cb"]]
                    assert cb["ts"] <= start < slice_end(event) <= slice_end(cb)
            if estimated and (not args.dump or data["cb"] in command_buffers):
                cb = command_buffers[data["cb"]]
                assert cb["ts"] <= start <= slice_end(event) <= slice_end(cb)
                assert event["dur"] <= 1
            encoders[data["uid"]] = event
        if event["cat"] == "frame":
            frames.append(event)
    for tid, spans in intervals.items():
        spans.sort()
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), track_names[tid]
    assert len(encoders) == summary["encoders_traced"]
    assert sum(event["args"]["gpu_time"] for event in encoders.values()) == summary["sampled"]
    if args.cpu_only:
        assert summary["sampled"] == 0

    starts, finishes = {}, {}
    for event in events:
        if event["ph"] not in ("s", "f"):
            continue
        group = starts if event["ph"] == "s" else finishes
        assert event["id"] not in group and event["bp"] == "e"
        assert event["cat"] in FLOW_CATEGORIES
        group[event["id"]] = event
    assert starts.keys() == finishes.keys() and len(starts) == summary["flows"]
    flows = [(start, finishes[flow_id]) for flow_id, start in starts.items()]
    completed = set()
    backward = Counter()
    for start, finish in flows:
        assert start["cat"] == finish["cat"] and start["args"] == finish["args"]
        cat = start["cat"]
        first, second = start["args"]["from_uid"], start["args"]["to_uid"]
        if cat == "completed":
            assert (first, second) not in completed
            completed.add((first, second))
            source, target = command_buffers[first], command_buffers[second]
        else:
            target = encoders[second]
            if cat == "event":
                if first in standalone_signals:
                    source = standalone_signals[first]
                    assert source["args"]["estimated"] is True
                else:
                    matches = [event for event in command_buffers.values()
                               if event["tid"] == start["tid"] and event["ts"] <= start["ts"] < slice_end(event)]
                    assert len(matches) == 1, start
                    source = matches[0]
            else:
                source = encoders[first]
        assert cpu_domain(start) == cpu_domain(finish), (start, finish)
        expected_start = slice_end(source) - min(source["dur"], Decimal("0.001"))
        if (cat in ("fence", "event") and source["ts"] <= target["ts"] < expected_start
                and (source["args"].get("estimated") or target["args"].get("estimated"))):
            expected_start = target["ts"]
        assert (start["tid"], start["ts"]) == (source["tid"], expected_start), start
        assert (finish["tid"], finish["ts"]) == (target["tid"], target["ts"]), finish
        if start["ts"] > finish["ts"]:
            backward[cat] += 1
            if cat in ("fence", "event"):
                assert cat == "fence" and source["cat"] == target["cat"] == "render", (start, finish)
                assert any(wait["last_updater_uid"] == first and wait["stage"] == "fragment"
                           for wait in target["args"]["fence_waits"]), (start, finish)

    markers = [event for event in events if event["ph"] == "i" and marker_kind(event)]
    counts = Counter(marker_kind(event) for event in markers)
    assert len(markers) == summary["markers"]
    assert args.dump or len(markers) <= 20000
    for marker in markers:
        data = marker["args"]
        assert marker["s"] == "t" and all(data[k] for k in ("resource", "first", "second", "why"))
        target = encoders[data["to_uid"]]
        assert (marker["tid"], marker["ts"]) == (target["tid"], target["ts"])
        source = encoders.get(data["from_uid"])
        overlap = None
        if source and source["args"]["gpu_time"] and target["args"]["gpu_time"]:
            overlap = source["ts"] < slice_end(target) and target["ts"] < slice_end(source)
        assert data["overlapped_on_gpu"] is overlap, marker
        order = None
        if overlap is not None:
            order = "overlapped" if overlap else "reversed" if slice_end(target) <= source["ts"] else "in order"
        assert data["gpu_order"] == order, marker
    missing = 0
    for kind in MARKER_PREFIXES.values():
        assert counts[kind] == summary[f"{kind}_markers"]
        scoped = summary[f"{kind}_marker_hits"]
        assert counts[kind] <= scoped <= summary[f"{kind}_hits"]
        if not args.frames and not args.dump:
            assert scoped == summary[f"{kind}_hits"]
        missing += scoped - counts[kind]
    assert missing <= summary["dropped"]
    if not summary["dropped"]:
        assert not missing

    if args.frames:
        first, last = args.frames
        assert all(first <= event["args"]["frame"] <= last for event in (*encoders.values(), *frames))
    if args.dump:
        assert args.trace.read_text().rstrip().endswith("]")
        triggers = [e for e in events if e["name"].startswith("TRIGGER ")]
        assert len(triggers) == 1
        trigger = triggers[0]
        data = trigger["args"]
        assert trigger["ph"] == "i" and trigger["s"] == "t" and track_names[trigger["tid"]] == "Frames"
        for frame in frames:
            if frame["args"]["frame"] == data["frame"]:
                assert frame["ts"] <= trigger["ts"] <= slice_end(frame)
        assert trigger["name"] == "TRIGGER " + data["reason"] and data["source"]
        assert args.trace.name.endswith(f"-{trigger['pid']}-f{data['frame']}-{data['reason']}.json")
        if args.trigger:
            assert (data["frame"], data["reason"]) == (int(args.trigger[0]), args.trigger[1])
        if args.frames:
            assert sorted(f["args"]["frame"] for f in frames) == list(range(first, last + 1))
        assert all(m["args"]["from_uid"] in encoders for m in markers)
    if args.shape:
        first, last = args.frames or (1, 2)
        check_shape(encoders, slices, flows, markers, frames, set(range(first, last + 1)) & {1, 2}, summary)
    if args.mtl4_timing:
        check_mtl4_timing(args.mtl4_timing, encoders, command_buffers, markers, summary)
    print(f"{args.trace}: {len(slices)} X slices, {len(encoders)} encoders ({summary['sampled']} GPU), "
          f"{len(flows)} flows ({len(completed)} completed, {sum(backward.values())} backward), "
          f"{len(markers)} markers, {summary['dropped']} dropped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--shape", action="store_true", help="check tracetest's exact encoders, flows and marker")
    parser.add_argument("--frames", metavar="A:B", help="expect only this inclusive capture range")
    parser.add_argument("--cpu-only", action="store_true", help="expect no sampled GPU encoder times")
    parser.add_argument("--mtl4-timing", choices=("ordered", "overlap", "render", "race", "disabled", "overflow"),
                        help="check a Metal 4 timing case")
    parser.add_argument("--dump", action="store_true", help="check a standalone triggered capture")
    parser.add_argument("--trigger", nargs=2, metavar=("FRAME", "REASON"), help="expect this trigger in the dump")
    args = parser.parse_args()
    if args.frames:
        args.frames = tuple(map(int, args.frames.split(":")))
        if len(args.frames) != 2 or not 1 <= args.frames[0] <= args.frames[1]:
            parser.error("--frames must be a 1-based inclusive A:B range")
    check_trace(load_trace(args.trace), args)


if __name__ == "__main__":
    main()
