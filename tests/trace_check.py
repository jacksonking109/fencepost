#!/usr/bin/env python3
"""Validate fencepost's Chrome JSON, with optional exact checks for tracetest."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path


ENCODERS = {"blit", "compute", "render"}
FLOW_CATEGORIES = {"fence", "event", "completed", "race", "order"}
MARKER_PREFIXES = {
    "RACE ": "race",
    "possible race ": "possible",
    "ORDER barrier not honoured": "order",
    "ORDER fragment-only": "partial",
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


def check_shape(encoders, slices, flows, markers, frames, selected, summary):
    candidates = {
        "fence update": 1, "fence wait": 1, "racing write": 1, "event wait": 2,
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


def check_trace(events, args):
    assert len({event["pid"] for event in events}) == 1
    summaries = [event for event in events if event["name"] == "fencepost summary"]
    assert len(summaries) == 1 and events[-1] is summaries[0]
    summary = summaries[0]["args"]
    timestamps = [event["ts"] for event in events if "ts" in event]
    assert timestamps[0] == 0 and all(isinstance(ts, Decimal) and ts >= 0 for ts in timestamps)

    track_names = {event["tid"]: event["args"]["name"] for event in events if event["name"] == "thread_name"}
    slices = [event for event in events if event["ph"] == "X"]
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
            assert ("(CPU encode time)" in track_names[tid]) == (not data["gpu_time"])
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
    command_buffers = [event for event in slices if event["cat"] == "command buffer"]
    for start, finish in flows:
        assert start["cat"] == finish["cat"] and start["args"] == finish["args"]
        source = (
            [event for event in command_buffers if event["tid"] == start["tid"]]
            if start["cat"] == "event" else [encoders[start["args"]["from_uid"]]]
        )
        assert any(
            start["tid"] == event["tid"] and event["ts"] + event["dur"] - start["ts"] == Decimal("0.001")
            for event in source
        ), start
        target = encoders[finish["args"]["to_uid"]]
        assert (finish["tid"], finish["ts"]) == (target["tid"], target["ts"]), finish

    markers = [event for event in events if event["ph"] == "i" and marker_kind(event)]
    counts = Counter(marker_kind(event) for event in markers)
    assert len(markers) == summary["markers"] <= 20000
    for marker in markers:
        data = marker["args"]
        assert marker["s"] == "t" and all(data[k] for k in ("resource", "first", "second", "why"))
        target = encoders[data["to_uid"]]
        assert (marker["tid"], marker["ts"]) == (target["tid"], target["ts"])
    missing = 0
    for kind in MARKER_PREFIXES.values():
        assert counts[kind] == summary[f"{kind}_markers"]
        scoped = summary[f"{kind}_marker_hits"]
        assert counts[kind] <= scoped <= summary[f"{kind}_hits"]
        if not args.frames:
            assert scoped == summary[f"{kind}_hits"]
        missing += scoped - counts[kind]
    assert missing <= summary["dropped"]
    if not summary["dropped"]:
        assert not missing

    if args.frames:
        first, last = args.frames
        assert all(first <= event["args"]["frame"] <= last for event in (*encoders.values(), *frames))
    if args.shape:
        first, last = args.frames or (1, 2)
        check_shape(encoders, slices, flows, markers, frames, set(range(first, last + 1)) & {1, 2}, summary)
    print(f"{args.trace}: {len(slices)} X slices, {len(encoders)} encoders ({summary['sampled']} GPU), "
          f"{len(flows)} flows, {len(markers)} markers, {summary['dropped']} dropped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--shape", action="store_true", help="check tracetest's exact encoders, flows and marker")
    parser.add_argument("--frames", metavar="A:B", help="expect only this inclusive capture range")
    parser.add_argument("--cpu-only", action="store_true", help="expect CPU encode timing for every encoder")
    args = parser.parse_args()
    if args.frames:
        args.frames = tuple(map(int, args.frames.split(":")))
        if len(args.frames) != 2 or not 1 <= args.frames[0] <= args.frames[1]:
            parser.error("--frames must be a 1-based inclusive A:B range")
    check_trace(load_trace(args.trace), args)


if __name__ == "__main__":
    main()
