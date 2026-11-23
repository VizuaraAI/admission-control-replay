"""Hold-time re-scoring of finished replay runs, offline and with the standard library only.

For every request in a cell it computes what a user waits through:
  door_wait_s  time in the proxy queue, t_send - t_arr (for a request never sent: until it left the queue or the cell ended)
  pauses_s     the gaps in its output after the first token that evictions cause: from a preemption to the moment
               the request generates again, estimated as its resume (the prefix-cache lookup in preempt.jsonl) plus
               the rebuild prefill (rebuilt tokens / PF), capped at its last token. An eviction that comes before
               the request gained a token since its return (same token count in the log) continues the same pause.
               Only pauses longer than PAUSE_MIN_S count.
  hold_s       door_wait_s + sum(pauses_s)
  hold_incl_engine_queue_s   the same with the wait to the first token instead of the door wait, because stock vLLM
               queues requests inside the engine rather than at the door, where door_wait_s does not see it
and per cell: evictions per request, median/p90/p99 hold, and goodput with a hold limit (a request is good when it
finished and its hold is at most HOLD_LIMIT_S). Output follows the AdmitPerf draft schemas admitperf.run/1 and
admitperf.request/1, one JSON line each, times in seconds from the cell start.

    python -I harness/rescore.py runs/exp02                       all cells, every request
    python -I harness/rescore.py runs/exp02 --length-band 2000:4000   only requests whose trace output is in the band
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure  # noqa: E402

PAUSE_MIN_S = 1.0
HOLD_LIMIT_S = 5.0
LENGTH_BUCKETS = [(0, 4000), (4000, 9000), (9000, 16001)]

STATUS = {"finished": "ok", "rejected_by_door": "rejected", "abandoned_in_proxy": "abandoned",
          "abandoned_in_engine": "abandoned", "short": "aborted", "error": "error"}


def status(state):
    return STATUS.get(state, "censored")     # censored, censored_in_proxy, queued, sent: still waiting when the cell ended


def eviction_pauses(plog, cell_name, t0, t_end, PF):
    """rid -> (pauses, evictions): pauses is a list of (t_preempt, t_back) in cell time, t_back None for a request never
    resumed; evictions counts every preemption."""
    by_rid = {}
    for e in plog:
        if e.get("ev") not in ("preempt", "lookup") or not (t0 <= e["t"] <= t_end + 5):
            continue
        m = measure.RID.match(e["rid"])
        if not m or not m.group(1).startswith(cell_name + "-r"):
            continue
        by_rid.setdefault(m.group(1), []).append(e)
    out = {}
    for rid, evs in by_rid.items():
        evs.sort(key=lambda e: e["t"])
        spans, open_t, back_tokens = [], None, None
        for e in evs:
            if e["ev"] == "preempt":
                if open_t is not None:
                    spans.append([open_t, None])
                if spans and back_tokens is not None and e["tokens"] <= back_tokens and spans[-1][1] is not None:
                    open_t = spans.pop()[0]     # evicted again before a new token: the same pause goes on
                else:
                    open_t = e["t"] - t0
                back_tokens = None
            elif open_t is not None:
                rebuilt = max(0, e["tokens"] - e["hit"])
                spans.append([open_t, e["t"] - t0 + rebuilt / PF])
                open_t, back_tokens = None, e["tokens"]
        if open_t is not None:
            spans.append([open_t, None])
        out[rid] = ([tuple(x) for x in spans], sum(1 for e in evs if e["ev"] == "preempt"))
    return out


def request_record(r, spans, n_evictions, run_id, arm, t_cell_end):
    st = status(r["state"])
    if r["t_send"] is not None:
        door = r["t_send"] - r["t_arr"]
    elif r["state"] == "abandoned_in_proxy":
        door = measure.TTFT_LIMIT           # the client gives up at the TTFT limit
    elif r["state"] == "rejected_by_door":
        door = 0.0                          # refused on arrival
    else:
        door = t_cell_end - r["t_arr"]      # still in the proxy queue when the cell ended
    if r["t_first"] is not None:
        first = r["t_first"] - r["t_arr"]
    elif r["state"].startswith("abandoned"):
        first = measure.TTFT_LIMIT
    elif r["state"] == "rejected_by_door":
        first = 0.0
    else:
        first = t_cell_end - r["t_arr"]
    merged = []                             # an eviction before the request is generating again extends the same pause
    for a, b in sorted(spans):
        end = t_cell_end if b is None else b
        if r["t_first"] is None or end <= r["t_first"]:
            continue                        # before the first token the user is still waiting for it: see first_wait_s
        a = max(a, r["t_first"])
        if r["t_last"] is not None and r["t_last"] > a:
            end = min(end, r["t_last"])
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([a, end])
    pauses = [round(b - a, 4) for a, b in merged if b - a > PAUSE_MIN_S]
    hold = round(door + sum(pauses), 4)
    return dict(schema="admitperf.request/1", run_id=run_id, arm=arm, oracle=(arm == "known"), request_id=r["rid"],
                t_arr=r["t_arr"], t_send=r["t_send"], t_first_token=r["t_first"],
                t_done=r["t_last"] if r["state"] == "finished" else None, prompt_tokens=r["p"], output_tokens=r["n"],
                trace_output_tokens=r["L"], evictions=n_evictions, pauses_s=pauses,
                door_wait_s=round(door, 4), hold_s=hold, status=st,
                first_wait_s=round(first, 4), hold_incl_engine_queue_s=round(first + sum(pauses), 4),
                max_gap_s=r.get("max_gap"))


def pct(xs, q):
    return measure._pct(xs, q)


def summarize(recs, count_window_s, hold_limit_s=HOLD_LIMIT_S):
    """Measures over the requests that arrived in the count window. Hold percentiles leave out censored requests (still
    waiting when the cell ended) and rejected ones (a hold of 0 s would flatter a gate that refuses); abandoned requests
    stay in with their 30 s. Rejected, abandoned and censored are reported as their own rates and are never good."""
    win = [x for x in recs if x["t_arr"] < count_window_s]
    held = [x for x in win if x["status"] not in ("censored", "rejected")]
    holds = [x["hold_s"] for x in held]
    full = [x["hold_incl_engine_queue_s"] for x in held]
    n = max(1, len(win))
    ok = [x for x in win if x["status"] == "ok"]
    good = [x for x in ok if x["hold_s"] <= hold_limit_s]
    evicted = [x for x in win if x["evictions"] > 0]
    all_pauses = [p for x in win for p in x["pauses_s"]]
    unexplained = [x for x in ok if (x["max_gap_s"] or 0) > PAUSE_MIN_S and not x["pauses_s"]]
    return dict(counted=len(win), finished=len(ok),
                refused=sum(1 for x in win if x["status"] in ("rejected", "abandoned")),
                rejected_rate=sum(1 for x in win if x["status"] == "rejected") / n,
                abandoned_rate=sum(1 for x in win if x["status"] == "abandoned") / n,
                censored=sum(1 for x in win if x["status"] == "censored"),
                evictions=sum(x["evictions"] for x in win),
                evictions_per_request=sum(x["evictions"] for x in win) / max(1, len(win)),
                evicted_share=len(evicted) / max(1, len(win)),
                hold_p50=pct(holds, 50), hold_p90=pct(holds, 90), hold_p99=pct(holds, 99),
                hold_full_p50=pct(full, 50), hold_full_p90=pct(full, 90), hold_full_p99=pct(full, 99),
                good_full=sum(1 for x in ok if x["hold_incl_engine_queue_s"] <= hold_limit_s),
                door_wait_p90=pct([x["door_wait_s"] for x in win if x["door_wait_s"] is not None], 90),
                pause_p90=pct(all_pauses, 90), pause_max=max(all_pauses) if all_pauses else 0.0,
                hold_limit_s=hold_limit_s, good=len(good), goodput_rps=len(good) / count_window_s,
                good_share=len(good) / max(1, len(win)),
                gaps_without_eviction=len(unexplained))


def by_length(recs, count_window_s, buckets=LENGTH_BUCKETS):
    return [dict(L_from=a, L_to=b, **summarize([x for x in recs if a <= x["trace_output_tokens"] < b], count_window_s))
            for a, b in buckets]


def rescore_cell(cell, plog, run_meta=None, length_band=None):
    cfg = cell["cfg"]
    name, t0, t_end = cfg["cell"], cell["t0"], cell["t_end"]
    run_id = cfg.get("experiment", "run") + "/" + name
    spans = eviction_pauses(plog, name, t0, t_end, cfg["PF"])
    recs = [request_record(r, *spans.get(r["rid"], ([], 0)), run_id, cfg["arm"], t_end - t0) for r in cell["requests"]]
    if length_band:
        lo, hi = length_band
        recs = [x for x in recs if lo <= x["trace_output_tokens"] <= hi]
    meta = run_meta or {}
    header = dict(schema="admitperf.run/1", run_id=run_id, engine=meta.get("engine", "vllm"),
                  engine_version=meta.get("engine_version"), model=meta.get("model"), gpu=cfg.get("gpu"),
                  kv_cap_tokens=cfg.get("C"), load_x=cfg.get("load"), seed=cfg.get("arrival_seed"), trace=cfg.get("trace"),
                  prefix_cache=cfg.get("prefix_caching"), hold_limit_s=HOLD_LIMIT_S,
                  length_band=list(length_band) if length_band else None)
    cw = float(cfg.get("count_window_s", measure.COUNT_WINDOW_S))
    summary = dict(cell=name, arm=cfg["arm"], door=cfg.get("door", False), load=cfg.get("load"),
                   **summarize(recs, cw), by_length=by_length(recs, cw))
    return header, recs, summary


def find_cells(root):
    return sorted(Path(root).glob("cells_*/*/cell.json"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--length-band", help="LO:HI, keep requests whose trace output length is in [LO, HI]")
    ap.add_argument("--out", help="folder for requests.jsonl and holds.json (default: the run folder)")
    a = ap.parse_args(argv)
    root = Path(a.run_dir)
    band = tuple(int(v) for v in a.length_band.split(":")) if a.length_band else None
    plog = measure.read_jsonl(root / "preempt.jsonl")
    meta = {}
    st = root / "state.json"
    if st.exists():
        s = json.loads(st.read_text())
        meta = dict(engine="vllm", engine_version=s.get("vllm"), model=s.get("model"))
    cells = find_cells(root)
    if not cells:
        sys.exit(f"no cells_*/<cell>/cell.json under {root}")
    out = Path(a.out or root)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_L{band[0]}-{band[1]}" if band else ""
    summaries = []
    with open(out / f"requests{tag}.jsonl", "w") as fh:
        for p in cells:
            header, recs, summary = rescore_cell(json.loads(p.read_text()), plog, meta, band)
            fh.write(json.dumps(header) + "\n")
            for x in recs:
                fh.write(json.dumps(x) + "\n")
            summaries.append(summary)
    (out / f"holds{tag}.json").write_text(json.dumps(summaries, indent=1))
    cols = ["cell", "counted", "evictions", "evicted_share", "hold_p50", "hold_p90", "hold_p99", "hold_full_p90", "good",
            "good_full"]
    print("  ".join(f"{c:>13}" for c in cols))
    for s in summaries:
        print("  ".join(f"{s[c]:>13.3f}" if isinstance(s[c], float) else f"{str(s[c]):>13}" for c in cols))
    return summaries


if __name__ == "__main__":
    main()
