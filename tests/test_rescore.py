"""Offline checks of harness/rescore.py on synthetic logs with known truth: a hand-made cell, then simulated cells with
similar output lengths and with mixed lengths (drawn from the real trace).

    python -I tests/test_rescore.py        (or: pytest tests)
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "tests"))
import rescore  # noqa: E402
import synth  # noqa: E402

REQUEST_KEYS = {"schema", "run_id", "arm", "oracle", "request_id", "t_arr", "t_send", "t_first_token", "t_done",
                "prompt_tokens", "output_tokens", "evictions", "pauses_s", "door_wait_s", "hold_s", "status"}
RUN_KEYS = {"schema", "run_id", "engine", "engine_version", "model", "gpu", "kv_cap_tokens", "load_x", "seed", "trace",
            "prefix_cache", "hold_limit_s"}


def _req(k, **kw):
    r = dict(rid=f"T_x-r{k:05d}", i=k, p=1000, L=100, t_arr=0.0, t_send=0.0, t_first=1.0, t_last=11.0, n=100,
             state="finished", marks=0, max_gap=0.1, err=None)
    r.update(kw)
    return r


def test_hand_made_cell():
    cfg = dict(cell="T_x", arm="gate", PF=10000.0, count_window_s=100.0, C=190000, load=2.5, arrival_seed=1,
               trace="t", prefix_caching="off", experiment="exp02")
    reqs = [_req(0),                                                      # never evicted
            _req(1, t_send=3.0, t_last=40.0, max_gap=20.0),               # door 3 s, evicted at 10, back at 28 + 0.2
            _req(2, t_send=None, t_first=None, t_last=None, n=0, state="abandoned_in_proxy"),
            _req(3, t_send=None, t_first=None, t_last=None, n=0, state="rejected_by_door"),
            _req(4, t_arr=50.0, t_send=50.0, t_first=51.0, t_last=None, n=50, state="censored"),  # evicted, never back
            _req(5, t_send=0.5, t_last=20.0, max_gap=0.6)]                # evicted, back within 1 s: no pause
    t0 = 500.0
    plog = [dict(ev="preempt", t=t0 + 10, rid="cmpl-T_x-r00001-a", tokens=2000, computed=2000),
            dict(ev="lookup", t=t0 + 10.1, rid="cmpl-T_x-r00001-a", hit=0, tokens=2000),   # vLLM retries the
            dict(ev="lookup", t=t0 + 19, rid="cmpl-T_x-r00001-a", hit=0, tokens=2000),     # lookup every step
            dict(ev="lookup", t=t0 + 28, rid="cmpl-T_x-r00001-a", hit=0, tokens=2000),     # until it fits
            dict(ev="preempt", t=t0 + 60, rid="cmpl-T_x-r00004", tokens=1050, computed=1050),
            dict(ev="preempt", t=t0 + 5, rid="cmpl-T_x-r00005", tokens=1000, computed=1000),
            dict(ev="lookup", t=t0 + 5.5, rid="cmpl-T_x-r00005", hit=0, tokens=1000),
            dict(ev="preempt", t=t0 + 5, rid="cmpl-T_y-r00001", tokens=9, computed=9)]
    cell = dict(cfg=cfg, t0=t0, t_end=t0 + 120, requests=reqs)
    header, recs, s = rescore.rescore_cell(cell, plog)
    by = {x["request_id"]: x for x in recs}
    assert set(header) >= RUN_KEYS and header["hold_limit_s"] == 5.0
    assert all(set(x) >= REQUEST_KEYS for x in recs)
    assert by["T_x-r00000"]["hold_s"] == 0.0 and by["T_x-r00000"]["evictions"] == 0
    assert by["T_x-r00001"]["pauses_s"] == [18.2] and by["T_x-r00001"]["hold_s"] == 21.2
    assert by["T_x-r00002"]["status"] == "abandoned" and by["T_x-r00002"]["hold_s"] == 30.0
    assert by["T_x-r00003"]["status"] == "rejected" and by["T_x-r00003"]["hold_s"] == 0.0
    assert by["T_x-r00004"]["pauses_s"] == [60.0] and by["T_x-r00004"]["status"] == "censored"
    assert by["T_x-r00005"]["evictions"] == 1 and by["T_x-r00005"]["pauses_s"] == []
    assert by["T_x-r00000"]["oracle"] is False
    # counted: r0-r5 arrive before 100 s; good = finished with hold <= 5: r0 and r5
    assert s["counted"] == 6 and s["good"] == 2 and s["evictions"] == 3 and s["refused"] == 2
    assert s["gaps_without_eviction"] == 0
    # percentiles over r0, r1, r2, r5 (r3 rejected and r4 censored are left out): holds 0, 21.2, 30, 0.5
    assert s["censored"] == 1 and s["rejected_rate"] == s["abandoned_rate"] == 1 / 6
    assert s["hold_p50"] == 21.2 and s["hold_p99"] == 30.0


def _check_against_truth(cell, plog, gaps):
    """Every estimated pause matches a true gap, and every true gap clearly over the 1 s threshold is found, both within
    two decode steps (the estimate ends at the return of the request, the truth at its next token)."""
    _, recs, s = rescore.rescore_cell(cell, plog)
    tol, n_paused = 2 * synth.STEP + 1e-6, 0
    for x in recs:
        if x["status"] != "ok":
            continue
        truth = gaps[x["request_id"]]
        for p in x["pauses_s"]:
            assert any(abs(p - g) <= tol for g in truth), (x, truth)
        for g in truth:
            if g > rescore.PAUSE_MIN_S + tol:
                assert any(abs(p - g) <= tol for p in x["pauses_s"]), (x, truth)
        n_paused += bool(x["pauses_s"])
    return s, n_paused


def test_similar_lengths():
    lengths = list(range(200, 401, 10))        # one length range, as in a trace cut to similar outputs
    prompts = [500] * len(lengths)
    stock = synth.simulate("S_stock", lengths, prompts, rate=1.0, C=10000, arm="stock")
    wm = synth.simulate("S_wm10", lengths, prompts, rate=1.0, C=10000, arm="wm10")
    s_stock, paused = _check_against_truth(*stock)
    s_wm, _ = _check_against_truth(*wm)
    assert s_stock["evictions"] > 0 and paused > 0, s_stock
    assert s_wm["evictions"] <= s_stock["evictions"]
    assert s_stock["door_wait_p90"] <= synth.STEP < s_wm["door_wait_p90"]


def test_mixed_lengths_from_trace():
    rows = [json.loads(line) for line in (ROOT / "data" / "thinking_long.jsonl").read_text().splitlines() if line.strip()]
    lengths = [max(20, min(r["L"], 16000) // 20) for r in rows]   # real length mix, scaled down 20x to keep it fast
    prompts = [max(50, r["p"] // 20) for r in rows]
    cell, plog, gaps = synth.simulate("M_stock", lengths, prompts, rate=1.5, C=8000, arm="stock")
    s, paused = _check_against_truth(cell, plog, gaps)
    assert paused > 0
    cw = cell["cfg"]["count_window_s"]
    _, recs, _ = rescore.rescore_cell(cell, plog)
    rows_b = rescore.by_length(recs, cw, buckets=[(0, 200), (200, 801)])
    assert sum(b["counted"] for b in rows_b) == s["counted"]


def test_command_line_and_length_band():
    with tempfile.TemporaryDirectory() as d:
        lengths = list(range(200, 801, 10))
        runs = [synth.simulate(n, lengths, [500] * len(lengths), rate=1.0, C=10000, arm=a)
                for n, a in (("S_stock", "stock"), ("S_wm10", "wm10"))]
        synth.write_run(d, runs)
        out = rescore.main([d])
        assert [x["cell"] for x in out] == ["S_stock", "S_wm10"]
        lines = [json.loads(line) for line in (Path(d) / "requests.jsonl").read_text().splitlines()]
        assert [x["schema"] for x in lines].count("admitperf.run/1") == 2
        band = rescore.main([d, "--length-band", "200:400"])
        lines_b = [json.loads(line) for line in (Path(d) / "requests_L200-400.jsonl").read_text().splitlines()]
        assert all(200 <= x["trace_output_tokens"] <= 400 for x in lines_b if x["schema"] == "admitperf.request/1")
        assert band[0]["counted"] < out[0]["counted"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
