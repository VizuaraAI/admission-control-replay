"""A small CPU stand-in for a KV-limited engine, used to test harness/rescore.py against known truth.

It writes the same cell.json and preempt.jsonl that a real run writes, and also returns, for every request, the true
gaps longer than 0.5 s in its token stream. The engine runs one decode step every STEP seconds and every running
request gains a token; when KV use exceeds C it evicts the newest running request (vLLM's order), keeps its tokens,
and puts it at the front of the waiting queue; a request (re)entering prefills prompt + generated tokens at PF
tokens/s before its next token. Clients give up 30 s after arrival with no first token, at the door or in the engine.
Arm "stock" sends every request at once; arm "wm10" holds requests at the door while used + prompt > 0.9 C."""
import json
import random
from pathlib import Path

STEP = 0.05


def simulate(name, lengths, prompts, rate, C, PF=20000.0, arm="stock", wall_s=300.0, seed=1, t0=1000.0):
    rng = random.Random(seed)
    t, arr = 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t >= wall_s:
            break
        arr.append(t)
    reqs = []
    for k, ta in enumerate(arr):
        i = rng.randrange(len(lengths))
        reqs.append(dict(rid=f"{name}-r{k:05d}", i=i, p=prompts[i], L=lengths[i], t_arr=round(ta, 4), t_send=None,
                         t_first=None, t_last=None, n=0, state="queued", marks=0, max_gap=0.0, err=None))
    plog, gaps = [], {r["rid"]: [] for r in reqs}
    door, waiting, running = [], [], []        # running: list of [req, ready_at]
    ia, now = 0, 0.0

    def used():
        return sum(r["p"] + r["n"] for r, _ in running)

    while now < wall_s:
        while ia < len(reqs) and reqs[ia]["t_arr"] <= now:
            door.append(reqs[ia])
            ia += 1
        for q, gone in ((door, "abandoned_in_proxy"), (waiting, "abandoned_in_engine")):
            for r in list(q):                  # clients give up after 30 s with no first token
                if r["t_first"] is None and now - r["t_arr"] > 30.0:
                    r["state"] = gone
                    q.remove(r)
        while door:
            r = door[0]
            if arm == "wm10" and used() + sum(x["p"] + x["n"] for x in waiting) + r["p"] > 0.9 * C:
                break
            door.pop(0)
            r["t_send"], r["state"] = round(now, 4), "sent"
            waiting.append(r)
        while waiting and used() + waiting[0]["p"] + waiting[0]["n"] + 1 <= C:
            r = waiting.pop(0)
            tok = r["p"] + r["n"]
            if r["marks"]:
                plog.append(dict(ev="lookup", t=t0 + now, rid=f"cmpl-{r['rid']}-x", hit=0, tokens=tok))
            running.append([r, now + tok / PF])
        for item in running:
            r, ready = item
            if ready > now:
                continue
            tt = round(now + STEP, 4)
            if r["t_first"] is None:
                r["t_first"] = tt
            else:
                g = tt - r["t_last"]
                r["max_gap"] = max(r["max_gap"], g)
                if g > 0.5:
                    gaps[r["rid"]].append(round(g, 4))
            r["t_last"], r["n"] = tt, r["n"] + 1
        for item in list(running):
            if item[0]["n"] >= item[0]["L"]:
                item[0]["state"] = "finished"
                running.remove(item)
        while used() > C and len(running) > 1:
            r, _ = running.pop()
            r["marks"] += 1
            plog.append(dict(ev="preempt", t=t0 + now + STEP, rid=f"cmpl-{r['rid']}-x", computed=r["p"] + r["n"],
                             prompt=r["p"], output=r["n"], tokens=r["p"] + r["n"], num=r["marks"]))
            waiting.insert(0, r)
        now = round(now + STEP, 6)
    for r in reqs:
        if r["state"] == "sent":
            r["state"] = "censored"
        elif r["state"] == "queued":
            r["state"] = "censored_in_proxy"
    cfg = dict(cell=name, arm=arm, door=False, trace="synthetic", load=None, rate=rate, arrival_seed=seed, wall_s=wall_s,
               C=C, PF=PF, count_window_s=wall_s * 0.6, experiment="synth", gpu="cpu", prefix_caching="off")
    cell = dict(cfg=cfg, t0=t0, t_end=t0 + wall_s, requests=reqs, metrics=[], controller=[])
    return cell, plog, gaps


def write_run(root, cells_and_logs):
    root = Path(root)
    with open(root / "preempt.jsonl", "w") as fh:
        for cell, plog, _ in cells_and_logs:
            d = root / f"cells_{cell['cfg']['C']}" / cell["cfg"]["cell"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "cell.json").write_text(json.dumps(cell))
            for e in plog:
                fh.write(json.dumps(e) + "\n")
