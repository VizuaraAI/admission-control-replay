"""Per-cell measures, shared by run.py (for the two in-run checks) and analyze.py.
Definitions are those of config/exp02.json. The count window comes from the cell config (720 s in Exp02)."""
import json
import re
from pathlib import Path

TTFT_LIMIT, TPOT_LIMIT = 30.0, 0.150
COUNT_WINDOW_S = 480.0
RID = re.compile(r"(?:cmpl-)?(.+?-r\d{5})")


def read_jsonl(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def tpot(r):
    if r["t_first"] is None or r["t_last"] is None or r["n"] < 2:
        return 0.0
    return (r["t_last"] - r["t_first"]) / (r["n"] - 1)


def good(r, allow_censored=False):
    if r["t_first"] is None:
        return False
    if r["t_first"] - r["t_arr"] > TTFT_LIMIT or tpot(r) > TPOT_LIMIT:
        return False
    if r["state"] == "finished":
        return True
    return allow_censored and r["state"] == "censored"


def rebuild_events(plog, cell_name, t0, t_end):
    """Preemptions of this cell's requests, each with the tokens rebuilt when it was resumed: request tokens minus the
    prefix-cache hit at the last lookup before the request's next preemption (no lookup: not resumed, 0 rebuilt)."""
    by_rid = {}
    for e in plog:
        if e.get("ev") not in ("preempt", "lookup") or not (t0 <= e["t"] <= t_end + 5):
            continue
        m = RID.match(e["rid"])
        if not m or not m.group(1).startswith(cell_name + "-r"):
            continue
        by_rid.setdefault(m.group(1), []).append(e)
    out = []
    for rid, evs in by_rid.items():
        evs.sort(key=lambda e: e["t"])
        cur = None
        for e in evs + [None]:
            if e is None or e["ev"] == "preempt":
                if cur is not None:
                    out.append(cur)
                if e is not None:
                    cur = dict(rid=rid, t=e["t"] - t0, tokens_at_preempt=e["tokens"], computed=e["computed"], rebuilt=0, hit=None, resumed=False)
            elif cur is not None:
                cur.update(rebuilt=max(0, e["tokens"] - e["hit"]), hit=e["hit"], resumed=True)
    return out


def cell_measures(cell, plog, PF, C):
    cfg, reqs = cell["cfg"], cell["requests"]
    name, t0, t_end = cfg["cell"], cell["t0"], cell["t_end"]
    wall = t_end - t0
    cw = float(cfg.get("count_window_s", COUNT_WINDOW_S))
    win = [r for r in reqs if r["t_arr"] < cw]
    fin = [r for r in win if r["state"] == "finished"]
    g = [r for r in win if good(r)]
    g2 = [r for r in win if good(r, allow_censored=True)]
    evs = rebuild_events(plog, name, t0, t_end)
    preempted = {e["rid"] for e in evs}
    W = sum(e["rebuilt"] for e in evs) / PF
    W_upper = sum(e["tokens_at_preempt"] for e in evs) / PF
    ms = [x for x in cell["metrics"] if len(x) >= 5 and isinstance(x[1], (int, float)) and x[1] >= 0]
    kv95 = sum(1 for x in ms if x[1] >= 0.95) / max(1, len(ms))
    pre_metric = (ms[-1][4] - ms[0][4]) if len(ms) >= 2 else None
    states = {}
    for r in win:
        states[r["state"]] = states.get(r["state"], 0) + 1
    hits = [e["hit"] / e["tokens_at_preempt"] for e in evs if e["resumed"] and e["tokens_at_preempt"]]
    return dict(
        cell=name, arm=cfg["arm"], door=cfg.get("door", False), trace=cfg["trace"], load=cfg["load"], rate=cfg["rate"], wall_s=wall,
        arrivals=len(reqs), counted=len(win), finished=len(fin), good=len(g),
        count_window_s=cw, goodput_rps=len(g) / cw, goodput_tok_s=sum(r["n"] for r in g) / cw,
        goodput_rps_incl_on_track=len(g2) / cw,
        states=states, censored_share=sum(v for k, v in states.items() if k.startswith("censored")) / max(1, len(win)),
        interrupted_share=sum(1 for r in fin if r["rid"] in preempted) / max(1, len(fin)),
        preemptions_log=len(evs), preemptions_metric=pre_metric, resumed=sum(1 for e in evs if e["resumed"]),
        rebuilt_tokens=sum(e["rebuilt"] for e in evs), rebuild_share=W / wall if wall > 0 else None,
        rebuild_share_no_hits=W_upper / wall if wall > 0 else None,
        prefix_hit_frac_at_resume_median=(sorted(hits)[len(hits) // 2] if hits else None),
        proxy_estimate_share=cell.get("W_est_total_s", 0.0) / wall if wall > 0 else None,
        kv_ge_095_share=kv95, kv_mean=sum(x[1] for x in ms) / max(1, len(ms)),
        door_rejected=cell.get("door_rejected", 0), wrapped=cell.get("wrapped"),
        errors=states.get("error", 0) + states.get("short", 0),
        h_last=(cell["controller"][-1][2] if cell.get("controller") else None),
        h_max=(max(c[2] for c in cell["controller"]) if cell.get("controller") else None),
        ttft_p50=_pct([r["t_first"] - r["t_arr"] for r in win if r["t_first"] is not None], 50),
        tpot_p50=_pct([tpot(r) for r in fin], 50), tpot_p90=_pct([tpot(r) for r in fin], 90),
        rejections_by_prompt=_by_prompt(win),
    )


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]


def _by_prompt(win):
    edges = [0, 1000, 4000, 8000, 16000, 10 ** 9]
    out = []
    for a, b in zip(edges, edges[1:]):
        rs = [r for r in win if a <= r["p"] < b]
        refused = sum(1 for r in rs if r["state"] in ("rejected_by_door", "abandoned_in_proxy", "abandoned_in_engine"))
        out.append(dict(prompt_from=a, prompt_to=b, n=len(rs), refused_or_abandoned=refused))
    return out
