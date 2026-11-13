"""Apply config/exp02.json to a run folder: python -I harness/analyze.py runs/exp02
Recomputes every cell's measures from cell.json and the preemption log, then writes summary.json into the run folder.
Cells that were not run read as missing, so a partial run gives INCONCLUSIVE for the gates that need them."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure  # noqa: E402


def main(root, config=None):
    root = Path(root)
    rule = json.loads(Path(config or HERE.parent / "config" / "exp02.json").read_text())
    state = json.loads((root / "state.json").read_text())
    plog = measure.read_jsonl(root / "preempt.jsonl")
    out = dict(state_notes=state["notes"], regimes=[])
    for reg in state["regimes"]:
        C = reg["C"]
        cal = reg["calibration"]
        cells = {}
        for name in reg["cells"]:
            p = root / f"cells_{C}" / name / "cell.json"
            if not p.exists():
                continue
            cell = json.loads(p.read_text())
            m = measure.cell_measures(cell, plog, cal["PF"], C)
            cfg = cell["cfg"]
            m["valid"] = m["wall_s"] >= rule["validity_min_wall_s"] and m["errors"] <= 0.05 * max(1, m["counted"])
            m["rate"] = cfg["rate"]
            cells[name] = m
        out["regimes"].append(dict(C=C, engine_reported_kv_tokens=reg.get("engine_reported_kv_tokens"), calibration=cal, rates=reg.get("rates"),
                                   in_run=dict(g1=reg.get("g1_in_run"), g3=reg.get("g3_in_run")), cells=cells))
    reg = out["regimes"][-1]
    cells = reg["cells"]

    def get(name, key):
        c = cells.get(name)
        if c is None or not c["valid"]:
            return None
        return c[key]

    G = {}
    s25, s10 = cells.get("T_stock_25"), cells.get("T_stock_10")
    R = rule["gates"]
    if s25 and s25["valid"]:
        G["G1"] = "PASS" if s25["kv_ge_095_share"] >= R["G1"]["kv_share_min"] and (s25["preemptions_metric"] or 0) > 0 else "FAIL"
    else:
        G["G1"] = "INCONCLUSIVE"
    g25, g10 = get("T_stock_25", "goodput_rps"), get("T_stock_10", "goodput_rps")
    intr = get("T_stock_25", "interrupted_share")
    if g25 is None or (g10 is None and intr is None):
        G["G2"] = "INCONCLUSIVE"
    else:
        G["G2"] = "PASS" if ((g10 is not None and g25 <= 0.85 * g10) or (intr is not None and intr >= 0.20)) else "FAIL"
    gg, gs = get("T_gate_25", "goodput_rps"), get("T_gate_25", "rebuild_share")
    if gg is None or gs is None or g25 is None:
        G["G3"] = "INCONCLUSIVE"
    else:
        G["G3"] = "PASS" if (gs <= R["G3"]["share_max"] and gg >= R["G3"]["goodput_vs_stock_min"] * g25) else "FAIL"
    gk = get("T_known_25", "goodput_rps")
    if gg is None or not gk:
        G["G4"] = "INCONCLUSIVE"
        dist = None
    else:
        dist = 1 - gg / gk
        G["G4"] = "PREDICTION_NOT_NEEDED" if dist < 0.15 else ("PREDICTION_MATTERS" if dist > 0.30 else "DECIDE_AFTER_THREE_TRACES")
    gw, gaa = get("T_wm10_25", "goodput_rps"), get("T_stock_25_AA", "goodput_rps")
    wm_flag = None
    if None not in (gw, gg, gaa, g25):
        wm_flag = abs(gw - gg) <= abs(gaa - g25)
    G["G4_distance"] = dist
    G["watermark_within_noise"] = wm_flag
    G["aa_noise_rps"] = None if None in (gaa, g25) else abs(gaa - g25)
    ggaa = get("T_gate_25_AA", "goodput_rps")
    G["gate_aa_noise_rps_reported"] = None if None in (ggaa, gg) else abs(ggaa - gg)
    G["gate_over_stock_reported"] = None if None in (gg, g25) or not g25 else gg / g25
    if G["G1"] == "FAIL":
        label = "G1_FAIL"
    elif "INCONCLUSIVE" in (G["G1"], G["G2"]):
        label = "INCONCLUSIVE"
    elif G["G2"] == "FAIL":
        label = "NO_FAILURE"
    elif G["G3"] == "INCONCLUSIVE":
        label = "INCONCLUSIVE"
    elif G["G3"] == "FAIL":
        label = "GATE_MISSES"
    elif G["G4"] == "INCONCLUSIVE":
        label = "INCONCLUSIVE"
    else:
        label = "PASS_" + G["G4"]
    out["gates"] = G
    out["outcome"] = label
    (root / "summary.json").write_text(json.dumps(out, indent=1))
    print("outcome", label, json.dumps(G))
    cols = ["arm", "door", "load", "rate", "arrivals", "counted", "finished", "good", "goodput_rps", "goodput_tok_s", "goodput_rps_incl_on_track",
            "rebuild_share", "rebuild_share_no_hits", "proxy_estimate_share", "interrupted_share", "preemptions_log", "kv_ge_095_share",
            "censored_share", "door_rejected", "h_max", "ttft_p50", "tpot_p50", "tpot_p90", "prefix_hit_frac_at_resume_median", "valid"]
    for name, c in cells.items():
        print(name, {k: (round(c[k], 4) if isinstance(c[k], float) else c[k]) for k in cols})


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python -I harness/analyze.py RUN_DIR [CONFIG]")
    main(*sys.argv[1:])
