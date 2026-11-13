"""Rebuild data/thinking_long.jsonl from the public Qwen-Bailian Thinking trace, offline, no GPU.

    python -I tools/make_requests.py TRACE_DIR

TRACE_DIR holds github.com/alibaba-edu/qwen-bailian-usagetraces-anon at commit 5f7439c51ec248a0c585f7d90a41a6f57773b912
(Apache-2.0; the file is stored with Git LFS, so run `git lfs pull` there first). Only input_length, output_length and
hash_ids are kept; the trace has no text.

Step 1, as in our first experiment: keep rows with output_length > 0 and input_length + min(output_length, 16000) <= 40896
(the Qwen3-8B context of 40,960 less 64), shuffle with random.Random(34) and keep the first 1,500.
Step 2, for Exp02: keep, in the same order, the requests whose trace output is at least 2,000 tokens. That leaves 622.
Both steps are checked against the SHA-256 values in data/manifest.json, and the output must match byte for byte.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
FILE, FILE_SHA = "qwen_thinking_blksz_16.jsonl", "41ac36d9d1b54d084eeb6b05e9d142ba9d02f8bf79cfc9fff418d8ab7cdee906"
STEP1_SHA = "f243ce58dbbf9df1a1e3768ad55eae21bed62b8b6bfa66f9bc3752f0fea30ec3"
KEEP1, MIN_OUT, OUTCAP, CTX = 1500, 2000, 16000, 40960 - 64


def dumps(rows):
    return "".join(json.dumps(x, separators=(",", ":")) + "\n" for x in rows)


def step1(raw):
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    ok = [r for r in rows if r["output_length"] > 0 and r["input_length"] + min(r["output_length"], OUTCAP) <= CTX]
    random.Random(34).shuffle(ok)
    out = []
    for i, r in enumerate(ok[:KEEP1]):
        assert len(r["hash_ids"]) == -(-r["input_length"] // 16), i
        out.append(dict(i=i, p=int(r["input_length"]), L=min(int(r["output_length"]), OUTCAP), L_raw=int(r["output_length"]), h=r["hash_ids"]))
    return out


def step2(rows):
    kept = [r for r in rows if r["L_raw"] >= MIN_OUT and r["p"] + min(r["L_raw"], OUTCAP) <= CTX]
    return [dict(i=k, i_exp01=r["i"], p=r["p"], L=min(r["L_raw"], OUTCAP), L_raw=r["L_raw"], h=r["h"]) for k, r in enumerate(kept)]


def stats(out):
    p = [x["p"] for x in out]
    L = [x["L"] for x in out]
    return dict(written=len(out), p_mean=sum(p) / len(p), L_clipped_mean=sum(L) / len(L), L_median=sorted(L)[len(L) // 2],
                memtime_tokens_steps_mean=sum(a * b + b * b / 2 for a, b in zip(p, L)) / len(out))


def main(tdir):
    raw = (Path(tdir) / FILE).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    assert got == FILE_SHA, f"{FILE} has SHA-256 {got}, expected {FILE_SHA}; is it the LFS file at the pinned commit?"
    rows1 = step1(raw)
    assert hashlib.sha256(dumps(rows1).encode()).hexdigest() == STEP1_SHA, "step 1 does not reproduce"
    out = step2(rows1)
    text = dumps(out).encode()
    man = json.loads((HERE / "data" / "manifest.json").read_text())
    want = man["traces"]["thinking_long"]["out_sha256"]
    assert hashlib.sha256(text).hexdigest() == want, "step 2 does not reproduce data/thinking_long.jsonl"
    (HERE / "data" / "thinking_long.jsonl").write_bytes(text)
    print("reproduced data/thinking_long.jsonl:", json.dumps(stats(out)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
