# Admission-control replay harness

This repository holds the harness we used to test admission control in front of vLLM. It replays a real request trace
against vLLM 0.32.0 serving Qwen3-8B, puts a small proxy between the clients and the server, and lets the proxy decide
when each request may enter the engine. With it you can rerun our Exp02 cells on your own NVIDIA GPU, one cell at a
time, without Modal or any cloud account.

The question behind the experiments is simple to state. When the KV cache fills, vLLM preempts running requests and
later rebuilds their KV by prefilling them again. That rebuild is wasted GPU time. Can a gate at the proxy, which sees
only what a gateway sees, keep that waste under a budget and still serve more requests within the latency limits than
stock vLLM does? And how far is such a gate from one that knows every request's output length in advance?

## What the harness does

The trace gives each request a prompt length, an output length and the 16-token block hashes of its prompt. There is
no text. The harness turns each block hash into 16 fixed token IDs, so requests that share blocks in the trace share
prefixes here, and sends the prompt as token IDs. Every request asks for exactly its output length from the trace, clipped at 16,000 tokens (`max_tokens`
with `ignore_eos`). vLLM's admission does not read `max_tokens`, so the engine does not learn the length from it.

Requests arrive as a Poisson process. The proxy reads the server's Prometheus metrics every half second and passes
requests to the engine according to one of four arms:

- `stock`: no gate. Every request goes to vLLM on arrival, and vLLM admits it when its prompt fits.
- `gate`: the recompute budget gate. The proxy keeps a FIFO and passes the head when the estimated KV in use, plus the
  prompt, plus a headroom `h` fits in the capacity C. Every 2 s it updates `h` from its own estimate of rebuild GPU
  time, so the headroom grows when evictions cost more than the budget (1% of GPU time) and shrinks when they cost less.
- `known`: the known-length gate. It passes the head only if the projected peak KV of all admitted requests plus the
  candidate, computed with their exact remaining lengths, fits in C. This is the only arm that reads output lengths,
  and it serves as the reference for how much knowing the lengths is worth.
- `wm10`: a watermark. It passes the head when the KV in use plus the prompt leaves at least 10% of C free.

All three proxy arms also hold the queue while vLLM has preempted requests waiting. Two extra cells add a front door
that refuses a request on arrival when the queued work would wait too long. Every decision goes through the `Policy`
class of AdmitPerf and is written to a JSON-line log in its format.

A small vLLM plugin (`plugin/acr_preemptlog.py`) logs each preemption and each prefix-cache lookup of a preempted
request. vLLM 0.32.0 keeps no count of the tokens it prefills again after a preemption, and this log is how we measure
them. The plugin wraps two scheduler methods, calls the originals and returns their results, so it changes no decision.

Before the cells, a calibration measures the GPU: the cold prefill rate (PF), the decode step time near a full cache
(tau_full), and the behaviour of a forced burst of evictions. The capacity in requests per second is the smaller of a
KV limit and a prefill limit computed from these numbers, and each cell's arrival rate is a multiple of it.

## The Exp02 load regime

Exp02 is the harder regime, where an eviction costs a real rebuild. Its settings are in `config/exp02.json`:

- KV cache capped at 190,000 tokens (`--kv-cache-memory-bytes` = 190,000 times 147,456 bytes), whatever the GPU.
- Prefix caching off, so a resumed request prefills its prompt and all the tokens it had generated.
- The 622 Thinking requests whose trace output is 2,000 tokens or more, with outputs clipped at 16,000.
- Load of 2.5 times the calibrated capacity for the main cells, and 1.0 times for one stock reference cell.
- Latency limits of 30 s for time to first token (TTFT) and 0.15 s mean time per output token (TPOT). A client gives up
  on a request with no first token 30 s after it arrived.
- Each cell runs 1,260 s and counts the requests that arrived in its first 720 s, so long outputs have time to finish.

There are nine cells: `T_stock_25`, `T_stock_10`, `T_gate_25`, `T_known_25`, `T_wm10_25`, a repeat of the stock cell
(`T_stock_25_AA`) and of the gate cell (`T_gate_25_AA`) for run-to-run noise, and the two front-door cells
`T_gatedoor_25` and `T_wm10door_25`. All 2.5x cells share the same arrivals and requests.

`config/exp02.json` is the decision rule we wrote before our run, with internal notes removed. No number, cell, arm or
gate changed, but its SHA-256 no longer matches the one we recorded on the day.

## What you need

- Linux on x86_64 with one NVIDIA GPU and a driver that runs torch 2.13.0. The weights take about 16 GB and the KV
  cache 28 GB, so you need a 48 GB card such as the L40S we used, or larger. On a smaller card the server will not start.
- Python 3.12, and disk space for the packages (several gigabytes) and the weights (about 16 GB).
- Qwen3-8B downloads from Hugging Face at a pinned revision. It is not gated, so no token is needed.

Install into a fresh virtual environment, with hashes checked:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install --require-hashes -r requirements-vllm.lock.txt
```

The calibration runs on your GPU, so the load is 2.5 times the capacity of your machine, not of ours. Absolute rates
and goodput will differ from ours on a different GPU. The ratios between arms are what the gates compare.

## Run one cell

Start with the smoke test. It runs a short calibration and four 2-minute cells, enough to check that the server starts,
the plugin loads and every arm sends traffic. Use its own folder, since it is never counted:

```bash
python -I run.py smoke --out runs/smoke
```

Then run a real cell:

```bash
python -I run.py cell T_stock_25 --out runs/exp02
```

Each command starts vLLM, works, and stops it again. The first command on a run folder calibrates first; later
commands on the same folder reuse that calibration, so keep one folder for all the cells you want to compare. The
eviction step of the calibration alone can take up to 420 s, and each cell takes 1,260 s plus the time for the engine
to drain. You can name several cells in one command, or run them all:

```bash
python -I run.py cell T_gate_25 T_known_25 --out runs/exp02
python -I run.py all --out runs/exp02
```

`all` runs the cells in order and applies the two in-run checks of the rule. If G1 fails on `T_stock_25`, it stops
there. If `T_gate_25` misses G3, it adds one cell, `T_gate_25_gain`, with the gate's gain multiplied by 4 or by 0.25.
It skips cells that are already done in the folder, so you can resume an interrupted run. Other options are listed by
`python -I run.py --help`; `--model` takes a local folder if you already have the weights.

When you have the cells you want, apply the gates:

```bash
python -I harness/analyze.py runs/exp02
```

## What each output file means

Everything goes into the run folder, with C (190000) in some names:

- `state.json`: the run record. It holds the vLLM version, the model, the GPU name, the KV size the engine reported,
  the calibration, the capacity rates, the cells done, the results of the in-run checks and any notes.
- `server_190000.log`: the vLLM server log, starting with the exact command line.
- `preempt.jsonl`: the plugin's log. A `preempt` line has the request ID, its tokens at that moment and how many were
  computed. A `lookup` line has the prefix-cache hit when a preempted request was looked up again (always 0 here, with
  caching off).
- `calib_190000/calibration.json`: PF and the TTFT measurements behind it, tau_full and tau_single (seconds per
  decode step), and every request of the eviction burst with its long gaps.
- `cells_190000/<cell>/cell.json`: the raw cell. `cfg` is the cell's settings. `requests` has one row per arrival with
  times relative to the cell start (`t_arr`, `t_send`, `t_first`, `t_last`), tokens received (`n`) and the final
  `state`: `finished`, `censored` (still streaming at the end), `censored_in_proxy`, `abandoned_in_proxy` or
  `abandoned_in_engine` (no first token within 30 s), `rejected_by_door`, `short` or `error`. `metrics` is the series
  of (time, KV use, running, waiting, preemptions so far, proxy queue, requests in flight, gate headroom h).
  `controller` is the gate's (time, estimated rebuild share, h) every 2 s.
- `cells_190000/<cell>/decisions.jsonl`: every admission decision, in AdmitPerf's log format.
- `cells_190000/<cell>/measures.json`: the cell's measures. The main ones are `goodput_rps` (counted requests that
  finished within both limits, per second of the 720 s window), `rebuild_share` (GPU time spent rebuilding preempted
  requests, from the plugin log and PF, as a share of the cell's wall time), `interrupted_share` (finished requests that
  were preempted at least once), `kv_ge_095_share` (metric samples with the KV cache at least 95% full),
  `censored_share`, TTFT and TPOT percentiles, and refusals by prompt length.
- `summary.json`, written by `analyze.py`: all measures recomputed from the raw files, the four gates and the outcome.
  G1 asks whether KV pressure is real, G2 whether stock vLLM is harmed by it, G3 whether the gate holds the rebuild
  budget while beating stock goodput by 10%, and G4 how far the gate falls short of the known-length gate. A cell under
  1,240 s, or with more than 5% of its requests in error, is invalid, and a gate that needs it reads `INCONCLUSIVE`.

## Tests on a laptop

The tests need no GPU and no vLLM, only `aiohttp` and `numpy` (Python 3.11 or later):

```bash
pip install --require-hashes -r requirements-test.lock.txt
python -I tests/test_offline.py
python -I tests/test_replay.py
```

`test_offline.py` checks the data against its manifest, the vendored AdmitPerf files against their recorded hashes,
each arm's decisions, the rebuild accounting, the vLLM command line and the plugin's entry point. `test_replay.py`
starts a fake engine on the loopback, runs a short cell of every arm through the real proxy, and runs `analyze.py` on
the result. It takes about half a minute. With `pytest` installed, `pytest tests` runs both.

## Next: SGLang

We plan to replay the same cells on SGLang, to see whether the result holds on a second engine. This is not
implemented yet, and nothing in this repository runs SGLang.

The plan has one twist. SGLang's scheduler takes `max_new_tokens` into account when it admits a request, so sending the
true output length, as we do with vLLM, would hand the engine the very knowledge the known-length arm is meant to
isolate. The replay will instead send a large `max_new_tokens`, set `min_new_tokens` to the trace length so the
request cannot stop early, and have the proxy cancel the request once it has streamed that many tokens. The engine
then sees an unknown length, as it would in production. The port also needs its own lockfile, its own preemption log
and its own calibration.

## Repository

- `run.py`: the command-line entry point (smoke, calibrate, cell, all).
- `harness/replay.py`: the proxy, the four arms, the front door and the calibration.
- `harness/server.py`: starts and stops vLLM with the Exp02 flags and the plugin on its path.
- `harness/measure.py` and `harness/analyze.py`: the per-cell measures and the gates.
- `plugin/`: the preemption log plugin, with the entry-point metadata vLLM uses to find it.
- `vendor/`: the vendored AdmitPerf core.
- `config/exp02.json`: the cells, arms, gates and metric definitions.
- `data/`: the request list, its manifest and the trace licence.
- `tools/make_requests.py`: rebuilds `data/thinking_long.jsonl` from the public trace and checks it byte for byte.
- `requirements-vllm.lock.txt`: the full hashed closure of vLLM 0.32.0 for Python 3.12 on Linux x86_64.
  `requirements-test.lock.txt`: the hashed subset the tests need.

The Modal wrapper we ran the experiments with, our run's results and logs, and our first experiment (prefix caching on,
with a chat trace as well) are not in this repository. An OSV check of the vLLM lockfile on 11 November 2026 found no
known malicious package. Two pinned packages, filelock 3.32.7 and setuptools 80.10.2, have published advisories; both
come from vLLM's dependency closure.

## Credits and licences

The admission decisions and their log format come from AdmitPerf,
[github.com/harshuljain13/AdmitPerf](https://github.com/harshuljain13/AdmitPerf). Its core (`admitperf/__init__.py`
and `admitperf/core`) is vendored unmodified from commit `5bcb81c30ba5639c3d2c35ff91b5a937436cc751` under the MIT
licence, which is in `vendor/ADMITPERF_LICENSE`. `vendor/ADMITPERF_COMMIT.json` records the SHA-256 of every file.

`data/thinking_long.jsonl` is derived from the Qwen-Bailian Thinking trace,
[github.com/alibaba-edu/qwen-bailian-usagetraces-anon](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon) at
commit `5f7439c51ec248a0c585f7d90a41a6f57773b912`, under the Apache License 2.0, a copy of which is in
`data/LICENSE-qwen-bailian-traces.txt`. We kept only the lengths and block hashes, renamed the fields, selected and
reordered the rows and clipped the outputs; `data/manifest.json` lists each step.

Qwen3-8B is under the Apache License 2.0 and is downloaded from Hugging Face, not stored here. vLLM and the packages
in the lockfiles carry their own licences. The rest of the code in this repository is under the MIT licence; see
[LICENSE](LICENSE).

Built in the Vizuara Inference Research Bootcamp.
