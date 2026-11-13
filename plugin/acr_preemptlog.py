"""Instrumentation only: one log line per preemption, and one per prefix-cache lookup of a preempted request.

vLLM 0.32.0 (commit 5df63a7203cdb9328d824a27b682e77c7b9928be) keeps no count of tokens prefilled again after a
preemption, so this general plugin wraps two Scheduler methods and writes JSON lines to the
file named in ACR_PREEMPT_LOG. It changes no decision: each wrapper calls the original and returns its result.
"""
import json
import os
import threading
import time

_lock = threading.Lock()
_fh = None


def _write(rec):
    global _fh
    try:
        with _lock:
            if _fh is None:
                _fh = open(os.environ["ACR_PREEMPT_LOG"], "a", buffering=1)
            _fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def register():
    if not os.environ.get("ACR_PREEMPT_LOG"):
        return
    from vllm.v1.core.sched.scheduler import Scheduler
    if getattr(Scheduler, "_acr_patched", False):
        return
    orig_preempt = Scheduler._preempt_request
    orig_hit = Scheduler._get_local_prefix_cache_hit

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        _write(dict(ev="preempt", t=time.time(), rid=request.request_id, computed=int(request.num_computed_tokens),
                    prompt=int(request.num_prompt_tokens), out=int(request.num_output_tokens), tokens=int(request.num_tokens),
                    n=int(request.num_preemptions) + 1))
        return orig_preempt(self, request, timestamp, drop_stale_output=drop_stale_output)

    def _get_local_prefix_cache_hit(self, request):
        res = orig_hit(self, request)
        if request.num_preemptions > 0:
            _write(dict(ev="lookup", t=time.time(), rid=request.request_id, hit=int(res[1]), tokens=int(request.num_tokens),
                        n=int(request.num_preemptions)))
        return res

    Scheduler._preempt_request = _preempt_request
    Scheduler._get_local_prefix_cache_hit = _get_local_prefix_cache_hit
    Scheduler._acr_patched = True
    _write(dict(ev="plugin_loaded", t=time.time(), pid=os.getpid()))
