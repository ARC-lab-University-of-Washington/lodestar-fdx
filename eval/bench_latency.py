# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file bench_latency.py
# @brief ON-DEVICE latency + power benchmark.
#
# Runs the DEPLOYED lodestar_edge.diagnose over a handful of representative
# in-corpus symptoms, timing each end-to-end (retrieval + generation), and —
# on a Jetson — samples `tegrastats` in parallel to report module power
# (VDD_IN) for the 7-15 W envelope claim. If tegrastats is unavailable it
# reports latency only (power = n/a). python bench_latency.py # 2 warmup + 5
# timed python bench_latency.py --n 8 --warmup 2
#
"""ON-DEVICE latency + power benchmark. Runs the DEPLOYED lodestar_edge.diagnose over a handful of
representative in-corpus symptoms, timing each end-to-end (retrieval + generation), and — on a
Jetson — samples `tegrastats` in parallel to report module power (VDD_IN) for the 7-15 W envelope
claim. If tegrastats is unavailable it reports latency only (power = n/a).

  python bench_latency.py                 # 2 warmup + 5 timed
  python bench_latency.py --n 8 --warmup 2
"""
import os, sys, json, time, argparse, subprocess, re, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.environ.get("LODESTAR_EDGE_DIR", os.path.dirname(HERE))
sys.path.insert(0, EDGE_ROOT)
from lodestar_edge.segment import load_corpus              # noqa: E402
from lodestar_edge.retriever import SalienceBM25Retriever  # noqa: E402
from lodestar_edge.diagnose import diagnose                # noqa: E402

RESULTS = os.path.join(HERE, "results")
CORPUS = os.environ.get("LODESTAR_CORPUS", os.path.join(EDGE_ROOT, "lodestar_edge", "corpus.json"))

SYMPTOMS = [
    "master alarm, MAIN B BUS UNDERVOLT, then MAIN A UNDERVOLT, O2 QUANTITY 2 reading zero",
    "CRYO PRESSURE light, O2 tank 2 pressure and quantity readings erratic",
    "MAIN BUS B UNDERVOLT, fuel cell 3 current dropping",
    "cabin pressure decreasing, suit flow low",
    "CO2 partial pressure rising, LiOH canister due",
    "RCS quad C helium pressure low, propellant temperature high",
    "gimbal drive fail on the SPS, thrust vector control suspect",
    "warning tone lost, caution and warning system unresponsive",
]


class TegraSampler:
    """Spawn tegrastats and collect VDD_IN (module) power in mW while the benchmark runs."""
    def __init__(self, interval_ms=500):
        self.proc = None
        self.samples = []
        try:
            self.proc = subprocess.Popen(["tegrastats", "--interval", str(interval_ms)],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True, bufsize=1)
        except (FileNotFoundError, OSError):
            self.proc = None
        self._re = re.compile(r"VDD_IN\s+(\d+)mW")

    def drain(self):
        if not self.proc or not self.proc.stdout:
            return
        # non-blocking-ish: read whatever lines are buffered
        import select
        while True:
            r, _, _ = select.select([self.proc.stdout], [], [], 0)
            if not r:
                break
            line = self.proc.stdout.readline()
            if not line:
                break
            m = self._re.search(line)
            if m:
                self.samples.append(int(m.group(1)))

    def stop(self):
        if self.proc:
            self.drain()
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()

    def summary(self):
        if not self.samples:
            return {"available": False}
        return {"available": True, "n_samples": len(self.samples),
                "mean_W": round(st.mean(self.samples) / 1000, 2),
                "peak_W": round(max(self.samples) / 1000, 2),
                "min_W": round(min(self.samples) / 1000, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("LODESTAR_EDGE_MODEL", "llama3.2:3b"))
    ap.add_argument("--base-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--hardware", default=os.environ.get("LODESTAR_HW", "jetson-orin-nano"))
    ap.add_argument("--no-power", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    t0 = time.perf_counter()
    retr = SalienceBM25Retriever(load_corpus(CORPUS))
    build_s = time.perf_counter() - t0
    print(f"[bench] index built in {build_s:.1f}s; model={args.model} k={args.k} hw={args.hardware}",
          flush=True)

    syms = (SYMPTOMS * ((args.n + args.warmup) // len(SYMPTOMS) + 1))[:args.n + args.warmup]
    lat, sampler = [], (None if args.no_power else TegraSampler())
    for i, sym in enumerate(syms, 1):
        t0 = time.perf_counter()
        res = diagnose(sym, retr, model=args.model, base_url=args.base_url, k=args.k, gate=False)
        dt = time.perf_counter() - t0
        if sampler:
            sampler.drain()
        warm = i <= args.warmup
        if not warm:
            lat.append(dt)
        print(f"  {'warm' if warm else 'time'} {i}: {dt:5.1f}s  "
              f"{'DECL' if res['declined'] else str(res.get('n_points', 0)) + 'pts'}", flush=True)
    if sampler:
        sampler.stop()

    out = {"hardware": args.hardware, "model": args.model, "k": args.k,
           "index_build_s": round(build_s, 1), "n_timed": len(lat),
           "latency_s": {"mean": round(st.mean(lat), 1), "p95": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 1),
                         "min": round(min(lat), 1), "max": round(max(lat), 1)} if lat else {},
           "power": (sampler.summary() if sampler else {"available": False})}
    print("\n==== ON-DEVICE LATENCY / POWER ====")
    print(f"  latency (s):  {out['latency_s']}")
    print(f"  power:        {out['power']}")
    json.dump(out, open(os.path.join(RESULTS, "latency_result.json"), "w", encoding="utf-8"), indent=1)
    print(f"-> {os.path.join(RESULTS, 'latency_result.json')}")


if __name__ == "__main__":
    main()
