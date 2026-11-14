#!/usr/bin/env python
"""Run baseline vs JEPA variants over several seeds and print a summary table.

    python scripts/sweep.py --name pilot --seeds 0 1 2 -- --langs eng --model distilroberta-base --epochs 3

Everything after `--` is passed to every run. Finished runs are skipped, so the
sweep can be resumed; `--summary-only` just re-prints the table.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

VARIANTS = {
    "baseline": [],
    "jepa-aux-random": ["--jepa-weight", "1.0", "--mask-strategy", "random"],
    "jepa-aux-opinion": ["--jepa-weight", "1.0", "--mask-strategy", "opinion"],
    "jepa-pretrain": ["--pretrain-epochs", "2", "--mask-strategy", "random"],
    "jepa-pretrain+aux": ["--pretrain-epochs", "2", "--jepa-weight", "1.0", "--mask-strategy", "random"],
}


def summarize(root: Path) -> str:
    rows = ["| variant | seeds | dev RMSE | test RMSE | test PCC-V | test PCC-A |", "|---|---|---|---|---|---|"]
    for variant in VARIANTS:
        results = [json.loads(p.read_text()) for p in sorted(root.glob(f"{variant}/s*/result.json"))]
        if not results:
            continue

        def ms(get):
            x = np.array([get(r) for r in results])
            return f"{x.mean():.4f} ± {x.std():.4f}"

        rows.append(f"| {variant} | {len(results)} | {ms(lambda r: r['dev']['ALL']['rmse_va'])} "
                    f"| {ms(lambda r: r['test']['ALL']['rmse_va'])} | {ms(lambda r: r['test']['ALL']['pcc_v'])} "
                    f"| {ms(lambda r: r['test']['ALL']['pcc_a'])} |")
    return "\n".join(rows)


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--variants", nargs="+", default=["baseline", "jepa-aux-random", "jepa-aux-opinion"],
                   choices=list(VARIANTS))
    p.add_argument("--gpus", type=int, default=0, help="run jobs in parallel, one per GPU (0 = sequential)")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args(argv)
    root = Path("runs") / args.name
    if not args.summary_only:
        jobs = []
        for seed in args.seeds:  # seed-major order: a partial sweep is still a fair comparison
            for variant in args.variants:
                out = root / variant / f"s{seed}"
                if not (out / "result.json").exists():
                    jobs.append((out, [sys.executable, "-m", "dimabsa_jepa.train", *extra, *VARIANTS[variant],
                                       "--seed", str(seed), "--out", str(out)]))
        slots: queue.Queue = queue.Queue()
        for g in range(max(1, args.gpus)):
            slots.put(g)

        def run(job):
            out, cmd = job
            gpu = slots.get()
            try:
                env = dict(os.environ)
                if args.gpus:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                out.mkdir(parents=True, exist_ok=True)
                print("+", " ".join(cmd), flush=True)
                with open(out / "log.txt", "w") as log:
                    code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                tail = (out / "log.txt").read_text().splitlines()[-1:]
                print(f"[{out}] exit={code} {' '.join(tail)}", flush=True)
                return code
            finally:
                slots.put(gpu)

        with ThreadPoolExecutor(max(1, args.gpus)) as pool:
            codes = list(pool.map(run, jobs))
        if any(codes):
            print(f"{sum(c != 0 for c in codes)} run(s) failed; see log.txt in the run directories", flush=True)
    table = summarize(root)
    (root / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
