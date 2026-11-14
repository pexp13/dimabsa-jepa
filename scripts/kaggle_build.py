#!/usr/bin/env python
"""Собирает самодостаточный Kaggle-скрипт, который запускает scripts/sweep.py на GPU.

    python scripts/kaggle_build.py --user <kaggle user> --name xlmr-s0 -- \\
        --name xlmr-all --seeds 0 --gpus 2 -- --model xlm-roberta-base --epochs 5 --amp
    kaggle kernels push -p kaggle/xlmr-s0 --accelerator NvidiaTeslaT4
    kaggle kernels output <user>/dimabsa-jepa-xlmr-s0 -p runs/kaggle/xlmr-s0

Всё после первого `--` — командная строка sweep.py. Исходники репозитория
встраиваются в скрипт, поэтому репозиторий на GitHub может быть приватным; данные
задачи клонируются из официального репозитория уже на Kaggle (нужен включённый интернет).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = '''\
import base64, io, os, shutil, subprocess, sys, tarfile

SRC = "{src}"
ARGS = {args!r}

work = "/kaggle/working"
repo = "/tmp/repo"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(SRC))).extractall(repo)
os.chdir(repo)
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/DimABSA/DimABSA2026",
                "data/DimABSA2026"], check=True)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
env = dict(os.environ, PYTHONPATH=os.path.join(repo, "src"), TOKENIZERS_PARALLELISM="false")
code = subprocess.run([sys.executable, "scripts/sweep.py", *ARGS], env=env).returncode
if os.path.isdir("runs"):
    shutil.copytree("runs", os.path.join(work, "runs"), dirs_exist_ok=True)
sys.exit(code)
'''


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit(__doc__)
    i = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True)
    p.add_argument("--name", required=True)
    args = p.parse_args(argv[:i])
    sweep_args = argv[i + 1:]

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in ("src", "scripts/sweep.py"):
            tar.add(ROOT / rel, arcname=rel,
                    filter=lambda t: None if "__pycache__" in t.name or t.name.endswith(".egg-info") else t)
    out = ROOT / "kaggle" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.py").write_text(TEMPLATE.format(src=base64.b64encode(buf.getvalue()).decode(), args=sweep_args))
    slug = f"dimabsa-jepa-{args.name}"
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }, indent=2))
    print(out)


if __name__ == "__main__":
    main()
