"""Обучение и оценка DimASR (подзадача 1) с необязательной JEPA-функцией потерь.

Три режима через одну точку входа:
  baseline              --jepa-weight 0 --pretrain-epochs 0   обычное дообучение
  JEPA как доп. лосс    --jepa-weight L                       L_reg + L * L_jepa, совместно
  JEPA-предобучение     --pretrain-epochs N                   N эпох только L_jepa, затем дообучение
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import Collator, Instance, find_files, merge_duplicates, read_file, write_predictions
from .metrics import va_report
from .model import DimAsrJepa


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/DimABSA2026")
    p.add_argument("--track", default="a", choices=["a", "b"])
    p.add_argument("--langs", nargs="*", default=None, help="например: eng zho (по умолчанию все)")
    p.add_argument("--domains", nargs="*", default=None, help="например: restaurant laptop (по умолчанию все)")
    p.add_argument("--model", default="xlm-roberta-base")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5, help="learning rate энкодера")
    p.add_argument("--head-lr", type=float, default=5e-4, help="learning rate головы VA и предиктора JEPA")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    # JEPA
    p.add_argument("--jepa-weight", type=float, default=0.0, help="вес JEPA-лосса при дообучении")
    p.add_argument("--pretrain-epochs", type=int, default=0, help="число эпох только с JEPA перед дообучением")
    p.add_argument("--pretrain-on-eval-text", action="store_true",
                   help="использовать в JEPA-предобучении ещё и *тексты* dev/test (метки — никогда)")
    p.add_argument("--mask-strategy", default="random", choices=["random", "opinion"])
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument("--max-span", type=int, default=5)
    p.add_argument("--ema", type=float, default=0.996, help="momentum EMA для target-энкодера (растёт до 1.0); 1.0 = замороженный target")
    p.add_argument("--pred-dim", type=int, default=256)
    p.add_argument("--pred-layers", type=int, default=2)
    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=None, help="взять подвыборку обучающих примеров (отладка / мало данных)")
    p.add_argument("--amp", action="store_true", help="смешанная точность (autocast) на CUDA")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out", default=None, help="каталог прогона (по умолчанию runs/<авто-имя>)")
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(args, split: str) -> dict[str, list[Instance]]:
    files = find_files(args.data_root, split, args.track, args.langs, args.domains)
    if not files:
        raise SystemExit(f"no {split} files under {args.data_root} for track={args.track} "
                         f"langs={args.langs} domains={args.domains}; run scripts/get_data.sh")
    return {f.name: read_file(f) for f in files}


@torch.no_grad()
def predict(model, instances, collate, device, batch_size, amp) -> np.ndarray:
    model.eval()
    order = sorted(range(len(instances)), key=lambda i: len(instances[i].text))  # батчи по длине
    preds = np.zeros((len(instances), 2), dtype=np.float32)
    for s in range(0, len(order), batch_size * 2):
        idx = order[s:s + batch_size * 2]
        batch = collate([instances[i] for i in idx])
        with torch.autocast(device.type, enabled=amp):
            out = model.predict_va(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds[idx] = out.float().cpu().numpy()
    return np.clip(preds * 4.0 + 5.0, 1.0, 9.0)


def evaluate(model, files: dict[str, list[Instance]], collate, device, args) -> tuple[dict, dict]:
    """Метрики по каждому файлу и суммарные; возвращает (отчёт, предсказания по файлам)."""
    report, all_pred, all_gold, preds_by_file = {}, [], [], {}
    for name, insts in files.items():
        pred = np.round(predict(model, insts, collate, device, args.batch_size, args.amp), 2)
        preds_by_file[name] = pred
        gold = np.array([i.va for i in insts])
        report[name] = va_report(pred, gold)
        all_pred.append(pred)
        all_gold.append(gold)
    report["ALL"] = va_report(np.concatenate(all_pred), np.concatenate(all_gold))
    return report, preds_by_file


def run_epoch(model, loader, opt, sched, device, args, *, jepa_weight: float, regress: bool,
              ema_state: dict, tag: str) -> dict:
    model.train()
    sums = {"reg": 0.0, "jepa": 0.0, "std": 0.0}
    n_steps, t0 = 0, time.time()
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, enabled=args.amp):
            loss = torch.zeros((), device=device)
            if regress:
                out = model.predict_va(batch["input_ids"], batch["attention_mask"])
                reg = F.mse_loss(out.float(), (batch["labels"] - 5.0) / 4.0)
                loss = loss + reg
                sums["reg"] += reg.item()
            if jepa_weight > 0:
                jl, info = model.jepa_loss(batch["input_ids"], batch["attention_mask"], batch["jepa_mask"])
                loss = loss + jepa_weight * jl
                sums["jepa"] += jl.item()
                sums["std"] += info["target_std"] if not math.isnan(info["target_std"]) else 0.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if jepa_weight > 0:
            # momentum линейно растёт от --ema до 1.0 за все JEPA-шаги (как в I-JEPA)
            frac = ema_state["step"] / max(1, ema_state["total"])
            model.ema_update(args.ema + (1.0 - args.ema) * frac)
            ema_state["step"] += 1
        n_steps += 1
        if (step + 1) % 50 == 0:
            print(f"  [{tag}] step {step + 1}/{len(loader)} reg={sums['reg'] / n_steps:.4f} "
                  f"jepa={sums['jepa'] / n_steps:.4f} {time.time() - t0:.0f}s", flush=True)
    return {k: v / max(1, n_steps) for k, v in sums.items()} | {"sec": time.time() - t0}


def make_optimizer(model, args, total_steps):
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters()
            if not n.startswith(("encoder.", "target.")) and p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": args.lr}, {"params": rest, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    return opt, sched


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_jepa = args.jepa_weight > 0 or args.pretrain_epochs > 0

    name = args.out or "runs/{}-{}-{}-jw{}-pt{}-{}-s{}".format(
        args.model.split("/")[-1], args.track, "+".join(args.langs or ["all"]),
        args.jepa_weight, args.pretrain_epochs, args.mask_strategy, args.seed)
    out_dir = Path(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files, dev_files, test_files = (load_split(args, s) for s in ("train", "dev", "test"))
    train = merge_duplicates([i for insts in train_files.values() for i in insts])
    if args.max_train and len(train) > args.max_train:
        train = random.Random(args.seed).sample(train, args.max_train)
    print(f"device={device} train={len(train)} dev={sum(map(len, dev_files.values()))} "
          f"test={sum(map(len, test_files.values()))} files={list(train_files)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    collate = Collator(tok, args.max_length, args.mask_strategy, args.mask_ratio, args.max_span, args.seed)
    model = DimAsrJepa(args.model, use_jepa, args.pred_dim, args.pred_layers).to(device)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=gen)

    history = []
    ft_jepa_steps = len(loader) * args.epochs if args.jepa_weight > 0 else 0

    # ---- этап 1: предобучение только с JEPA (метки не используются) ----
    if args.pretrain_epochs > 0:
        pre = list(train)
        if args.pretrain_on_eval_text:
            pre += [i for fs in (dev_files, test_files) for insts in fs.values() for i in insts]
        pre_loader = DataLoader(pre, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=gen)
        steps = len(pre_loader) * args.pretrain_epochs
        opt, sched = make_optimizer(model, args, steps)
        ema_state = {"step": 0, "total": steps + ft_jepa_steps}
        for ep in range(args.pretrain_epochs):
            stats = run_epoch(model, pre_loader, opt, sched, device, args, jepa_weight=1.0,
                              regress=False, ema_state=ema_state, tag=f"pretrain {ep + 1}")
            print(f"pretrain epoch {ep + 1}: jepa={stats['jepa']:.4f} target_std={stats['std']:.3f} "
                  f"({stats['sec']:.0f}s)", flush=True)
            history.append({"stage": "pretrain", "epoch": ep + 1, **stats})
    else:
        ema_state = {"step": 0, "total": ft_jepa_steps}

    # ---- этап 2: дообучение с метками (+ необязательный JEPA-лосс) ----
    opt, sched = make_optimizer(model, args, len(loader) * args.epochs)
    best = {"rmse": float("inf"), "epoch": 0, "state": None}
    for ep in range(args.epochs):
        stats = run_epoch(model, loader, opt, sched, device, args, jepa_weight=args.jepa_weight,
                          regress=True, ema_state=ema_state, tag=f"epoch {ep + 1}")
        dev_report, _ = evaluate(model, dev_files, collate, device, args)
        rmse = dev_report["ALL"]["rmse_va"]
        print(f"epoch {ep + 1}: reg={stats['reg']:.4f} jepa={stats['jepa']:.4f} "
              f"dev_rmse={rmse:.4f} ({stats['sec']:.0f}s)", flush=True)
        history.append({"stage": "finetune", "epoch": ep + 1, **stats, "dev_rmse_va": rmse})
        if rmse < best["rmse"]:
            keep = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    if k.startswith(("encoder.", "head."))}
            best = {"rmse": rmse, "epoch": ep + 1, "state": keep}

    # ---- тест на чекпойнте с лучшим dev ----
    model.load_state_dict(best["state"], strict=False)
    dev_report, _ = evaluate(model, dev_files, collate, device, args)
    test_report, test_preds = evaluate(model, test_files, collate, device, args)
    for fname, pred in test_preds.items():
        write_predictions(out_dir / "pred" / fname.replace("test_task1", "pred"), test_files[fname], pred)
    result = {"args": vars(args), "best_epoch": best["epoch"], "dev": dev_report,
              "test": test_report, "history": history}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"best epoch {best['epoch']}: dev_rmse={dev_report['ALL']['rmse_va']:.4f} "
          f"test_rmse={test_report['ALL']['rmse_va']:.4f} -> {out_dir}/result.json", flush=True)
    return result


if __name__ == "__main__":
    main()
