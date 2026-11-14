"""Метрики подзадачи 1. `rmse_va` совпадает с официальным скриптом оценки."""
from __future__ import annotations

import numpy as np


def rmse_va(pred: np.ndarray, gold: np.ndarray) -> float:
    """sqrt( sum_i ((Vp-Vg)^2 + (Ap-Ag)^2) / N ) для массивов формы (N, 2)."""
    pred, gold = np.asarray(pred, float), np.asarray(gold, float)
    return float(np.sqrt(((pred - gold) ** 2).sum(axis=1).mean()))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def va_report(pred: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    pred, gold = np.asarray(pred, float), np.asarray(gold, float)
    return {
        "rmse_va": rmse_va(pred, gold),
        "rmse_v": float(np.sqrt(((pred[:, 0] - gold[:, 0]) ** 2).mean())),
        "rmse_a": float(np.sqrt(((pred[:, 1] - gold[:, 1]) ** 2).mean())),
        "pcc_v": pearson(pred[:, 0], gold[:, 0]),
        "pcc_a": pearson(pred[:, 1], gold[:, 1]),
        "n": int(len(gold)),
    }
