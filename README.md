# dimabsa-jepa

Testing JEPA-style latent prediction on [SemEval-2026 Task 3: DimABSA](https://github.com/DimABSA/DimABSA2026),
starting with **Subtask 1 (DimASR)**: given a text and an aspect, regress valence and arousal in [1, 9]
(metric: RMSE over VA). Track B (DimStance) uses the same format and works with `--track b`.

## The idea

A pretrained encoder reads `<s> aspect </s></s> text </s>` and a small head regresses (V, A) from the
mean-pooled states. On top of that baseline there is an I-JEPA-style objective:

- a set of **text tokens is hidden from the context encoder** (attention-masked, so no token can attend
  to them — they are removed from the view, not replaced by `[MASK]`);
- an **EMA target encoder** reads the full input and produces the latents of the hidden tokens;
- a narrow transformer **predictor** gets the context latents plus positional mask tokens and regresses
  the target latents (smooth-L1 on layer-normalised targets). Nothing is reconstructed in token space.

What gets hidden (`--mask-strategy`):

| strategy | hidden tokens |
|---|---|
| `random` | random contiguous blocks covering ~30% of the text |
| `opinion` | the annotated **opinion span** of that aspect (available in `*_train_alltasks` files), falling back to `random` when there is none. The model has to predict *the latent of the sentiment expression* from the aspect and the rest of the sentence. |

How it is used:

| regime | flags | loss |
|---|---|---|
| baseline | *(default)* | `L_reg` |
| JEPA auxiliary | `--jepa-weight 1.0` | `L_reg + λ·L_jepa`, jointly |
| JEPA pretraining | `--pretrain-epochs N` | `N` epochs of `L_jepa` (no labels), then fine-tuning |

`--pretrain-on-eval-text` additionally feeds dev/test *texts* (never labels) to the pretraining stage.

## Setup

```bash
uv venv && uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
./scripts/get_data.sh                   # clones the official repo into data/DimABSA2026
pytest -q
```

## Running

```bash
# baseline
python -m dimabsa_jepa.train --langs eng --model xlm-roberta-base --epochs 5

# JEPA auxiliary loss with opinion-span masking
python -m dimabsa_jepa.train --langs eng --model xlm-roberta-base --epochs 5 \
    --jepa-weight 1.0 --mask-strategy opinion

# baseline vs JEPA variants over seeds, with a summary table
python scripts/sweep.py --name eng-xlmr --seeds 0 1 2 \
    --variants baseline jepa-aux-random jepa-aux-opinion jepa-pretrain \
    -- --langs eng --model xlm-roberta-base --epochs 5
```

Omit `--langs` / `--domains` to train one multilingual model on everything. Each run selects the epoch
with the best dev RMSE, evaluates it on the gold test sets and writes
`runs/<name>/result.json` (per-file and pooled metrics) plus submission-format predictions in
`runs/<name>/pred/`. CUDA is used automatically when present (`--amp` for mixed precision).

RMSE is reported unnormalised, on the 1–9 scale (the official script divides by √128 when run with
normalisation); `tests/test_core.py` checks agreement with the official script.

## Results

No results yet — the first CPU pilot (English, `distilroberta-base`, 2k training instances, 3 seeds) is running.

## Layout

```
src/dimabsa_jepa/data.py     loading all task formats, block/opinion masking, submission writer
src/dimabsa_jepa/model.py    encoder + VA head + EMA target + JEPA predictor
src/dimabsa_jepa/train.py    training / evaluation entry point
scripts/sweep.py             variants × seeds runner and summary table
scripts/get_data.sh          fetch official data and evaluation script
```

## Not done yet

- Subtasks 2 and 3 (triplet / quadruplet extraction) — only Subtask 1 is implemented.
- Full-scale runs with a base/large multilingual encoder on all languages (needs a GPU).
