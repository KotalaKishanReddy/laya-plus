<div align="center">

# Laya-Plus · context-kit

**Stock Laya with Jev-scale context. Same weights. Same brain. 64× the memory.**

[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](./LICENSE.Apache-2.0)
[![Unofficial fork](https://img.shields.io/badge/fork-unofficial-grey)](./NOTICE)
[![Context](https://img.shields.io/badge/context-32k-green)]()
[![Kit](https://img.shields.io/badge/kit-single--file-orange)]()

_An unofficial community fork — not affiliated with ConvAI Innovations or TypeSafe AI._

</div>

**Jev** (TypeSafe AI) proved decisions beat text: state in, typed `choice` / `score` / `noul`
probabilities out — no parsing, no format hallucinations. **Laya** (Nandakishor Mukkunnoth /
ConvAI Innovations — [arXiv:2503.23303](https://arxiv.org/abs/2503.23303),
[arXiv:2510.01237](https://arxiv.org/abs/2510.01237)) made it open, $0, and ~33ms.
Both ride the same architecture class: a **bidirectional encoder with a
non-autoregressive decision head** (no token-by-token generation — one forward
pass, honest probabilities). This project improves that architecture, release by release.

## Releases

- **V1 — 32k context, technically not literally.** Stock Laya weights +
  retrieve-then-decide: chunk → rank → top-k single pass. Sees 32k of state,
  decides in one 512-token window. Zero weight changes.
- **V1.1 — native 8k: 8× more real context.** Continued training stretched
  Laya's native window 1k → 8k (encoder already supports 8192 positions).
  No retrieval step inside the window — genuinely longer single-pass decisions.

## The gap

| | Stock Laya | Jev | **This kit** |
|---|---|---|---|
| English window | 512 | 32,768 state | **32,768 effective** |
| Multilingual window | 1024 | 32,768 state | **32,768 effective** |
| Weights changed? | — | — | **No. Zero.** |

## How it works

1. **Native 8k.** The ModernBERT encoder already supports 8192 positions
   (`max_position_embeddings=8192`). Set `"max_len": 8192` in
   `rl_agent_config.json`. No retraining.
2. **32k retrieve-then-decide** (`src/laya_longctx.py`, one file, no new deps):
   chunk → TF-IDF rank vs your questions → top-3 inside a true-fit 1280-char
   state budget → single forward pass. Millisecond overhead.

## Use it (60 seconds)

**Step 1 — terminal (shell, not Python). Pick one:**
```bash
pip install laya scikit-learn          # classic
python -m pip install laya scikit-learn  # when `pip` isn't on PATH
uv pip install laya scikit-learn       # uv users
conda install scikit-learn && pip install laya  # conda envs (laya is PyPI-only)
# + copy src/laya_longctx.py from this repo next to your code
```

**Step 2 — Python:**
```python
from laya_longctx import LongContextRouter
router = LongContextRouter(preload=True)

res = router.predict(long_doc_or_dict, {   # up to ~32k tokens
    "queue": {"type": "choice", "instructions": "Which queue owns this?",
              "criteria": {"infra": "outages, downtime",
                           "billing": "refunds, SLA"}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["low", "medium", "high", "critical"]},
    "churn": {"type": "noul", "instructions": "Will they cancel?"},
})
print(res["answers"]["queue"]["choice"])  # branch on it directly
```

## Accuracy
Identical to base Laya by construction. See the base project's benchmarks —
we claim nothing beyond them.

## Status as of 2026-09-26
- **V1 (this release): STABLE.** Stock weights + 32k retrieve-then-decide. Kit, docs, site, donations live.
- **V1.1 native 8k: recipe redesign in progress.** Stage-1 did not teach retrieval
  (see table above); distractor dedupe + short replay + gold-location loss
  being built. No ETA claimed.
- **Research track:** td 0.7675 dual-beat (Jev + base); banking77 architecturally
  walled (cascade best 0.244, direct training chance) — inference-time and
  head-redesign tracks running.
- **Gate:** dual-target (beat Jev AND base on all suites) — currently CLOSED.
  Releases stay INTERIM-labeled until it passes.

## Native context: shape vs comprehension (measured)

Setting `"max_len": 8192` gives 8k **shape** — the model accepts 8k tokens.
Comprehension (actually deciding correctly at depth) must be trained and
measured. Needle-by-depth eval, 112 held-out items:

| Model @ length | acc | ECE | early / mid / late |
|---|---|---|---|
| base @ 512 (truncated) | 0.268 | 0.265 | 0.333 / 0.262 / 0.206 |
| base @ 2048 (zero-shot) | 0.286 | — | 0.389 / 0.286 / 0.176 |
| tuned @ 2048 (3ep full-FT) | 0.259 | 0.097 | 0.389 / 0.143 / 0.265 |

Honest verdict: stage-1 did **not** teach retrieval — tuned trails zero-shot
base on accuracy (mid-band collapse 0.143), though calibration improved
(ECE 0.265 → 0.097). Stage-2 (8192) held until the recipe is redesigned
(distractor dedupe, short-task replay, banking-head diagnostics). V1 wrapper
above remains the working 32k path.

## Research track (fine-tunes, same architecture)
- **td breakthrough:** full-77/budget-512 run scored **typed-decisions 0.7675**
  (Jev 0.727, base 0.766), AG 0.9154, Emotion 0.9095, ECE 0.031 — 5/6 vs Jev.
- **banking77 conclusive negative:** direct 77-way training scores chance
  (0.013) — the marker head cannot scale options; best remains shortlist
  cascade (0.244). Inference-time fix, not a training fix.

## Roadmap (rolling releases)
- [x] **V1 — 32k technically** (retrieve-then-decide over stock weights)
- [ ] **V1.1 — native 8k** (8× real context, in training now)
- [ ] **V2 — native 32k** (single-pass, no retrieval)
- [ ] Calibrated fine-tune releases

## Contributing
Open an issue with your use case — long-context evals, languages, integrations
welcome. PRs: keep it minimal, tested, and honest (no fabricated benchmarks).

## Credits & license
Base: **Nandakishor Mukkunnoth / ConvAI Innovations** (code, weights, papers —
see NOTICE). Additions: Apache-2.0. Base weights: ConvAI terms.
**Enhanced by Kotala Kishan Reddy.**

## Donate
Like the project? Fuel V1.1/V2 training compute:

[![Donate UPI](https://img.shields.io/badge/donate-UPI-green)](https://github.com/KotalaKishanReddy/laya-plus#donate)
**UPI:** `kishaniscool@upi` (any GPay/PhonePe/Paytm app) or
[upi://pay?pa=kishaniscool@upi&pn=Laya-Plus&cu=INR](upi://pay?pa=kishaniscool@upi&pn=Laya-Plus&cu=INR) on mobile.
