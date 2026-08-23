# PCT Gate-0 Diagnostic Report

Date: 2026-08-23

## Decision

**NO-GO for Qwen3-4B/5k training under the current protocol.**

The 500-problem diagnostic does not satisfy the preregistered gate as a whole. Correct-reference
flows can be separated from an unrelated wrong-reference control, but the strongest separation is
provided by Euclidean/PHF in the training-aligned conditioned-flow setting. OT/FGW/UOT do not beat
Euclidean by the required 0.05 margin, and every tested metric fails on the shuffled-reference
contrast.

## Data

- Source: `siyanzhao/Openthoughts_math_30k_opsd`, first 500 rows.
- Model used to create two additional references: `Qwen/Qwen3-4B`.
- Per problem: source `solution`, source `COT_Reason`, two generated alternatives, one unrelated
  next-problem solution as the wrong control, and one deterministically shuffled source solution.
- Automated audit: 500/500 rows pass reference count, source/generated answer markers, uniqueness,
  lexical-similarity threshold (<0.8), and control-presence/change checks.
- Important limitation: automated final-answer and lexical checks do not prove that all generated
  derivations are mathematically valid or genuinely use independent strategies. Manual strategy
  verification remains required before treating the dataset as paper-grade.

Verified candidate file: `multiref_500_verified_final.jsonl`.

## UOT Objective Fix

Training now optimizes the full entropic UOT objective:

`<C, pi> + epsilon KL(pi || a x b) + rho KL(pi 1 || a) + rho KL(pi^T 1 || b)`

The logger separately records raw transport cost, mass-normalized matched cost, and transported
mass. A Qwen3-0.6B smoke illustrates why this matters:

| Method | Raw cost | Normalized cost | Mass | Full PCT loss |
|---|---:|---:|---:|---:|
| Balanced OT | 0.28335 | 0.28335 | 1.00000 | 0.23920 |
| UOT | 0.22433 | 0.27249 | 0.84501 | 0.30325 |

The raw UOT transport cost is lower partly because it transports less mass; the complete objective
does not reward that reduction for free.

## Reference-Trajectory Diagnostic

Configuration: Qwen3-4B last-layer normalized hidden-state differences, at most 32 flow atoms,
Sinkhorn epsilon 0.05, 40 iterations, UOT rho 0.5, FGW 4 outer iterations. AUROC is computed within
each problem and macro-averaged; confidence intervals are 2,000 problem-level bootstrap samples.

### Correct/correct versus correct/wrong

| Metric | Macro AUROC | 95% CI |
|---|---:|---:|
| Euclidean | 0.540 | [0.523, 0.556] |
| PHF cosine | 0.524 | [0.507, 0.540] |
| Sinkhorn | 0.644 | [0.628, 0.659] |
| FGW | 0.649 | [0.634, 0.665] |
| UOT objective | 0.632 | [0.617, 0.648] |
| UOT raw cost | **0.675** | **[0.661, 0.690]** |
| UOT normalized cost | 0.657 | [0.641, 0.672] |
| UOT mass | 0.632 | [0.616, 0.647] |

The best confidence interval remains below the 0.70 gate.

### Correct/correct versus correct/shuffled

All macro AUROCs are between 0.378 and 0.539. OT/FGW/UOT are below 0.5, so the structured methods
rank shuffled controls as closer rather than farther. The current FGW relation uses pairwise hidden
feature distances and is largely permutation-invariant; it does not explicitly encode temporal
position or adjacency.

## Conditioned-Continuation Diagnostic

This variant matches the training loss more closely: every teacher sees a different privileged
reference but is evaluated on the same 128-token student continuation.

### Correct/correct versus correct/wrong

| Metric | Macro AUROC | 95% CI |
|---|---:|---:|
| Euclidean | **0.811** | **[0.797, 0.825]** |
| PHF cosine | 0.801 | [0.785, 0.815] |
| Sinkhorn | 0.693 | [0.677, 0.708] |
| FGW | 0.685 | [0.671, 0.701] |
| UOT objective | 0.700 | [0.686, 0.716] |
| UOT raw cost | 0.713 | [0.699, 0.728] |
| UOT normalized cost | 0.712 | [0.698, 0.728] |
| UOT mass | 0.700 | [0.686, 0.715] |

UOT raw/normalized cost narrowly cross 0.70 as point estimates, and UOT transported mass is higher
for correct/correct (1.15368) than correct/wrong (1.14735). However, neither structured metric beats
Euclidean, let alone by the required +0.05.

### Correct/correct versus correct/shuffled

All macro AUROCs are between 0.392 and 0.419. This contrast fails decisively.

## Local Training and Evaluation Smoke

Qwen3-0.6B one-step LoRA training completed for `none`, `phf_single`, `phf_mean`, `phf_set`,
`set_ot`, `set_uot`, and `set_fgw` while the Qwen3-4B candidate generator was also resident on the
same RTX 4090. The Transformers evaluation backend was fixed to load PEFT adapters and verified on
one AIME24 problem with `val_n=1` and 8 generated tokens. Its 0/1 score has no accuracy meaning.

## Next Scientific Step

Do not start the 5k training matrix from these results. First resolve the failed shuffled-control
hypothesis and manually verify reference strategy quality. A defensible next diagnostic would compare
the current permutation-invariant FGW relation with a preregistered order-sensitive temporal/adjacency
relation; that is a new method change and should be evaluated as a fresh diagnostic rather than
silently substituted into the current result.
