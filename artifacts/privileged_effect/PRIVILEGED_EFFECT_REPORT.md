# Privileged-Effect and Paired-Rescue Checkpoint

## Decision

- Original non-aligned PCT / OT / FGW hypothesis: **NO-GO**.
- Mechanistic hypothesis of dependency-sensitive causal mediation: **preliminary GO**.
- Qwen3-4B/5k causal-subspace distillation: **NO-GO pending further gates**.

## Main observation

Reference conditioning produces a large, causally active context effect, but most of that
effect is not specific to reference correctness. With matched corruptions, the correct-specific
cosine margin is only about 0.02--0.026 in the strongest layers.

The more precise question is therefore not whether a universal correctness direction exists,
but whether the model contains a causal correction representation for behaviorally consequential
dependency errors.

## Matched-control diagnostic

Three controls were generated for 500 OPSD problems:

- locally corrupted equation/operator;
- coherent counterfactual rationale;
- dependency-swapped premise.

Strict automatic patch validation retained 108 problems for which all three controls were valid.
Across layers 3--35, correct/correct minus correct/control cosine margins peaked around 0.02--0.026
for residual, K, and V effects. This is much smaller than separation against the previous unmatched
wrong-reference control.

## Paired counterfactual rescue

For a correct reference and its matched corrupted reference, define

```text
c_l,t = h_l,t(correct) - h_l,t(corrupt)
```

Sufficiency patches `h(corrupt) + c`; necessity patches `h(correct) - c`. The outcome in this
checkpoint is mean teacher-forced log-probability over 128 gold continuation tokens. Results use
108 problems and 2,000 problem-level bootstrap replicates.

Only dependency swap produced a clear pre-intervention behavioral gap:

| Control | Correct minus control log-prob | 95% CI |
|---|---:|---:|
| Locally corrupted | 0.00035 | [-0.00282, 0.00393] |
| Counterfactual rationale | 0.00500 | [-0.00016, 0.01216] |
| Dependency swap | **0.04172** | **[0.03108, 0.05375]** |

Dependency-swap residuals showed bidirectional causal mediation in layers 23 and 25:

| Layer | Rescue gain | 95% CI | Necessity change | 95% CI |
|---:|---:|---:|---:|---:|
| 21 | 0.00464 | [-0.00113, 0.01077] | **-0.00845** | **[-0.01528, -0.00206]** |
| 23 | **0.00927** | **[0.00242, 0.01648]** | **-0.01125** | **[-0.01915, -0.00426]** |
| 25 | **0.01356** | **[0.00509, 0.02299]** | **-0.01643** | **[-0.02555, -0.00810]** |

At layer 25, rescue recovers about 32.5% of the observed correct-versus-dependency-swap gap;
necessity removes an effect equivalent to about 39.4% of that gap. These are descriptive
effect-recovery fractions, not formal natural indirect-effect estimates.

Locally corrupted and generated counterfactual controls did not show the same mediation pattern.
Because those controls also caused no significant behavioral damage, the working hypothesis is:

> A correct-minus-corrupt residual becomes causally useful when the corruption is behaviorally
> consequential, rather than merely because it is labeled incorrect.

## Required next gates

1. Test per-example correlation between total behavioral damage and rescue/necessity effect,
   controlling for corruption category.
2. Patch short token windows and evaluate gold-versus-corrupt answer margin and free-generation
   exact-match accuracy.
3. Learn an active-versus-inert causal contrastive subspace on training problems and evaluate
   projected interventions on held-out problems.

Formal distillation should begin only if the short-window or held-out-subspace gates pass.

## Follow-up: effect size, temporal support, and held-out structure

The per-example analysis pools all 324 matched pairs (108 problems times three corruption
categories). At layer 25, total behavioral damage strongly predicts both rescue and necessity:

- Pearson `TE -> rescue = 0.858` and `TE -> -necessity = 0.902`;
- after controlling for corruption category, standardized damage coefficients are `0.696`
  (95% CI `[0.549, 0.793]`) for rescue and `0.697` (`[0.611, 0.742]`) for necessity.

The association rises across depth and is therefore not explained only by the discrete corruption
label. This supports the narrower claim that the residual tracks behavioral consequence.

Short-window teacher-forced patches do **not** rescue behavior. At layer 25, windows 1 and 4 are
approximately null; windows 8, 16, and 32 move significantly in the wrong direction. Only the
full 128-token intervention yields positive rescue (`+0.01356`) and negative necessity
(`-0.01643`). The current evidence therefore rejects a one-shot static correction vector and
motivates an online, prefix-matched, time-varying residual field.

On a fixed problem-level 70/30 split, the full held-out residual remains causal (`n=33`, rescue
`+0.01705`, 95% CI `[0.00583, 0.02941]`; necessity `-0.01840`,
`[-0.03058, -0.00716]`). A train-only PCA rank-16 projection preserves `+0.00717` rescue and
`-0.00684` necessity, about 42% and 37% of the full effects respectively. The proposed
active-versus-inert generalized eigenspace does not show stable held-out rescue and is therefore
a negative result, not a selected method.

Separable functional PCA further distinguishes spatial compression from temporal compression.
After subtracting the train-set time-varying mean, one hidden-space component explains about 95%
of held-out energy only when all 128 learned time functions are retained. One time function plus
one spatial direction explains only 0.8%; 32 time functions explain about 69%, and 64 explain about
90%. Thus the large shared spatial component does not imply a static, temporally transferable
steering direction. Linear one-step dynamics also fit poorly (`R^2 = 0.0094`), while the low raw
Hankel rank is dominated by the same nuisance component.

The decisive remaining gate is online autoregressive intervention with residuals recomputed on
the arm's actual prefix, evaluated by mathematical boxed-answer equivalence. Correct, corrupt,
rescue, sign-reversed, and norm-matched-random arms plus layer/alpha/refresh sweeps are running;
PRM800K paired-step validation and residual-persistence estimation follow automatically.
