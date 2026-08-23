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
