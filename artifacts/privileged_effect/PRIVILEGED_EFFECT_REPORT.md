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

## Final dynamic-mechanism checkpoint

The online autoregressive gate is now complete. At every generated token, the correct and
dependency-broken references are evaluated on the arm's current prefix, and the paired residual
is recomputed before sampling the next token. The original single-problem matrix covered all
`4 layers x 4 alphas x 5 refresh schedules x 3 interventions = 240` configurations, plus correct
and corrupt baselines. It did not yield a strictly selective configuration and is a negative
generalization result.

Independent PRM800K validation first evaluated 100 same-prefix correct/incorrect step pairs.
Teacher-forced mediation replicated in late layers: layer-23 rescue was `+0.03705`
(`95% CI [0.01065, 0.06345]`) and necessity was `-0.02860`
(`[-0.04759, -0.00961]`); layer-25 rescue was `+0.05213`
(`[0.02846, 0.07580]`) and necessity was `-0.03580`
(`[-0.05653, -0.01507]`). A norm-matched random layer-25 intervention was null
(`-0.01192`, `[-0.02490, 0.00106]`).

Only five of the 100 PRM pairs had a correct-reference boxed answer and an incorrect-reference
boxed answer under 1,024-token free generation. The following online results are therefore
**conditional on behaviorally damaged examples selected by the baseline**, not an estimate of
population accuracy:

| Layer / alpha | Rescue | Sign-reversed | Norm-matched random | Aggregate recovery |
|---|---:|---:|---:|---:|
| 21 / 0.5 | 4/5 | 2/5 | 1/5 | 0.80 |
| 21 / 1.0 | 4/5 | 1/5 | 1/5 | 0.80 |
| 23 / 0.5 | 4/5 | 2/5 | 1/5 | 0.80 |
| **23 / 1.0** | **5/5** | **1/5** | **2/5** | **1.00** |
| 25 / 1.0 | 5/5 | 3/5 | 4/5 | 1.00 |

Layer 23 at alpha 1 is the cleanest oracle result. Layer 25 is not selective despite perfect
rescue because random perturbations also solve four of five examples. With only five selected
problems, bootstrap intervals are wide and paired superiority is not independently significant;
this is evidence that the dynamic mediator exists, not evidence for a deployable method.

A train-only rank-16 PCA projection fails the independent online gate. At both alpha 0.5 and 1,
projected rescue solves `0/5`; sign-reversed solves `1/5` and `2/5`, respectively, and random
solves `1/5` at each alpha. Thus the earlier teacher-forced partial retention does not transfer to
online generation. Static PCA16 and the causal contrastive eigenspace are both **NO-GO**.

Residual-persistence measurement uses 108 problems, six injection origins per problem, and all
128 future lags. The meaningful signal is target-projected recovery minus its norm-matched-random
control. It falls from `0.52659` at lag 0 to `0.07055` at lag 1 and `0.02236` at lag 2. The fitted
initial excess time constant is `0.633 token`, one-step excess retention is `13.4%`, the
interpolated half-life is `0.577 token`, and the e-fold crossing is `0.730 token`. A naive fit to
raw projected recovery gives an apparent `tau=205`, but that is a random-control floor and is
retained only as a legacy diagnostic. The causal correction has sub-token effective memory and
must be recurrently refreshed.

The held-out FPCA result is consistent with this decay: 64 temporal functions are needed for
about 90% held-out variance and all 128 for about 95%, whereas increasing spatial rank from 1 to
64 adds little at fixed temporal rank. The mediator is temporally complex rather than a fixed
low-dimensional steering vector.

## Updated decision

- Dynamic, prefix-matched privileged mediation: **mechanism GO**, conditional and oracle-only.
- Static direction, PCA16, generalized eigenspace, and one-shot/short-window steering:
  **NO-GO**.
- Deployable dynamic distillation or Qwen3-4B/5k training: **still NO-GO** until a learned field
  predicts the online residual on held-out prefixes without requiring correct and corrupt
  references at inference time.

The strict artifact audit passes all 12 checks, including the 240-arm matrix, deterministic random
controls, PRM teacher-forced and online validation, 108-problem decay, held-out subspace, and FPCA.

## Predictability gate: conditional projection is not enough

The first reference-free predictability gate uses Qwen3-4B and 1,000 unique PRM800K problems
(74,791 token states), with a strict 70/15/15 problem split. The primary analysis further restricts
to 598 pairs that contain an explicit downstream continuation, leaving 418/89/91 train/validation/
test problems. Student-visible layer-23 states predict the layer-23 paired oracle residual
`C_t = h_t^+ - h_t^-`; no privileged input enters any predictor at evaluation time.

Linear prediction is measurable but weak. On held-out tokens, Ridge obtains `R^2=0.00798` and
cosine `0.1311`; RRR-16/64 obtain `0.00769/0.00811`; CCA-16/64 obtain `0.00471/0.00444`.
Problem-mean R-squared values are approximately zero or negative, showing that the small positive
token-weighted score is concentrated in high-energy problems. This is not a lack of structure in
the target itself: oracle target PCA-16 and PCA-64 retain `16.9%` and `29.8%` of held-out residual
variance. Rather, the high-variance privileged structure is largely not linearly identifiable from
the deployment-visible state. Repeating the analysis on all 1,000 pairs gives the same ordering and
conclusion.

More importantly, a predicted residual must be evaluated as an intervention, not only as a
regression target. On all 91 held-out downstream problems, adding the exact oracle residual to the
unprivileged student state is not an oracle rescue: at alpha 1 it changes mean gold-continuation
log-prob by `-0.02280` (95% bootstrap CI `[-0.04546,-0.00244]`) and tail-16 log-prob by
`-0.05262`. The effect is dose dependent: alpha 0.25 is approximately null overall and already
negative on the tail; alpha 0.5 is null overall and negative on the tail. Thus the displacement
`h^+ - h^-` is causal in its original corrupt-privileged base point but is not translation-equivariant
to the unprivileged student base point.

The learned full fields inherit the same failure. At alpha 1, Mean, Ridge, RRR-64, and CCA-64
change mean gold log-prob by `-0.04617`, `-0.04359`, `-0.04337`, and `-0.04400`, respectively.
Norm-matched random intervention is null. Reversing RRR-64 instead gives `+0.03965`, but an
explicit reverse-mean control gives `+0.04366`; therefore this large effect is almost entirely a
generic global steering direction, not sample-specific conditional prediction.

After subtracting the train-set mean residual, the sample-dependent predictor has a small positive
effect: centered Ridge, RRR-64, and CCA-64 at alpha 1 give `+0.00316`, `+0.00266`, and `+0.00273`.
Their 95% intervals narrowly exclude zero for the overall continuation metric; only centered CCA-64
also has a positive tail-16 interval. These effects are too small to pass a deployability gate and
must be checked in free generation before being interpreted as transferable computation.

### Predictability decision

- Linear conditional predictability of the paired field: **weak positive, below gate**.
- Full oracle or learned field transported to the student base point: **NO-GO**.
- Centered sample-specific field: **small exploratory signal, not yet a method**.
- Qwen3-4B/5k distillation and MLP expansion: **do not launch** unless reference-free online
  generation validates the centered component.

The result refines the mathematical object required by a deployable method. Conditional expectation
`E[C_t | F_t^S]` minimizes residual prediction error, but it does not guarantee that the predicted
vector belongs to a beneficial intervention tangent space at `h_t^S`. A valid next target must be
student-anchored and outcome-aware (for example, a student-base causal effect or a locally learned
transport/Jacobian), rather than the paired privileged displacement alone.

A final 2,048-token reference-free online pilot does not rescue the gate. On the held-out problem,
student and corrupt arms truncate without an answer; centered RRR-64 produces the wrong answer
`180`; centered CCA-64 and the transported oracle both produce the same wrong answer `720`; and
centered Ridge, reverse mean, reverse-centered RRR, and random-centered RRR truncate. The dataset
gold answer is `5/16`. Because the transported oracle itself fails, no meaningful online oracle
recovery retention can be computed on this sample. Together with the complete 91-problem
teacher-forced intervention, this pilot is sufficient to stop the proposed RRR/CCA expansion rather
than launch a selectively screened free-generation matrix.

## Jacobian effect transport gate

The next diagnostic asks whether a privileged contrast should be transported through output space
rather than copied as a hidden vector. For one held-out next-token decision per problem, the source
finite effect is defined by actually patching `c = h+ - h-` at the corrupt-conditioned layer state.
Its outcome space contains the gold token plus the 15 logits with largest absolute source change.
At the student base point, a rank-16 explicit Jacobian and dual ridge solve produce the minimum-norm
local intervention that reconstructs this finite source effect. Ridge damping `0.01` was selected on
50 validation problems and then frozen for all 91 test problems. Layers 23 and 25 were evaluated.

The local geometry is real and the inverse solve works. Source linearization has mean cosine
`0.9750` at layer 23 and `0.9688` at layer 25 with the finite source effect. Applying the same vector
through the student Jacobian lowers cosine to `0.9492/0.9564` and raises relative squared error from
`0.1177/0.0947` to `0.2081/0.1615`; the paired gaps exclude zero. Nonlinear direct transport obtains
effect cosine `0.9101/0.9339`, whereas Jacobian transport obtains `0.9884/0.9916` and lowers relative
squared error from `0.2299/0.1737` to `0.0437/0.0284`. The transported intervention uses only about
`24.9%/24.6%` of the direct residual norm.

Outcome results require stratification because the privileged source intervention itself improves
the selected gold token on only 33/91 layer-23 examples and 31/91 layer-25 examples, harms it on
25/30, and is inert on 33/30. On the pre-specified source-positive stratum, Jacobian transport gives
gold-log-prob gains of `+0.1097` (95% CI `[0.0194,0.2578]`) at layer 23 and `+0.1115`
(`[0.0199,0.2516]`) at layer 25. Sign-reversed effects are `-0.2458/-0.2594`, while norm-matched
random effects are approximately zero. The transported intervention therefore reproduces both the
direction and task consequence of an oracle source effect.

However, this does **not** pass the proposed method gate. Direct residual transport on the same
source-positive examples is already positive: `+0.1058` at layer 23 and `+0.1345` at layer 25.
Jacobian-minus-direct is `+0.0039` (`[-0.0386,0.0383]`) and `-0.0230`
(`[-0.0901,0.0219]`), respectively. Across all 91 examples, Jacobian gains are only
`+0.0243/+0.0212`, with both confidence intervals crossing zero. Source-positive membership also
requires privileged information and cannot serve as a deployment-time gate.

### Jacobian-transport decision

- Base-point dependence in local effect geometry: **confirmed**, but modest.
- Low-dimensional Jacobian reconstruction of a source effect: **GO as a mechanism diagnostic**.
- Jacobian transport uniquely rescues cases where direct transport fails: **not established**.
- Reference-free/free-generation method gate: **NO-GO**; not run after the teacher-forced gate.

The precise conclusion is therefore narrower than “privileged effects are transportable.” A local
inverse can faithfully reproduce a specified source logit effect with a smaller student intervention,
but this experiment does not show a correctness benefit beyond direct patching, and the desired
effect remains privileged and outcome-selected. This is useful evidence about model geometry, not
yet a deployable distillation target.
