# NVIP node-advantage variance gate

## Question

Can the causal teacher advantage

\[
A_T=TV(p,q)\{\mathbb E_{\nu_+}R-\mathbb E_{\nu_-}R\}
\]

be estimated with substantially fewer generated tokens than long-horizon
binary terminal Monte Carlo?

## Frozen development set

Use exactly the first 26 nodes already present in
`artifacts/nvip/signed_advantage_audit_100.jsonl`. Do not add nodes selected by
their noisy advantage estimates. Node selection remains outcome-independent:
frontier problem plus high student/teacher total variation.

## Cross-fitted prefix value

Train a linear ridge head on frozen Qwen3-4B prefix activations from the other
screened problems. Every problem represented in the 26-node development set is
excluded before fitting. Select ridge strength on a problem-level validation
fold. Report held-out Brier score, AUROC, calibration mean, and base rate.

The value head is never trusted as an outcome label. For two independent
sampling pools, estimate

\[
\hat A_{DR}=\alpha\left[
  \frac1M\sum_{i=1}^M (V_i^+-V_i^-)
  +\frac1N\sum_{j=1}^N\{(R_j^+-R_j^-)-(V_j^+-V_j^-)\}
\right].
\]

The residual correction makes the estimator unbiased even when the value head
is miscalibrated. The two pools remain disjoint so their confidence intervals
can be combined with a simple union bound.

## Comparison

Compare at matched generated-token cost, not merely matched pair count:

1. terminal exact-match Monte Carlo;
2. short-horizon value only (diagnostic, potentially biased);
3. value plus independent terminal residual correction;
4. each of the above with and without common random numbers.

Sweep short horizons 16/32/64, proxy pairs 16/32, residual pairs 4/8, and
terminal pairs 8/16/32/64. Record confidence width, residual variance, sign,
abstention, generated tokens, and wall-clock time.

## Gates

Proceed to fresh-node detection only if:

- held-out value Brier improves at least 10% over the constant base-rate
  predictor, or value AUROC is at least 0.65;
- residual variance is at most half terminal signed-reward variance;
- at matched generated-token cost, the doubly robust confidence interval is at
  most 60% as wide as terminal Monte Carlo;
- decision/sign agreement is at least 80% against an independent high-budget
  reference.

Failure ends SC-NVIP training experiments. Passing the gate permits a new,
outcome-independent high-TV detection set. Detection, q-star construction, and
final p/q/q-star evaluation use mutually independent rollouts.
