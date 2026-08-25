# CCSD Gate 0 pilot (NO-GO checkpoint)

The pilot tests whether a student-side correctness gradient has causal value
under free generation.  For each PRM800K same-prefix correct/incorrect pair we
used

`g = grad_h log p(correct_step) - grad_h log p(incorrect_step)`

and patched one Qwen3-4B hidden state at the selected layer.  Arms were base,
`+g`, `-g`, and norm-matched random.  Rewards are final boxed-answer exact
match after 128 generated tokens.  The implementation and command are in
`scripts/run_ccsd_gate0.py`; raw pilot outputs are kept in the adjacent JSONL
files.

## 50-node pilot (alpha = 0.1)

| layer | base | +g | -g | random |
|---:|---:|---:|---:|---:|
| 17 | 4/50 | 4/50 | 6/50 | 5/50 |
| 23 | 5/50 | 6/50 | 5/50 | 4/50 |

Paired +g-minus-base means were 0.00 (layer 17) and +0.02 (layer 23);
problem-bootstrap 95% intervals included zero.  Layer 17 was directionally
negative against both controls.

## Layer-23 alpha sweep (50 nodes)

| alpha | base | +g | -g | random | +g - base |
---:|---:|---:|---:|---:|---:|
| .025 | 4/50 | 6/50 | 5/50 | 5/50 | +.04, CI [0.00, .10] |
| .05  | 4/50 | 5/50 | 5/50 | 4/50 | +.02, CI [0.00, .06] |
| .2   | 5/50 | 4/50 | 5/50 | 4/50 | -.02, CI [-.06, 0.00] |

The corresponding +g-vs-random and +g-vs-minus intervals also included zero.
There is no stable positive dose-response, and no preregistered layer has a
bootstrap lower bound above zero.  The local next-step margin increases as
expected from the gradient definition, but this does not transfer to final
free-generation accuracy.

**Decision:** Gate 0 fails on this pilot.  CCSD causal-gradient mapping (CCA,
RRR, or canonical subspace fitting) is not started; NVIP remains archived as
NO-GO.  The checkpoint is intentionally retained as negative evidence rather
than expanded into a 500-node run.
