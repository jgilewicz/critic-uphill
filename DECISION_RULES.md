# Decision rules for E1 and E2

Fixed on 2026-11-21, before the first E1 or E2 job. Do not edit after the
first submission. Anything not covered here is descriptive only.

## Conditions

All EvoRainbow conditions use `algorithm=evorainbow algorithm.agent=td3`.

| name | Hydra overrides |
|---|---|
| TD3 | `algorithm=td3` |
| ERL | `algorithm.eval.mode=full` |
| critic (always surrogate) | `algorithm.eval.mode=critic algorithm.eval.p_real=0` |
| random gate (SC-ERL) | `algorithm.eval.mode=critic`, per-env `eval.p_real` preset |
| prefix h=200 | `algorithm.eval.mode=prefix algorithm.eval.h=200` |

## Score

- Per run: `perf/eval_rl` of the final evaluation (10 deterministic
  episodes after the last update, 1M step budget).
- Normalized per env: run score / IQM of the TD3 runs in that env, so
  TD3 sits at 1.0 and 10% of the TD3 score is 0.10.
- Per condition: IQM of normalized scores over all runs (envs x seeds).
- Difference: IQM(A) - IQM(B). 95% CI by percentile bootstrap, 10 000
  resamples, seeds resampled within each env.

## Primary comparisons

1. critic vs random gate
2. critic vs TD3
3. prefix h=200 vs TD3

- p-value: two-sided bootstrap p for difference = 0, from the same
  resamples.
- Holm over the three p-values at family-wise alpha = 0.05.

## Verdict per comparison

1. **Equivalent**: the 95% CI of the difference lies inside [-0.10, 0.10].
2. **Different**: otherwise, if Holm rejects. The sign gives the direction.
3. **Inconclusive**: neither.
