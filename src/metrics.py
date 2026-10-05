# -*- coding: utf-8 -*-
"""Statistics for the re-selection differential D = Q(S1) - Q(S2):
standard error, test statistic, and the three-way verdict.

Two standard-error conventions:
- Independent approximation (conservative upper bound, needs only Q1, Q2 and
  the test-set size n):
      SE_ind = sqrt( Q1(1-Q1)/n + Q2(1-Q2)/n )
- Paired McNemar (needs per-sample predictions, more precise):
      SE_mc  = sqrt(b+c)/n ,  D = (b-c)/n
  where b counts samples correct only under S1 and c only under S2.
  When only cached accuracies are available (no per-sample predictions),
  SE_mc falls back to SE_ind.

Verdict rule (noise band, threshold +-2, roughly the 95% level):
    t > +2   -> 'positive'  (D significantly positive)
    |t| <= 2 -> 'noise'    (indistinguishable from zero)
    t < -2   -> 'negative'  (D significantly negative)
"""
import numpy as np


def se_indep(q1, q2, n):
    q1 = np.clip(q1, 0.0, 1.0)
    q2 = np.clip(q2, 0.0, 1.0)
    return float(np.sqrt(q1 * (1 - q1) / n + q2 * (1 - q2) / n))


def se_mcnemar(pred1, pred2, y_true, n=None):
    """Paired McNemar standard error from S1/S2 test-set predictions.

    Returns NaN when per-sample predictions are missing; the caller then
    falls back to SE_ind.
    """
    if pred1 is None or pred2 is None:
        return float('nan')
    b = int(np.sum((pred1 == y_true) & (pred2 != y_true)))
    c = int(np.sum((pred1 != y_true) & (pred2 == y_true)))
    n = len(y_true) if n is None else n
    return float(np.sqrt(b + c) / n), b, c


def t_stat(delta, se):
    if se is None or se <= 0 or not np.isfinite(se):
        return float('nan')
    return float(delta / se)


def verdict(t, lo=-2.0, hi=2.0):
    if np.isnan(t):
        return 'n/a'
    if t > hi:
        return 'positive'
    if t < lo:
        return 'negative'
    return 'noise'
