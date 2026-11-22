"""Summaries and paired statistics."""

import numpy as np
import pandas as pd

from offnav.analysis import bootstrap_ci, paired_table, sign_flip_p
from offnav.evaluate import _censored_median, summarize


def _rec(cid, suite, cat, success, first, ret):
    return {"case_id": cid, "suite": suite, "category": cat, "horizon": 120, "success": success,
            "first_success_step": first, "return": ret, "final_distance": 0.1 if success else 2.0,
            "path_length_to_success": 3.0 if success else np.nan}


def test_summarize_counts_failures():
    recs = [_rec(0, "pairs", "through", True, 50, 10), _rec(1, "pairs", "through", False, -1, 0),
            _rec(2, "pairs", "avoid", True, 30, 20), _rec(3, "benchmark", "benchmark", False, -1, 0)]
    s = summarize(recs)
    assert s["pairs"]["n_episodes"] == 3
    assert abs(s["pairs"]["success_rate"] - 2 / 3) < 1e-12
    assert s["pairs"]["mean_first_success_step_successes"] == 40
    # budget = 120 // 3 = 40 steps: the success at step 30 counts, the one at step 50 does not
    assert abs(s["pairs"]["success_rate_third_horizon"] - 1 / 3) < 1e-12
    assert s["pairs_through"]["success_rate"] == 0.5
    assert s["benchmark"]["success_rate"] == 0.0
    assert np.isnan(s["benchmark"]["mean_first_success_step_successes"])


def test_censored_median():
    assert _censored_median(np.array([10, 20, -1])) == 20
    assert np.isnan(_censored_median(np.array([10, -1, -1, -1])))


def test_sign_flip_p_and_bootstrap():
    assert sign_flip_p(np.array([1.0, 1, 1, 1, 1])) == 2 / 32
    assert sign_flip_p(np.array([1.0, -1, 1, -1])) == 1.0
    lo, hi = bootstrap_ci(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert 0.1 <= lo <= 0.3 <= hi <= 0.5


def test_paired_table_differences_by_seed():
    rows = []
    for seed in range(3):
        rows.append({"maze": "m", "method": "bc", "condition": "restricted", "seed": seed, "success_pairs": 0.5 + seed / 10})
        rows.append({"maze": "m", "method": "bc", "condition": "uniform", "seed": seed, "success_pairs": 0.6})
    t = paired_table(pd.DataFrame(rows), "condition", "restricted", "uniform", ["maze", "method"], ["success_pairs"])
    r = t.iloc[0]
    assert np.isclose(r.mean_diff, np.mean([-0.1, 0.0, 0.1]))
    assert r.n_negative == 1 and r.n_positive == 1 and r.n_zero == 1
    assert r.per_seed == "-0.1000;0.0000;0.1000"
