import numpy as np, pandas as pd, sys
sys.path.insert(0, "src")
from scr.core import net_issuance, quintile_labels, portfolio_returns, nw_tstat, turnover

IDX = pd.date_range("2010-01-31", periods=24, freq="ME")


def test_issuance_sign():
    s = pd.DataFrame({"buy": np.linspace(100, 90, 24), "iss": np.linspace(100, 120, 24)}, index=IDX)
    i = net_issuance(s).iloc[-1]
    assert i["buy"] < 0 < i["iss"]


def test_quintile_one_is_most_shrinking():
    sig = pd.DataFrame([np.arange(10.0)], index=IDX[:1], columns=list("abcdefghij"))
    lab = quintile_labels(sig, sig.notna(), 5)
    assert lab.iloc[0, 0] == 1 and lab.iloc[0, -1] == 5


def test_equal_weight_portfolio_and_no_lookahead():
    lab = pd.DataFrame({"a": [1, 1], "b": [5, 5]}, index=IDX[:2])
    fwd = pd.DataFrame({"a": [0.10, 0.0], "b": [-0.05, 0.0]}, index=IDX[:2])
    p = portfolio_returns(lab, fwd, 5)
    assert abs(p.loc[IDX[0], 1] - 0.10) < 1e-12 and abs(p.loc[IDX[0], 5] + 0.05) < 1e-12


def test_turnover_zero_when_static():
    lab = pd.DataFrame({"a": [1] * 5, "b": [1] * 5}, index=IDX[:5])
    assert turnover(lab, 1) == 0


def test_tstat_large_for_constant_positive():
    x = 0.01 + np.random.default_rng(0).normal(0, 1e-4, 200)
    assert nw_tstat(x) > 50
