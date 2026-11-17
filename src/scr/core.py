"""Net share issuance and subsequent stock returns: panels, quintile portfolios and statistics."""
import numpy as np
import pandas as pd

RAW = "data/raw/"


def price_panels():
    m = pd.read_parquet("data/monthly_long.parquet")
    m["date"] = m.m.dt.to_timestamp("M")
    px = m.pivot_table(index="date", columns="symbol", values="close").sort_index()
    nd = m.pivot_table(index="date", columns="symbol", values="n").sort_index()
    dv = m.pivot_table(index="date", columns="symbol", values="dv").sort_index()
    return px, nd, dv


def dividends(index, columns):
    d = pd.read_parquet(RAW + "stock_dividend_events.parquet")
    d = d[d.symbol.isin(columns)]
    d["amount"] = pd.to_numeric(d.amount, errors="coerce").fillna(0.0)
    d["date"] = pd.to_datetime(d.report_date).dt.to_period("M").dt.to_timestamp("M")
    dm = d.groupby(["date", "symbol"]).amount.sum().unstack()
    return dm.reindex(index=index, columns=columns).fillna(0.0)


def total_returns(px, nd, div, max_up=3.0):
    """Monthly total return (split-adjusted price plus cash dividends); 10+ trading days required; >+300% treated as a bad print."""
    r = (px + div) / px.shift(1) - 1
    r = r.where(nd >= 10)
    return r.mask(r > max_up)


def share_counts(index, columns, lag_months=3):
    """Split-adjusted shares outstanding, as known `lag_months` after the report date (filing lag), sampled at month-ends."""
    s = pd.read_parquet(RAW + "shares_outstanding.parquet")
    s = s[s.symbol.isin(columns) & (s.shares_outstanding > 0)]
    s["date"] = pd.to_datetime(s.report_date).dt.to_period("M").dt.to_timestamp("M")
    sm = s.sort_values("report_date").groupby(["date", "symbol"]).shares_outstanding.last().unstack()
    sm = sm.reindex(index=sm.index.union(index)).sort_index().ffill(limit=6)  # stale beyond six months -> missing
    sm = sm.reindex(index=index, columns=columns)
    return sm.shift(lag_months)


def net_issuance(shares, months=12):
    """Log change in split-adjusted shares over the past year: negative = net buybacks, positive = net issuance."""
    return np.log(shares / shares.shift(months))


def universe(dv, n=1000):
    """Each month the n largest names by trailing 12-month average daily dollar volume (known at month-end)."""
    t = dv.rolling(12, min_periods=10).mean()
    rank = t.rank(axis=1, ascending=False)
    return rank <= n


def quintile_labels(signal, eligible, q=5):
    """Quintile 1 = largest share-count reduction, q = largest issuance; ranks within each month."""
    s = signal.where(eligible)
    pct = s.rank(axis=1, pct=True)
    return np.ceil(pct * q).clip(1, q)


def sector_neutral_labels(signal, eligible, sectors, q=5):
    s = signal.where(eligible)
    out = pd.DataFrame(np.nan, index=s.index, columns=s.columns)
    for sec, cols in sectors.groupby(sectors).groups.items():
        cols = [c for c in cols if c in s.columns]
        if len(cols) < 2 * q:
            continue
        pct = s[cols].rank(axis=1, pct=True)
        out[cols] = np.ceil(pct * q).clip(1, q)
    return out


def portfolio_returns(labels, fwd, q=5, weights=None):
    """Next-month return of each quintile (equal-weighted, or weighted by `weights` known at formation)."""
    out = {}
    for k in range(1, q + 1):
        mask = labels == k
        if weights is None:
            out[k] = fwd.where(mask).mean(axis=1)
        else:
            w = weights.where(mask & fwd.notna())
            out[k] = (fwd * w).sum(axis=1) / w.sum(axis=1)
    return pd.DataFrame(out)


def turnover(labels, k):
    """Average monthly one-way turnover of an equal-weighted quintile (share of names replaced)."""
    m = (labels == k).astype(float)
    w = m.div(m.sum(axis=1), axis=0)
    return (w - w.shift(1)).abs().sum(axis=1).div(2).iloc[1:].mean()


def ann(r):
    r = r.dropna()
    return (1 + r).prod() ** (12 / len(r)) - 1


def nw_tstat(x, lags=6):
    """Newey-West t-statistic of the mean of a monthly series."""
    x = pd.Series(x).dropna().values
    n = len(x); e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(v / n)


def capm(y, mkt):
    d = pd.concat([y, mkt], axis=1).dropna()
    X = np.column_stack([np.ones(len(d)), d.iloc[:, 1].values])
    b, *_ = np.linalg.lstsq(X, d.iloc[:, 0].values, rcond=None)
    resid = d.iloc[:, 0].values - X @ b
    a_t = nw_tstat(resid + b[0])
    return {"alpha_ann": float(b[0] * 12), "beta": float(b[1]), "alpha_t": float(a_t)}


def max_drawdown(r):
    w = (1 + r.dropna()).cumprod()
    return float((w / w.cummax() - 1).min())
