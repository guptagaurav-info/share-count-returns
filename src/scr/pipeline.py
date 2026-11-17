"""Run the study: quintile portfolios on 12-month net share issuance, statistics, figures and metrics.json."""
import json, os, yaml
import numpy as np, pandas as pd, duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
from scr import core as C

FIG = "reports/figures/"
NAMES = {1: "Shrinking most", 2: "2", 3: "3", 4: "4", 5: "Issuing most"}
BIGTECH = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]


def spy_returns(start):
    c = duckdb.connect()
    s = c.sql("select report_date d, close from 'data/raw/stock_prices.parquet' where symbol='SPY'").df()
    s["d"] = pd.to_datetime(s.d); s = s.set_index("d").close.astype(float).resample("ME").last()
    sd = pd.read_parquet(C.RAW + "stock_dividend_events.parquet"); sd = sd[sd.symbol == "SPY"]
    sd["d"] = pd.to_datetime(sd.report_date).dt.to_period("M").dt.to_timestamp("M")
    dv = pd.to_numeric(sd.amount).groupby(sd.d).sum().reindex(s.index).fillna(0)
    return ((s + dv) / s.shift(1) - 1).loc[start:]


def shift_to_holding(df, start):
    df = df.copy(); df.index = df.index + pd.offsets.MonthEnd(1)
    return df.loc[start:].dropna(how="all")


def main():
    cfg = yaml.safe_load(open("configs/default.yaml")); start = cfg["start"]; q = cfg["quintiles"]
    os.makedirs(FIG, exist_ok=True)
    px, nd, dv = C.price_panels()
    div = C.dividends(px.index, px.columns)
    r = C.total_returns(px, nd, div)
    sh = C.share_counts(px.index, px.columns, cfg["report_lag_months"])
    iss = C.net_issuance(sh)
    uni = C.universe(dv, cfg["universe_size"])
    elig = uni & iss.notna() & px.notna() & (iss.abs() < 2)  # |log change| >= 2 treated as a data error
    lab = C.quintile_labels(iss, elig, q)
    fwd = r.shift(-1)
    spy = spy_returns(start)

    P = shift_to_holding(C.portfolio_returns(lab, fwd, q), start)
    spy = spy.reindex(P.index)
    PV = shift_to_holding(C.portfolio_returns(lab, fwd, q, weights=px * sh), start)
    U = shift_to_holding(fwd.where(elig).mean(axis=1).to_frame("u"), start)["u"].reindex(P.index)
    ex5 = shift_to_holding(fwd.where(elig & (lab < q)).mean(axis=1).to_frame("x"), start)["x"].reindex(P.index)
    sec = pd.read_parquet("data/universe.parquet").set_index("symbol").sector.reindex(px.columns).dropna()
    PS = shift_to_holding(C.portfolio_returns(C.sector_neutral_labels(iss, elig, sec, q), fwd, q), start)
    lab10 = C.quintile_labels(iss, elig, 10)
    P10 = shift_to_holding(C.portfolio_returns(lab10, fwd, 10), start)

    cost = cfg["cost_bps_per_side"] / 1e4
    to1, to5 = C.turnover(lab.loc[start:], 1), C.turnover(lab.loc[start:], q)
    spr = P[1] - P[q]
    spr_net = spr - 2 * cost * (to1 + to5)  # buy and sell legs on both quintiles each month
    split = str(cfg["split_year"])
    med = {k: float(iss.where(lab == k).loc[start:].stack().median()) for k in range(1, q + 1)}

    # share of the universe that shrank its share count, by calendar year (December snapshots)
    shrink = (iss.where(elig) < 0).sum(axis=1) / elig.sum(axis=1)
    shrink_y = shrink[shrink.index.month == 12].loc[start:]
    shrink_y.index = shrink_y.index.year
    big = sh[BIGTECH].dropna(how="all")
    last = big.index[-1]
    bt5 = (big.loc[last] / big.loc[last - pd.DateOffset(years=5)] - 1).sort_values()

    m = {
        "sample": f"{P.index[0]:%Y-%m} to {P.index[-1]:%Y-%m}", "months": int(len(P)),
        "names_per_month": float(elig.loc[start:].sum(axis=1).mean()),
        "median_issuance_by_quintile": med,
        "ew_ann": {str(k): C.ann(P[k]) for k in P}, "vw_ann": {str(k): C.ann(PV[k]) for k in PV},
        "vol_ann": {str(k): float(P[k].std() * np.sqrt(12)) for k in P},
        "max_drawdown": {"q1": C.max_drawdown(P[1]), "q5": C.max_drawdown(P[q]), "spy": C.max_drawdown(spy)},
        "universe_ann": C.ann(U), "universe_ex_q5_ann": C.ann(ex5), "spy_ann": C.ann(spy),
        "spread": {"mean_ann": float(spr.mean() * 12), "t": C.nw_tstat(spr), "net_mean_ann": float(spr_net.mean() * 12),
                   "vw_mean_ann": float((PV[1] - PV[q]).mean() * 12), "vw_t": C.nw_tstat(PV[1] - PV[q]),
                   "sector_neutral_mean_ann": float((PS[1] - PS[q]).mean() * 12), "sector_neutral_t": C.nw_tstat(PS[1] - PS[q]),
                   "first_half_mean_ann": float(spr.loc[:str(int(split) - 1)].mean() * 12), "first_half_t": C.nw_tstat(spr.loc[:str(int(split) - 1)]),
                   "second_half_mean_ann": float(spr.loc[split:].mean() * 12), "second_half_t": C.nw_tstat(spr.loc[split:]),
                   "hit_rate_years": float(((1 + spr).groupby(spr.index.year).prod() > 1).mean())},
        "capm": {"q1": C.capm(P[1], spy), "q5": C.capm(P[q], spy), "spread": C.capm(spr, spy)},
        "legs": {"q1_minus_universe": float((P[1] - U).mean() * 12), "universe_minus_q5": float((U - P[q]).mean() * 12)},
        "deciles_ew_ann": {str(k): C.ann(P10[k]) for k in P10}, "decile10_median_issuance": float(iss.where(lab10 == 10).loc[start:].stack().median()),
        "d1_d10": {"mean_ann": float((P10[1] - P10[10]).mean() * 12), "t": C.nw_tstat(P10[1] - P10[10]), "capm": C.capm(P10[1] - P10[10], spy)},
        "turnover_monthly": {"q1": float(to1), "q5": float(to5)},
        "share_shrinking_by_year": {int(k): float(v) for k, v in shrink_y.items()},
        "bigtech_5y_share_change": {k: float(v) for k, v in bt5.items()}, "bigtech_asof": f"{last:%Y-%m}",
    }
    ex = spr[(spr.index.year < 2020) | (spr.index.year > 2022)]
    m["spread"]["ex_2020_2022_mean_ann"] = float(ex.mean() * 12); m["spread"]["ex_2020_2022_t"] = C.nw_tstat(ex)
    st = lab.loc[start:].stack().rename("q").reset_index(); st["sec"] = st.iloc[:, 1].map(sec)
    mix = pd.crosstab(st.sec, st.q, normalize="columns")
    m["sector_mix"] = {"q1": mix[1.0].round(3).to_dict(), "q5": mix[float(q)].round(3).to_dict()}
    yr = (1 + spr).groupby(spr.index.year).prod() - 1
    m["spread_by_year"] = {int(k): float(v) for k, v in yr.items()}
    # illustrative book: $1bn equal-weight book, dropping the top-issuance fifth
    m["illustrative"] = {"book": 1e9, "gain_ex_q5_per_year": float((C.ann(ex5) - C.ann(U)) * 1e9)}
    json.dump(m, open("reports/metrics.json", "w"), indent=2, default=float)
    figures(P, U, spy, m, shrink_y, bt5, q)
    print(json.dumps({k: m[k] for k in ["ew_ann", "spread", "capm", "legs"]}, indent=1, default=float))


def years_axis(ax):
    ax.xaxis.set_major_locator(MultipleLocator(4)); ax.xaxis.set_minor_locator(MultipleLocator(1))


def figures(P, U, spy, m, shrink_y, bt5, q):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    gold, navy, grey, red = "#B08D57", "#1F3A5F", "#8A8F98", "#A23B3B"
    # 1 quintile returns
    fig, ax = plt.subplots(figsize=(8, 4.6))
    vals = [m["ew_ann"][str(k)] for k in range(1, q + 1)]
    ax.bar([NAMES[k] for k in range(1, q + 1)], vals, color=[navy, grey, grey, grey, red])
    ax.axhline(m["spy_ann"], color=gold, lw=2, ls="--", label=f"S&P 500 (SPY) {m['spy_ann']:.1%}")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.1%}", ha="center")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Annual return, equal-weighted"); ax.set_xlabel("Fifths by 12-month change in share count")
    ax.set_title(f"Companies issuing the most shares lagged, {m['sample'][:4]}–{m['sample'][-7:-3]}")
    ax.legend(frameon=False, loc="upper right"); fig.tight_layout(); fig.savefig(FIG + "quintiles.png", dpi=160); plt.close(fig)
    # 2 growth of $1
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for s, lbl, c in [(P[1], "Shrinking most (top fifth)", navy), (U, "All 1,000 stocks", grey), (P[q], "Issuing most (bottom fifth)", red), (spy, "S&P 500 (SPY)", gold)]:
        w = (1 + s.dropna()).cumprod(); ax.plot(w.index.year + (w.index.month - 0.5) / 12, w.values, label=lbl, color=c, lw=2)
    ax.set_yscale("log"); ax.set_ylabel("Growth of $1 (log scale)"); years_axis(ax)
    ax.set_title("Growth of $1, total return, monthly rebalanced"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG + "growth.png", dpi=160); plt.close(fig)
    # 3 calendar-year spread
    spr = P[1] - P[q]; yr = (1 + spr).groupby(spr.index.year).prod() - 1
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(yr.index, yr.values, color=[navy if v > 0 else red for v in yr.values])
    ax.axhline(0, color="black", lw=0.8); years_axis(ax)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); ax.set_ylabel("Shrinking minus issuing, per year")
    ax.set_title(f"The gap paid in {m['spread']['hit_rate_years']:.0%} of years, with large swings"); fig.tight_layout(); fig.savefig(FIG + "by_year.png", dpi=160); plt.close(fig)
    # 4 share of universe shrinking + big tech
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    axs[0].plot(shrink_y.index, shrink_y.values, color=navy, lw=2, marker="o", ms=3)
    axs[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); years_axis(axs[0])
    axs[0].set_title("Share of the 1,000 largest stocks\nwith fewer shares than a year earlier", fontsize=11)
    axs[1].barh(bt5.index, bt5.values, color=[navy if v < 0 else red for v in bt5.values])
    axs[1].axvline(0, color="black", lw=0.8); axs[1].xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axs[1].set_title(f"Largest tech companies:\nshare count change, 5 years to {m['bigtech_asof']}", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG + "who_shrinks.png", dpi=160); plt.close(fig)





if __name__ == "__main__":
    main()
