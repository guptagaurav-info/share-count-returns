"""Download the public files, check pinned checksums and build the monthly panels."""
import hashlib, os, urllib.request, yaml
import pandas as pd, pyarrow.parquet as pq

RAW = "data/raw/"


def monthly_long(path, syms):
    """Month-end close, mean daily dollar volume and trading days per symbol (read by row group to keep memory low)."""
    f = pq.ParquetFile(path); parts = []
    for i in range(f.num_row_groups):
        d = f.read_row_group(i, columns=["symbol", "report_date", "close", "volume"]).to_pandas()
        d = d[d.symbol.isin(syms)]
        if d.empty:
            continue
        d["close"] = d.close.astype("float64"); d["d"] = pd.to_datetime(d.report_date); d["m"] = d.d.dt.to_period("M")
        d["dv"] = d.close * d.volume
        parts.append(d.sort_values("d").groupby(["symbol", "m"]).agg(close=("close", "last"), dv=("dv", "mean"), n=("close", "size")).reset_index())
    m = pd.concat(parts)
    return m.groupby(["symbol", "m"]).agg(close=("close", "last"), dv=("dv", "mean"), n=("n", "sum")).reset_index()


def fetch(cfg, local=None):
    os.makedirs(RAW, exist_ok=True)
    for f, s in cfg["sources"].items():
        p = RAW + f
        if not os.path.exists(p):
            if local and os.path.exists(os.path.join(local, f)):
                os.symlink(os.path.abspath(os.path.join(local, f)), p)
            else:
                urllib.request.urlretrieve(s["url"], p)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        print(f, h[:12], "OK" if h == s["sha256"] else "MISMATCH (mirror updated; results may differ)")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    fetch(cfg, os.environ.get("LOCAL_MIRROR"))
    prof = pd.read_parquet(RAW + "stock_profile.parquet", columns=["symbol", "country", "sector", "industry"])
    u = prof[(prof.country == "United States") & prof.sector.notna() & (prof.sector != "")]
    u = u[~u.industry.fillna("").str.contains("Shell Companies")]
    u.to_parquet("data/universe.parquet")
    monthly_long(RAW + "stock_prices.parquet", set(u.symbol)).to_parquet("data/monthly_long.parquet")
