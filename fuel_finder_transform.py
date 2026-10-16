"""
Turn repeated Fuel Finder snapshots into analysis-ready data.

Your scraper appends the *current state* of every station on every run, so the
raw CSV is mostly repeated rows. This script turns it into:

  events.csv       one row per genuine price change (station, fuel, effective time, price)
  daily_panel.csv  one row per station x fuel x day: the price in force at a fixed
                   daily snapshot time, with staleness and data-quality flags
  qa_report.txt    counts of every problem found, for your data appendix

Usage:
    python fuel_finder_transform.py "data/events/*.csv" "data/snapshots/*_prices.csv.gz" --out out/
    (add your original fuel_prices.csv as another input if you have one)

Requires pandas (pip install pandas).
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

# ---- Settings you may want to change (state them in your data section) ----
SNAPSHOT_HOUR_UK = 12          # daily price = price in force at 12:00 UK time
PRICE_MIN, PRICE_MAX = 80, 300 # plausible pence-per-litre range for 2025-27
STALE_DAYS = 14                # flag prices not updated for this many days
JUMP_PP = 20                   # flag single changes larger than this (pence)
PETROL_ABOVE_DIESEL_PP = 5     # flag E10 more than this above B7 (likely swapped)


def normalise_price(p: pd.Series) -> pd.Series:
    """Fuel Finder mixes pounds (1.759), pence (175.9) and tenths of a penny (1759)."""
    p = pd.to_numeric(p, errors="coerce")
    p = p.where(p >= 10, p * 100)    # pounds -> pence
    p = p.where(p <= 1000, p / 10)   # tenths of a penny -> pence
    return p.round(1)


def load_raw(paths) -> pd.DataFrame:
    """Read one or more CSVs (plain or .gz) with the scraper's columns."""
    files = []
    for p in paths:
        files += sorted(glob.glob(p)) or [p]
    df = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    for col in ["fetched_at", "price_last_updated", "price_change_effective_timestamp"]:
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601", errors="coerce")
    df["price"] = normalise_price(df["price"])
    df["fuel_type"] = df["fuel_type"].str.strip().str.upper()
    # Use the effective timestamp as the event time; fall back to last-updated.
    df["effective_ts"] = df["price_change_effective_timestamp"].fillna(df["price_last_updated"])
    return df.drop(columns=["public_phone_number"], errors="ignore")


def build_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated snapshots into one row per price change."""
    ev = (raw.dropna(subset=["node_id", "fuel_type", "effective_ts", "price"])
             .sort_values("fetched_at")
             .drop_duplicates(["node_id", "fuel_type", "effective_ts", "price"], keep="first")
             .rename(columns={"fetched_at": "first_seen"}))
    ev = ev[["node_id", "trading_name", "fuel_type", "effective_ts", "price", "first_seen"]]
    ev = ev.sort_values(["node_id", "fuel_type", "effective_ts"]).reset_index(drop=True)

    # Reporting lag: how long after the change your scraper first saw it.
    # Only meaningful for changes made after your scraper started running.
    start = raw["fetched_at"].min()
    ev["report_lag_min"] = ((ev["first_seen"] - ev["effective_ts"]).dt.total_seconds() / 60)
    ev.loc[ev["effective_ts"] < start, "report_lag_min"] = pd.NA

    g = ev.groupby(["node_id", "fuel_type"])["price"]
    ev["change_pp"] = g.diff()
    # A re-submission of the same price gets a new timestamp but is not a price change.
    ev["is_price_change"] = ev["change_pp"].fillna(1) != 0
    ev["flag_out_of_range"] = ~ev["price"].between(PRICE_MIN, PRICE_MAX)
    ev["flag_jump"] = ev["change_pp"].abs() > JUMP_PP
    return ev


def build_daily_panel(ev: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Price in force at SNAPSHOT_HOUR_UK each day, carried forward from the last change."""
    last_day = raw["fetched_at"].max().tz_convert("Europe/London").normalize()
    first = ev.groupby(["node_id", "fuel_type"])["effective_ts"].min().reset_index()

    # Grid: each station-fuel from the day of its first known price to your last fetch.
    # A station's latest change before your first fetch is still valid from its
    # effective time, so you recover a little history before the scraper started.
    rows = []
    for r in first.itertuples(index=False):
        start = r.effective_ts.tz_convert("Europe/London").normalize()
        days = pd.date_range(start, last_day, freq="D")
        rows.append(pd.DataFrame({"node_id": r.node_id, "fuel_type": r.fuel_type, "date": days}))
    grid = pd.concat(rows, ignore_index=True)
    grid["snapshot_ts"] = (grid["date"] + pd.Timedelta(hours=SNAPSHOT_HOUR_UK)).dt.tz_convert("UTC")

    panel = pd.merge_asof(
        grid.sort_values("snapshot_ts"),
        ev.sort_values("effective_ts")[["node_id", "fuel_type", "effective_ts", "price", "flag_out_of_range"]],
        left_on="snapshot_ts", right_on="effective_ts",
        by=["node_id", "fuel_type"], direction="backward",
    ).dropna(subset=["price"])

    panel["days_since_update"] = (panel["snapshot_ts"] - panel["effective_ts"]).dt.days
    panel["flag_stale"] = panel["days_since_update"] > STALE_DAYS
    panel["date"] = panel["date"].dt.date

    # Was the station in your feed that day? Separates "price unchanged" from "dropped out".
    seen = raw.assign(date=raw["fetched_at"].dt.tz_convert("Europe/London").dt.date)[["node_id", "date"]]
    seen = seen.drop_duplicates().assign(in_feed=True)
    panel = panel.merge(seen, on=["node_id", "date"], how="left")
    panel["in_feed"] = panel["in_feed"].fillna(False).astype(bool)
    scraper_start = raw["fetched_at"].min().tz_convert("Europe/London").date()
    panel["after_scraper_start"] = panel["date"] >= scraper_start

    # Cross-fuel check: petrol far above diesel at the same station usually means swapped entries.
    wide = panel.pivot_table(index=["node_id", "date"], columns="fuel_type", values="price")
    if {"E10", "B7_STANDARD"} <= set(wide.columns):
        swapped = (wide["E10"] - wide["B7_STANDARD"]) > PETROL_ABOVE_DIESEL_PP
        panel = panel.merge(swapped.rename("flag_petrol_above_diesel").reset_index(),
                            on=["node_id", "date"], how="left")
        panel["flag_petrol_above_diesel"] = panel["flag_petrol_above_diesel"].fillna(False).astype(bool)

    cols = ["node_id", "fuel_type", "date", "price", "effective_ts", "days_since_update",
            "in_feed", "after_scraper_start", "flag_stale", "flag_out_of_range"]
    cols += [c for c in ["flag_petrol_above_diesel"] if c in panel.columns]
    return panel[cols].sort_values(["node_id", "fuel_type", "date"]).reset_index(drop=True)


def qa_report(raw: pd.DataFrame, ev: pd.DataFrame, panel: pd.DataFrame) -> str:
    lines = [
        f"Raw rows: {len(raw):,} from {raw['fetched_at'].nunique()} fetches "
        f"({raw['fetched_at'].min()} to {raw['fetched_at'].max()})",
        f"Stations: {raw['node_id'].nunique():,}; fuel types: {sorted(raw['fuel_type'].dropna().unique())}",
        f"Rows with unparseable price or timestamp: {int(raw['price'].isna().sum() + raw['effective_ts'].isna().sum())}",
        f"Price records: {len(ev):,}, of which actual price changes: {int(ev['is_price_change'].sum()):,}",
        f"  out of range ({PRICE_MIN}-{PRICE_MAX}p): {int(ev['flag_out_of_range'].sum())}",
        f"  jumps over {JUMP_PP}p: {int(ev['flag_jump'].sum())}",
        f"  median reporting lag (min, changes after scraper start): {ev['report_lag_min'].median()}",
        f"Daily panel rows: {len(panel):,}",
        f"  stale (> {STALE_DAYS} days since update): {int(panel['flag_stale'].sum()):,}",
        f"  station missing from your feed (days after scraper start): "
        f"{int((~panel['in_feed'] & panel['after_scraper_start']).sum()):,}",
    ]
    if "flag_petrol_above_diesel" in panel:
        lines.append(f"  E10 > B7 by more than {PETROL_ABOVE_DIESEL_PP}p: "
                     f"{int(panel['flag_petrol_above_diesel'].sum()):,}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="CSV files or glob patterns, e.g. 'data/events/*.csv'")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw(a.inputs)
    ev = build_events(raw)
    panel = build_daily_panel(ev, raw)
    report = qa_report(raw, ev, panel)

    ev.to_csv(out / "events.csv", index=False)
    panel.to_csv(out / "daily_panel.csv", index=False)
    (out / "qa_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
