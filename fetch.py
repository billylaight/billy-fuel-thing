"""
Fetch Fuel Finder prices and store only what changed.

Run every 30 minutes (GitHub Actions does this for you, see README.md). Each run:
  1. gets an access token and downloads every price batch;
  2. compares each station-fuel record with the last one seen (data/state/latest.csv)
     and appends new or changed records to data/events/YYYY-MM-DD.csv;
  3. on the first run of each UK day, also saves a full price snapshot and the
     station details (location, brand, flags) to data/snapshots/;
  4. logs the run in data/runs.csv, so you can document outages and gaps.

Credentials come from environment variables, never from the code:
    FUEL_FINDER_CLIENT_ID, FUEL_FINDER_CLIENT_SECRET
Endpoint URLs can be overridden with FUEL_FINDER_TOKEN_URL, FUEL_FINDER_PRICES_URL,
FUEL_FINDER_STATIONS_URL (use the values from your working script if they differ).
"""

import csv
import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

BASE = "https://www.fuel-finder.service.gov.uk"
TOKEN_URL = os.environ.get("FUEL_FINDER_TOKEN_URL", f"{BASE}/api/v1/oauth/generate_access_token")
PRICES_URL = os.environ.get("FUEL_FINDER_PRICES_URL", f"{BASE}/api/v1/pfs/fuel-prices")
STATIONS_URL = os.environ.get("FUEL_FINDER_STATIONS_URL", f"{BASE}/api/v1/pfs")
HEADERS_BASE = {"Accept": "application/json", "Content-Type": "application/json"}
if os.environ.get("FUEL_FINDER_USER_AGENT"):  # copy the one from your working script if it sets one
    HEADERS_BASE["User-Agent"] = os.environ["FUEL_FINDER_USER_AGENT"]

DATA = Path(os.environ.get("FUEL_DATA_DIR", "data"))
STATE = DATA / "state" / "latest.csv"
COLUMNS = ["fetched_at", "node_id", "trading_name", "fuel_type", "price",
           "price_last_updated", "price_change_effective_timestamp"]
KEY = ("node_id", "fuel_type")
COMPARE = ("price", "price_change_effective_timestamp")


def request_with_retry(method, url, tries=4, **kw):
    """Retry network errors, 429 and 5xx with backoff; the API has outages."""
    for i in range(tries):
        try:
            r = requests.request(method, url, timeout=60, **kw)
            if r.status_code not in (429, 500, 502, 503, 504):
                return r
        except requests.exceptions.RequestException:
            if i == tries - 1:
                raise
        time.sleep(5 * 2 ** i)
    return r


def get_token():
    cid = os.environ.get("FUEL_FINDER_CLIENT_ID", "").strip()
    secret = os.environ.get("FUEL_FINDER_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        raise RuntimeError("FUEL_FINDER_CLIENT_ID or FUEL_FINDER_CLIENT_SECRET is empty: "
                           "check the secret names in Settings > Secrets and variables > Actions")
    print(f"Token URL: {TOKEN_URL} | client_id starts {cid[:4]}..., length {len(cid)}; "
          f"secret length {len(secret)}")
    r = request_with_retry("POST", TOKEN_URL, headers=HEADERS_BASE,
                           json={"client_id": cid, "client_secret": secret})
    if not r.ok:
        body = r.text[:600].replace("\n", " ")
        looks_like_firewall = ("<html" in body.lower() or "cloudflare" in body.lower()
                               or "request blocked" in body.lower() or "access denied" in body.lower())
        hint = ("The reply is an HTML block page, so a firewall rejected this machine's IP "
                "before your credentials were checked." if looks_like_firewall else
                "The reply is from the API itself: check the URL and credentials against your local script.")
        raise RuntimeError(f"Token request HTTP {r.status_code} (server: {r.headers.get('server', '?')}). "
                           f"{hint} Body: {body}")
    body = r.json()
    payload = body.get("data", body) if isinstance(body, dict) else {}
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in response: {json.dumps(body)[:300]}")
    return token


def fetch_all(url, token):
    headers = {**HEADERS_BASE, "Authorization": f"Bearer {token}"}
    items, batch = [], 1
    while True:
        r = request_with_retry("GET", url, headers=headers, params={"batch-number": batch})
        if r.status_code == 404 and batch > 1:
            break
        r.raise_for_status()
        body = r.json()
        page = body.get("data", body) if isinstance(body, dict) else body
        if not page:
            break
        items.extend(page)
        if len(page) < 500:
            break
        batch += 1
        time.sleep(1)  # stay well inside the fair-use limits
    return items


def flatten(stations, fetched_at):
    """One row per station x fuel, same columns as your original CSV."""
    rows = []
    for st in stations:
        base = {k: v for k, v in st.items() if not isinstance(v, (list, dict))}
        prices = next((v for v in st.values()
                       if isinstance(v, list) and v and isinstance(v[0], dict)), [])
        for p in prices:
            row = {**base, **p, "fetched_at": fetched_at}
            rows.append({c: row.get(c, "") for c in COLUMNS})
    return rows


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, append=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or not append
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerows(rows)


def log_run(fetched_at, status, n_rows=0, n_new=0, note=""):
    path = DATA / "runs.csv"
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["fetched_at", "status", "price_rows", "new_events", "note"])
        w.writerow([fetched_at, status, n_rows, n_new, note[:200]])


def main():
    now = datetime.now(timezone.utc)
    fetched_at = now.isoformat(timespec="seconds")
    uk_day = now.astimezone(ZoneInfo("Europe/London")).strftime("%Y-%m-%d")

    try:
        token = get_token()
        rows = flatten(fetch_all(PRICES_URL, token), fetched_at)
    except Exception as e:  # record the failure, then fail the job so you notice
        log_run(fetched_at, "error", note=str(e))
        print(f"Run failed: {e}")
        sys.exit(1)

    # Keep only records that are new or changed since the last run
    latest = {(r["node_id"], r["fuel_type"]): r for r in read_csv(STATE)}
    new_events = [r for r in rows
                  if tuple(str(r[c]) for c in COMPARE) !=
                  tuple(str(latest.get((r["node_id"], r["fuel_type"]), {}).get(c, "")) for c in COMPARE)]
    for r in rows:
        latest[(r["node_id"], r["fuel_type"])] = r

    if new_events:
        write_csv(DATA / "events" / f"{uk_day}.csv", new_events, append=True)
    write_csv(STATE, list(latest.values()))

    # First run of the UK day: full snapshot (shows who is still reporting) + station details
    snap = DATA / "snapshots" / f"{uk_day}_prices.csv.gz"
    if not snap.exists():
        snap.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(snap, "wt", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        try:
            stations = fetch_all(STATIONS_URL, token)
            with gzip.open(DATA / "snapshots" / f"{uk_day}_stations.json.gz", "wt", encoding="utf-8") as f:
                json.dump(stations, f)
        except Exception as e:
            print(f"Station details failed (prices saved): {e}")

    log_run(fetched_at, "ok", len(rows), len(new_events))
    print(f"{fetched_at}: {len(rows)} price rows, {len(new_events)} new or changed")


if __name__ == "__main__":
    main()
