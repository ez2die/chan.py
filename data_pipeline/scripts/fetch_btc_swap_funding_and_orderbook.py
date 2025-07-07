#!/usr/bin/env python3
"""Fetch BTC-USDT-SWAP funding-rate history (≈5 years) and a fresh order-book snapshot.

* Funding rates are saved under:
  data/okx/derivatives/funding_rates/BTC/USDT/
  – Parquet file per execution.
* Order-book snapshot is stored as JSON in:
  data/okx/derivatives/order_books/

Notes
-----
1. OKX API may limit the range of funding-rate history it returns. The script
   requests 1825 days (≈5 years); the exchange will return what it has.
2. Historical order-book data is not provided by OKX. We therefore capture a
   real-time depth snapshot (default depth = 40) at execution time.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone

# Ensure project root import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from data_pipeline.collectors.derivatives_collector import DerivativesCollector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOL = "BTC-USDT-SWAP"  # Funding-rate endpoint needs this hyphen style
START_YEAR = 2021          # Earliest year desired
# The OKX REST endpoint only returns ~90 days per call. We'll iterate backwards
# using the 'before' cursor until reaching START_YEAR or no more data.
ORDER_BOOK_DEPTH = 40      # Ask/Bid levels to pull
# ---------------------------------------------------------------------------

def fetch_funding_rates(dc: DerivativesCollector) -> None:
    """Collect funding-rate history back to START_YEAR by paging backwards."""

    from data_pipeline.collectors.okx_client import OKXClient
    import pandas as pd

    client = dc.client if hasattr(dc, "client") else OKXClient()

    print(f"📥 Back-filling {SYMBOL} funding-rate history back to {START_YEAR}…")

    all_records: list[dict] = []
    before: int | None = None  # ms timestamp cursor for pagination

    while True:
        try:
            params = {"before": before} if before else {}
            chunk = client.exchange.fetch_funding_rate_history(
                SYMBOL, limit=100, params=params
            )

            if not chunk:
                break

            # Convert dict list to dataframe
            df_chunk = pd.DataFrame(chunk)
            all_records.extend(chunk)

            # Earliest timestamp in this chunk
            earliest_ms = min(r["timestamp"] for r in chunk if "timestamp" in r)
            earliest_dt = datetime.fromtimestamp(earliest_ms / 1000, tz=timezone.utc)

            if earliest_dt.year <= START_YEAR:
                break

            # Set cursor to earliest_ms - 1 to fetch older records
            before = earliest_ms - 1
            # Respect rate limit
            time.sleep(0.2)
        except Exception as e:
            print(f"⚠️  Error paging funding rates: {e}")
            break

    if not all_records:
        print("⚠️  Unable to retrieve any funding-rate data (API restriction).")
        return

    # Consolidate and save via DerivativesCollector helper
    df = pd.DataFrame(all_records)
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        df.drop("timestamp", axis=1, inplace=True)

    # Delegate saving so path conventions remain consistent
    dc._save_funding_rate_data(SYMBOL, df)  # type: ignore

    first_ts, last_ts = df.index.min(), df.index.max()
    print(
        f"✅ Retrieved {len(df):,} rows. Range: {first_ts:%Y-%m-%d} → {last_ts:%Y-%m-%d}."
    )


def fetch_order_book(dc: DerivativesCollector) -> None:
    """Capture a live order-book snapshot and persist it."""
    print(f"📥 Capturing order-book snapshot for {SYMBOL} (depth {ORDER_BOOK_DEPTH})…")

    data = dc.collect_order_books(symbols=[SYMBOL], depth=ORDER_BOOK_DEPTH, save_data=True)

    if SYMBOL in data:
        ob = data[SYMBOL]
        bids, asks = len(ob.get("bids", [])), len(ob.get("asks", []))

        # Order-book timestamp could be ISO string (from our wrapper) or ms integer.
        raw_ts = ob.get("timestamp") or ob.get("T")
        if isinstance(raw_ts, (int, float)):
            ts = datetime.fromtimestamp(raw_ts / 1000, tz=timezone.utc)
        elif isinstance(raw_ts, str):
            try:
                ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        print(f"✅ Order-book snapshot recorded with {bids} bids & {asks} asks @ {ts:%Y-%m-%d %H:%M:%S%z}.")
    else:
        print("⚠️  Failed to capture order-book snapshot.")


def main() -> None:
    dc = DerivativesCollector()

    fetch_funding_rates(dc)
    fetch_order_book(dc)


if __name__ == "__main__":
    main() 