#!/usr/bin/env python3
"""Fetch 5 years of 1-hour BTC-USDT-SWAP data.

This utility leverages the existing DataCollector infrastructure to pull
historical OHLCV bars from OKX and store them as yearly Parquet partitions
under the canonical data directory (defaults to ./data/okx/swap/BTC/USDT:USDT/1h/).

Usage
-----
$ python data_pipeline/scripts/fetch_btc_swap_1h_5y.py

The script is idempotent – if you already have stored bars, they will be
merged and deduplicated; otherwise a fresh download will occur.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Ensure project root is on PYTHONPATH so that `data_pipeline` can be imported
# regardless of the current working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from data_pipeline.collectors.collector import DataCollector

# ---------------------------------------------------------------------------
# Configuration parameters
# ---------------------------------------------------------------------------
SYMBOL = "BTC/USDT:USDT"  # OKX perpetual swap symbol recognised by ccxt
TIMEFRAME = "1h"          # Desired bar interval
YEARS_BACK = 5            # Number of years of history to fetch
INSTRUMENT_TYPE = "swap"  # Ensure we store under data/okx/swap/
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the script."""
    # Determine date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=YEARS_BACK * 365)

    print(
        (
            f"🚀 Fetching {TIMEFRAME} OHLCV for {SYMBOL} "
            f"from {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d} (≈{YEARS_BACK} years)"
        )
    )

    dc = DataCollector()

    # ------------------------------------------------------------------
    # Pull data – this might take several minutes because the OKX REST
    # endpoint is paginated and we respect the rate-limit in DataCollector.
    # ------------------------------------------------------------------
    df = dc.fetch_historical_data(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=start_date,
        end_date=end_date,
        instrument_type=INSTRUMENT_TYPE,
    )

    if df.empty:
        print("❌ No data was returned – please verify network connectivity and symbol name.")
        return

    # Persist to Parquet partitioned by year (handled internally).
    dc.save_data(df, SYMBOL, TIMEFRAME, instrument_type=INSTRUMENT_TYPE)

    print(f"✅ Successfully fetched and stored {len(df):,} bars of {TIMEFRAME} data.")


if __name__ == "__main__":
    main() 