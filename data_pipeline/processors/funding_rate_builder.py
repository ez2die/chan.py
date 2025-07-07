import pandas as pd
import time
import os
import sys

# Add project root to sys.path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_pipeline.collectors.okx_client import OKXClient
from logger import get_logger

logger = get_logger('funding_rate_builder')

class FundingRateBuilder:
    def __init__(self, historical_csv_path: str):
        self.historical_csv_path = historical_csv_path
        self.okx_client = OKXClient()

    def _load_historical_data(self) -> pd.DataFrame:
        """Loads and prepares historical funding rate data from the CSV file."""
        logger.info(f"Loading historical funding rates from {self.historical_csv_path}...")
        try:
            df = pd.read_csv(self.historical_csv_path)
            if 'timestamp' not in df.columns or 'funding_rate' not in df.columns:
                raise ValueError("CSV must contain 'timestamp' and 'funding_rate' columns.")
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('datetime')
            df = df[['funding_rate']].rename(columns={'funding_rate': 'fundingRate'})
            df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
            df.dropna(inplace=True)
            
            logger.info(f"Loaded {len(df)} historical records from {df.index.min()} to {df.index.max()}.")
            return df

        except FileNotFoundError:
            logger.error(f"Historical data file not found at: {self.historical_csv_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()

    def _fetch_recent_data(self, since_dt: pd.Timestamp) -> pd.DataFrame:
        """Fetches recent funding rate data from OKX since the given datetime."""
        if pd.isna(since_dt):
            logger.warning("Invalid 'since' timestamp for fetching recent data. Skipping.")
            return pd.DataFrame()

        since_ms = int(since_dt.timestamp() * 1000)
        logger.info(f"Fetching recent funding rates from OKX since {since_dt}...")
        
        try:
            df = self.okx_client.fetch_funding_rate('BTC-USDT-SWAP', since=since_ms)
            if not df.empty:
                logger.info(f"Fetched {len(df)} new records from {df.index.min()} to {df.index.max()}.")
            else:
                logger.info("No new records fetched from OKX.")
            return df
        except Exception as e:
            logger.error(f"Error fetching recent data from OKX: {e}")
            return pd.DataFrame()

    def build_full_history(self) -> pd.DataFrame:
        """Builds a complete funding rate history by combining historical and recent data."""
        df_hist = self._load_historical_data()
        
        last_historical_date = df_hist.index.max() if not df_hist.empty else None
        
        # Fetch data starting from the day after the last historical record
        fetch_since_date = last_historical_date + pd.Timedelta(days=1) if last_historical_date else None

        df_recent = pd.DataFrame()
        if fetch_since_date:
             df_recent = self._fetch_recent_data(since_dt=fetch_since_date)

        if df_hist.empty and df_recent.empty:
            logger.error("Failed to build funding rate history. No data available.")
            return pd.DataFrame()
            
        # Combine historical and recent data
        full_df = pd.concat([df_hist, df_recent])
        
        # Sort by index and remove duplicates, keeping the first entry
        full_df = full_df.sort_index()
        full_df = full_df[~full_df.index.duplicated(keep='first')]
        
        logger.info(f"Successfully built full funding rate history with {len(full_df)} records.")
        logger.info(f"Final date range: {full_df.index.min()} to {full_df.index.max()}.")
        
        return full_df

if __name__ == '__main__':
    # --- Example Usage ---
    # Path to the downloaded Kaggle dataset
    kaggle_csv_path = '/Users/sy/.cache/kagglehub/datasets/jesusgraterol/bitcoin-funding-rate-binance-futures/versions/4/funding_rate.csv'
    
    # Define a path to save the final combined data
    output_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'BTC_USDT_SWAP_funding_rate_full_history.parquet')

    # Build the full history
    builder = FundingRateBuilder(historical_csv_path=kaggle_csv_path)
    full_funding_rates = builder.build_full_history()

    # Save the result to a parquet file
    if not full_funding_rates.empty:
        full_funding_rates.to_parquet(output_path)
        logger.info(f"Full funding rate history saved to: {output_path}")

        # Verify by loading and printing info
        df_check = pd.read_parquet(output_path)
        print("\n--- Verification ---")
        print(f"Loaded {len(df_check)} records from {output_path}")
        print("Date Range:", df_check.index.min(), "->", df_check.index.max())
        print("Head:\n", df_check.head())
        print("Tail:\n", df_check.tail()) 