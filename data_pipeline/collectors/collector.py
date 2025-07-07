#!/usr/bin/env python3
"""
数据收集器主模块
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import json
import time
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import settings
from logger import data_collector_logger as logger
from .okx_client import OKXClient


class DataCollector:
    """Main data collector for fetching and storing market data."""
    
    def __init__(self):
        """Initialize the data collector."""
        self.client = OKXClient()
    
    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        instrument_type: str = "spot"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            start_date: Start date for data collection
            end_date: End date for data collection
            instrument_type: Type of instrument (spot/swap)
            
        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            logger.info(f"Fetching historical data for {symbol} {timeframe}")
            
            # Validate inputs
            if not self.client.validate_symbol(symbol):
                raise ValueError(f"Invalid symbol: {symbol}")
            
            if not self.client.validate_timeframe(timeframe):
                raise ValueError(f"Invalid timeframe: {timeframe}")
            
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now(timezone.utc)
            
            if start_date is None:
                start_date = end_date - timedelta(days=30)  # Default to 30 days
            
            # Convert to milliseconds
            since = int(start_date.timestamp() * 1000)
            end_ms = int(end_date.timestamp() * 1000)
            
            all_data = []
            current_since = since
            
            # Fetch data in chunks to handle API limits
            while current_since < end_ms:
                try:
                    # Calculate limit based on timeframe
                    limit = self._calculate_limit(timeframe)
                    
                    # Fetch data chunk
                    chunk_data = self.client.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=current_since,
                        limit=limit
                    )
                    
                    if chunk_data.empty:
                        logger.warning(f"No data returned for chunk starting at {current_since}")
                        break
                    
                    all_data.append(chunk_data)
                    
                    # Update current_since to the last timestamp + 1 interval
                    last_timestamp = chunk_data.index[-1]
                    interval_ms = self._timeframe_to_ms(timeframe)
                    current_since = int(last_timestamp.timestamp() * 1000) + interval_ms
                    
                    # Respect rate limits
                    time.sleep(0.1)
                    
                    logger.info(f"Fetched {len(chunk_data)} records, last timestamp: {last_timestamp}")
                    
                except Exception as e:
                    logger.error(f"Error fetching chunk: {e}")
                    # Wait before retrying
                    time.sleep(settings.retry_delay_seconds)
                    continue
            
            if not all_data:
                logger.warning(f"No data fetched for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Combine all chunks
            combined_data = pd.concat(all_data)
            
            # Remove duplicates and sort
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data.sort_index(inplace=True)
            
            # Filter by date range
            combined_data = combined_data[
                (combined_data.index >= start_date) & 
                (combined_data.index <= end_date)
            ]
            
            logger.info(f"Successfully fetched {len(combined_data)} total records for {symbol} {timeframe}")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol} {timeframe}: {e}")
            raise
    
    def get_last_timestamp(self, symbol: str, timeframe: str, instrument_type: str = "spot") -> Optional[datetime]:
        """
        Get the last timestamp of stored data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            instrument_type: Type of instrument (spot/swap)
            
        Returns:
            Last timestamp or None if no data exists
        """
        try:
            data_path = settings.get_data_path("okx", instrument_type, symbol, timeframe)
            
            # Look for the most recent year file
            current_year = datetime.now().year
            for year in range(current_year, current_year - 5, -1):  # Check last 5 years
                file_path = data_path / f"{year}.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    if not df.empty:
                        # Ensure datetime index
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index)
                        return df.index.max()
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting last timestamp for {symbol} {timeframe}: {e}")
            return None
    
    def update_data(self, symbol: str, timeframe: str, instrument_type: str = "spot") -> bool:
        """
        Update data for a symbol (incremental update).
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            instrument_type: Type of instrument (spot/swap)
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            logger.info(f"Updating data for {symbol} {timeframe}")
            
            # Get last timestamp
            last_timestamp = self.get_last_timestamp(symbol, timeframe, instrument_type)
            
            if last_timestamp:
                # Start from last timestamp + 1 interval
                interval_ms = self._timeframe_to_ms(timeframe)
                start_date = last_timestamp + timedelta(milliseconds=interval_ms)
                logger.info(f"Incremental update from {start_date}")
            else:
                # Full historical download
                start_date = datetime.now(timezone.utc) - timedelta(days=365)  # 1 year default
                logger.info(f"Full historical download from {start_date}")
            
            # Fetch new data
            new_data = self.fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                instrument_type=instrument_type
            )
            
            if new_data.empty:
                logger.info(f"No new data to update for {symbol} {timeframe}")
                return True
            
            # Save new data
            self.save_data(new_data, symbol, timeframe, instrument_type)
            
            logger.info(f"Successfully updated {len(new_data)} records for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating data for {symbol} {timeframe}: {e}")
            return False
    
    def save_data(self, data: pd.DataFrame, symbol: str, timeframe: str, instrument_type: str = "spot"):
        """
        Save data to Parquet files organized by year.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            instrument_type: Type of instrument (spot/swap)
        """
        try:
            if data.empty:
                logger.warning("No data to save")
                return
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Group data by year
            data_by_year = data.groupby(data.index.year)
            
            data_path = settings.get_data_path("okx", instrument_type, symbol, timeframe)
            data_path.mkdir(parents=True, exist_ok=True)
            
            for year, year_data in data_by_year:
                file_path = data_path / f"{year}.parquet"
                
                if file_path.exists():
                    # Load existing data and combine
                    existing_data = pd.read_parquet(file_path)
                    
                    # Ensure datetime index for existing data
                    if not isinstance(existing_data.index, pd.DatetimeIndex):
                        existing_data.index = pd.to_datetime(existing_data.index)
                    
                    # Combine and remove duplicates
                    combined_data = pd.concat([existing_data, year_data])
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    combined_data.sort_index(inplace=True)
                    
                    # Save combined data
                    combined_data.to_parquet(file_path)
                    logger.info(f"Updated {file_path} with {len(year_data)} new records")
                else:
                    # Save new file
                    year_data.to_parquet(file_path)
                    logger.info(f"Created {file_path} with {len(year_data)} records")
            
        except Exception as e:
            logger.error(f"Error saving data for {symbol} {timeframe}: {e}")
            raise
    
    def _calculate_limit(self, timeframe: str) -> int:
        """Calculate appropriate limit based on timeframe."""
        # OKX typically allows up to 100-300 candles per request
        timeframe_limits = {
            '1m': 100,
            '5m': 100,
            '15m': 100,
            '30m': 100,
            '1h': 100,
            '4h': 100,
            '1d': 100,
            '1w': 100,
        }
        return timeframe_limits.get(timeframe, 100)
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe to milliseconds."""
        timeframe_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
        }
        return timeframe_ms.get(timeframe, 60 * 1000)  # Default to 1m