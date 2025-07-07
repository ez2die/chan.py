#!/usr/bin/env python3
"""
衍生品数据收集器
"""

import sys
import os
import asyncio
import aiohttp
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


class DerivativesCollector:
    """Collector for derivatives market data from OKX."""
    
    def __init__(self):
        """Initialize the derivatives collector."""
        self.client = OKXClient()
        self.data_dir = Path(settings.data_dir)
        self.derivatives_dir = self.data_dir / "okx" / "derivatives"
        
        # Create directories
        self.derivatives_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Derivatives collector initialized")
    
    def collect_funding_rates(
        self, 
        symbols: Optional[List[str]] = None,
        days: int = 30,
        save_data: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect funding rate data for swap symbols.
        
        Args:
            symbols: List of swap symbols to collect. If None, collect top swap symbols.
            days: Number of days of history to collect
            save_data: Whether to save data to files
            
        Returns:
            Dictionary mapping symbol to funding rate DataFrame
        """
        try:
            logger.info(f"Collecting funding rates for {len(symbols) if symbols else 'auto'} symbols")
            
            # Get swap symbols if not provided
            if symbols is None:
                symbols = self.client.get_derivatives_symbols('SWAP')[:20]  # Top 20 swaps
                logger.info(f"Auto-selected {len(symbols)} swap symbols")
            
            # Calculate since timestamp
            since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
            
            funding_data = {}
            
            for symbol in symbols:
                try:
                    logger.info(f"Collecting funding rates for {symbol}")
                    
                    # Fetch funding rate data
                    df = self.client.fetch_funding_rate(symbol, limit=None, since=since)
                    
                    if df.empty:
                        logger.warning(f"No funding rate data for {symbol}")
                        continue
                    
                    funding_data[symbol] = df
                    
                    # Save data if requested
                    if save_data:
                        self._save_funding_rate_data(symbol, df)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error collecting funding rates for {symbol}: {e}")
                    continue
            
            logger.info(f"Successfully collected funding rates for {len(funding_data)} symbols")
            return funding_data
            
        except Exception as e:
            logger.error(f"Error collecting funding rates: {e}")
            return {}
    
    def collect_open_interest(
        self,
        symbols: Optional[List[str]] = None,
        save_data: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect open interest data for derivatives symbols.
        
        Args:
            symbols: List of derivatives symbols to collect
            save_data: Whether to save data to files
            
        Returns:
            Dictionary mapping symbol to open interest DataFrame
        """
        try:
            logger.info(f"Collecting open interest for {len(symbols) if symbols else 'auto'} symbols")
            
            # Get derivatives symbols if not provided
            if symbols is None:
                swap_symbols = self.client.get_derivatives_symbols('SWAP')[:15]
                future_symbols = self.client.get_derivatives_symbols('FUTURES')[:5]
                symbols = swap_symbols + future_symbols
                logger.info(f"Auto-selected {len(symbols)} derivatives symbols")
            
            oi_data = {}
            
            for symbol in symbols:
                try:
                    logger.info(f"Collecting open interest for {symbol}")
                    
                    # Fetch open interest data
                    df = self.client.fetch_open_interest(symbol)
                    
                    if df.empty:
                        logger.warning(f"No open interest data for {symbol}")
                        continue
                    
                    oi_data[symbol] = df
                    
                    # Save data if requested
                    if save_data:
                        self._save_open_interest_data(symbol, df)
                    
                    # Rate limiting
                    time.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"Error collecting open interest for {symbol}: {e}")
                    continue
            
            logger.info(f"Successfully collected open interest for {len(oi_data)} symbols")
            return oi_data
            
        except Exception as e:
            logger.error(f"Error collecting open interest: {e}")
            return {}
    
    def collect_tick_data(
        self,
        symbols: List[str],
        hours: int = 1,
        save_data: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect tick data (recent trades) for symbols.
        
        Args:
            symbols: List of symbols to collect
            hours: Number of hours of recent data to collect
            save_data: Whether to save data to files
            
        Returns:
            Dictionary mapping symbol to trades DataFrame
        """
        try:
            logger.info(f"Collecting tick data for {len(symbols)} symbols")
            
            # Calculate since timestamp
            since = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() * 1000)
            
            tick_data = {}
            
            for symbol in symbols:
                try:
                    logger.info(f"Collecting tick data for {symbol}")
                    
                    # Fetch trades data
                    df = self.client.fetch_trades(symbol, limit=500, since=since)
                    
                    if df.empty:
                        logger.warning(f"No tick data for {symbol}")
                        continue
                    
                    tick_data[symbol] = df
                    
                    # Save data if requested
                    if save_data:
                        self._save_tick_data(symbol, df)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error collecting tick data for {symbol}: {e}")
                    continue
            
            logger.info(f"Successfully collected tick data for {len(tick_data)} symbols")
            return tick_data
            
        except Exception as e:
            logger.error(f"Error collecting tick data: {e}")
            return {}
    
    def collect_instrument_info(
        self,
        symbols: Optional[List[str]] = None,
        save_data: bool = True
    ) -> Dict[str, Dict]:
        """
        Collect instrument information including listing times.
        
        Args:
            symbols: List of symbols to collect info for
            save_data: Whether to save data to files
            
        Returns:
            Dictionary mapping symbol to instrument info
        """
        try:
            logger.info(f"Collecting instrument info for {len(symbols) if symbols else 'all'} symbols")
            
            # Get top volume symbols if not provided
            if symbols is None:
                volume_data = self.client.get_top_volume_symbols(limit=50)
                symbols = [item['symbol'] for item in volume_data]
                logger.info(f"Auto-selected {len(symbols)} symbols by volume")
            
            instrument_data = {}
            
            for symbol in symbols:
                try:
                    logger.info(f"Collecting instrument info for {symbol}")
                    
                    # Fetch instrument info
                    info = self.client.fetch_instrument_info(symbol)
                    
                    if not info:
                        logger.warning(f"No instrument info for {symbol}")
                        continue
                    
                    instrument_data[symbol] = info
                    
                    # Rate limiting
                    time.sleep(0.05)
                    
                except Exception as e:
                    logger.error(f"Error collecting instrument info for {symbol}: {e}")
                    continue
            
            # Save data if requested
            if save_data and instrument_data:
                self._save_instrument_info(instrument_data)
            
            logger.info(f"Successfully collected instrument info for {len(instrument_data)} symbols")
            return instrument_data
            
        except Exception as e:
            logger.error(f"Error collecting instrument info: {e}")
            return {}
    
    def collect_order_books(
        self,
        symbols: List[str],
        depth: int = 20,
        save_data: bool = True
    ) -> Dict[str, Dict]:
        """
        Collect order book snapshots for symbols.
        
        Args:
            symbols: List of symbols to collect
            depth: Order book depth
            save_data: Whether to save data to files
            
        Returns:
            Dictionary mapping symbol to order book data
        """
        try:
            logger.info(f"Collecting order books for {len(symbols)} symbols")
            
            order_books = {}
            
            for symbol in symbols:
                try:
                    logger.info(f"Collecting order book for {symbol}")
                    
                    # Fetch order book
                    order_book = self.client.fetch_order_book(symbol, limit=depth)
                    
                    if not order_book:
                        logger.warning(f"No order book data for {symbol}")
                        continue
                    
                    order_books[symbol] = order_book
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error collecting order book for {symbol}: {e}")
                    continue
            
            # Save data if requested
            if save_data and order_books:
                self._save_order_books(order_books)
            
            logger.info(f"Successfully collected order books for {len(order_books)} symbols")
            return order_books
            
        except Exception as e:
            logger.error(f"Error collecting order books: {e}")
            return {}
    
    def _save_funding_rate_data(self, symbol: str, df: pd.DataFrame):
        """Save funding rate data to parquet file."""
        try:
            # Create symbol directory
            symbol_parts = symbol.replace('/', '-').split('-')
            if len(symbol_parts) >= 2:
                base, quote = symbol_parts[0], symbol_parts[1]
                symbol_dir = self.derivatives_dir / "funding_rates" / base / quote
            else:
                symbol_dir = self.derivatives_dir / "funding_rates" / symbol.replace('/', '-')
            
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to parquet
            file_path = symbol_dir / f"funding_rates_{datetime.now().strftime('%Y%m%d')}.parquet"
            df.to_parquet(file_path)
            
            logger.info(f"Saved funding rate data for {symbol} to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving funding rate data for {symbol}: {e}")
    
    def _save_open_interest_data(self, symbol: str, df: pd.DataFrame):
        """Save open interest data to parquet file."""
        try:
            # Create symbol directory
            symbol_parts = symbol.replace('/', '-').split('-')
            if len(symbol_parts) >= 2:
                base, quote = symbol_parts[0], symbol_parts[1]
                symbol_dir = self.derivatives_dir / "open_interest" / base / quote
            else:
                symbol_dir = self.derivatives_dir / "open_interest" / symbol.replace('/', '-')
            
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to parquet
            file_path = symbol_dir / f"open_interest_{datetime.now().strftime('%Y%m%d')}.parquet"
            df.to_parquet(file_path)
            
            logger.info(f"Saved open interest data for {symbol} to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving open interest data for {symbol}: {e}")
    
    def _save_tick_data(self, symbol: str, df: pd.DataFrame):
        """Save tick data to parquet file."""
        try:
            # Create symbol directory
            symbol_parts = symbol.replace('/', '-').split('-')
            if len(symbol_parts) >= 2:
                base, quote = symbol_parts[0], symbol_parts[1]
                symbol_dir = self.derivatives_dir / "tick_data" / base / quote
            else:
                symbol_dir = self.derivatives_dir / "tick_data" / symbol.replace('/', '-')
            
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to parquet with timestamp
            file_path = symbol_dir / f"tick_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            df.to_parquet(file_path)
            
            logger.info(f"Saved tick data for {symbol} to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving tick data for {symbol}: {e}")
    
    def _save_instrument_info(self, instrument_data: Dict[str, Dict]):
        """Save instrument info to JSON file."""
        try:
            import json
            
            # Convert datetime objects to strings for JSON serialization
            json_data = {}
            for symbol, info in instrument_data.items():
                json_info = {}
                for key, value in info.items():
                    if isinstance(value, datetime):
                        json_info[key] = value.isoformat()
                    else:
                        json_info[key] = value
                json_data[symbol] = json_info
            
            # Save to JSON file
            file_path = self.derivatives_dir / f"instrument_info_{datetime.now().strftime('%Y%m%d')}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved instrument info for {len(instrument_data)} symbols to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving instrument info: {e}")
    
    def _save_order_books(self, order_books: Dict[str, Dict]):
        """Save order book snapshots to JSON file."""
        try:
            import json
            
            # Save to JSON file
            file_path = self.derivatives_dir / f"order_books_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(order_books, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved order books for {len(order_books)} symbols to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving order books: {e}")
    
    def get_summary(self) -> Dict:
        """Get summary of collected derivatives data."""
        try:
            summary = {
                'funding_rates': 0,
                'open_interest': 0,
                'tick_data': 0,
                'order_books': 0,
                'instrument_info': 0,
                'total_files': 0,
                'total_size_mb': 0
            }
            
            if not self.derivatives_dir.exists():
                return summary
            
            # Count files and calculate sizes
            for data_type in ['funding_rates', 'open_interest', 'tick_data']:
                type_dir = self.derivatives_dir / data_type
                if type_dir.exists():
                    files = list(type_dir.rglob('*.parquet'))
                    summary[data_type] = len(files)
                    summary['total_files'] += len(files)
                    
                    for file_path in files:
                        summary['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
            
            # Count JSON files
            json_files = list(self.derivatives_dir.glob('*.json'))
            for json_file in json_files:
                if 'instrument_info' in json_file.name:
                    summary['instrument_info'] += 1
                elif 'order_books' in json_file.name:
                    summary['order_books'] += 1
                
                summary['total_files'] += 1
                summary['total_size_mb'] += json_file.stat().st_size / (1024 * 1024)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting derivatives data summary: {e}")
            return {} 