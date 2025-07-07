#!/usr/bin/env python3
"""
OKX API客户端
"""

import sys
import os
import hmac
import hashlib
import base64
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import ccxt
import requests
import asyncio
import aiohttp
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import settings
from logger import data_collector_logger as logger


class OKXClient:
    """OKX API client for fetching market data."""
    
    def __init__(self):
        """Initialize the OKX client."""
        self.exchange = None
        self.base_url = "https://www.okx.com"
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize the ccxt exchange instance."""
        try:
            self.exchange = ccxt.okx({
                **settings.okx_credentials,
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                }
            })
            logger.info("OKX client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OKX client: {e}")
            raise
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from OKX.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT', 'BTC-USDT-SWAP')
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '4h', '1d')
            since: Start time in milliseconds
            limit: Maximum number of records to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if not self.exchange:
                self._initialize_exchange()
            
            logger.info(f"Fetching OHLCV data for {symbol} {timeframe}")
            
            # Fetch data from OKX
            ohlcv_data = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            
            if not ohlcv_data:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            # Drop the original timestamp column
            df.drop('timestamp', axis=1, inplace=True)
            
            # Ensure numeric types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            logger.info(f"Successfully fetched {len(df)} records for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol} {timeframe}: {e}")
            raise
    
    def fetch_funding_rate(self, symbol: str, limit: Optional[int] = None, since: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch funding rate data from OKX.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT-SWAP')
            limit: Maximum number of records to fetch
            since: Start time in milliseconds
            
        Returns:
            DataFrame with funding rate data
        """
        try:
            if not self.exchange:
                self._initialize_exchange()
            
            logger.info(f"Fetching funding rate data for {symbol}")
            
            # Check if symbol is a perpetual swap
            if 'SWAP' not in symbol.upper():
                logger.warning(f"Symbol {symbol} is not a perpetual swap, no funding rate available")
                return pd.DataFrame()
            
            # -------------------- 新分页实现 --------------------
            limit = limit or 100  # OKX 最大 100
            inst_id = symbol.replace('/', '-')
            url = f"{self.base_url}/api/v5/public/funding-rate-history"

            all_rows: List[dict] = []
            before_ts: Optional[int] = None  # 毫秒

            while True:
                params = {
                    'instId': inst_id,
                    'limit': str(limit)
                }
                if before_ts is not None:
                    params['before'] = str(before_ts)
                try:
                    resp = requests.get(url, params=params, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get('code') != '0':
                        logger.warning(f"OKX API error: {data.get('msg')}")
                        break
                    batch = data.get('data', [])
                except Exception as e:
                    logger.warning(f"HTTP error paging funding rates: {e}")
                    break

                if not batch:
                    break

                all_rows.extend(batch)

                # 提取最早 fundingTime
                earliest_ts = int(batch[-1]['fundingTime'])

                # 如果返回不足 limit 或已达到 since 边界，则停止
                if len(batch) < limit or (since is not None and earliest_ts <= since):
                    break

                before_ts = earliest_ts - 1  # 下一轮请求更早的数据
                time.sleep(0.2)  # 避免频率限制
            # ---------------------------------------------------

            if not all_rows:
                logger.warning(f"No funding rate data returned for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame and去重排序
            df = pd.DataFrame(all_rows)
            # 兼容 REST 字段 fundingTime -> timestamp
            if 'fundingTime' in df.columns:
                df['datetime'] = pd.to_datetime(df['fundingTime'].astype(int), unit='ms', utc=True)
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
            else:
                raise ValueError('No time column in funding data')
            df.set_index('datetime', inplace=True)
            df = df.sort_index()
            # 去重
            df = df[~df.index.duplicated(keep='first')]
            # 类型转换
            if 'fundingRate' in df.columns:
                df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')

            logger.info(f"Successfully fetched {len(df)} funding rate records for {symbol}. Range: {df.index.min()} → {df.index.max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching funding rate data for {symbol}: {e}")
            raise
    
    def fetch_open_interest(self, symbol: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch open interest data from OKX.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT-SWAP')
            limit: Maximum number of records to fetch
            
        Returns:
            DataFrame with open interest data
        """
        try:
            if not self.exchange:
                self._initialize_exchange()
            
            logger.info(f"Fetching open interest data for {symbol}")
            
            # Check if symbol is a derivatives contract
            if not any(x in symbol.upper() for x in ['SWAP', 'FUTURES']):
                logger.warning(f"Symbol {symbol} is not a derivatives contract, no open interest available")
                return pd.DataFrame()
            
            # Use OKX REST API directly for open interest
            try:
                # Convert symbol format for OKX API
                inst_id = symbol.replace('/', '-')
                
                url = f"{self.base_url}/api/v5/public/open-interest"
                params = {
                    'instType': 'SWAP' if 'SWAP' in symbol.upper() else 'FUTURES',
                    'instId': inst_id
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('code') != '0':
                    logger.error(f"OKX API error: {data.get('msg')}")
                    return pd.DataFrame()
                
                oi_data = data.get('data', [])
                
                if not oi_data:
                    logger.warning(f"No open interest data returned for {symbol}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(oi_data)
                
                # Convert timestamp
                if 'ts' in df.columns:
                    df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                    df.set_index('datetime', inplace=True)
                    df.drop('ts', axis=1, inplace=True)
                
                # Ensure numeric types
                numeric_columns = ['oi', 'oiCcy']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                logger.info(f"Successfully fetched {len(df)} open interest records for {symbol}")
                return df
                
            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP request error: {e}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching open interest data for {symbol}: {e}")
            raise
    
    def fetch_order_book(self, symbol: str, limit: Optional[int] = None) -> Dict:
        """
        Fetch order book snapshot from OKX.
        
        Args:
            symbol: Trading pair symbol
            limit: Depth of order book (default 20, max 400)
            
        Returns:
            Dictionary with order book data
        """
        try:
            if not self.exchange:
                self._initialize_exchange()
            
            logger.info(f"Fetching order book for {symbol}")
            
            # Set default limit
            if limit is None:
                limit = 20
            
            order_book = self.exchange.fetch_order_book(symbol, limit)
            
            # Add timestamp and symbol info
            order_book['timestamp'] = datetime.now(timezone.utc).isoformat()
            order_book['symbol'] = symbol
            order_book['limit'] = limit
            
            # Calculate spread
            if order_book.get('bids') and order_book.get('asks'):
                best_bid = order_book['bids'][0][0] if order_book['bids'] else 0
                best_ask = order_book['asks'][0][0] if order_book['asks'] else 0
                if best_bid and best_ask:
                    order_book['spread'] = best_ask - best_bid
                    order_book['spread_pct'] = (best_ask - best_bid) / best_bid * 100
            
            logger.info(f"Successfully fetched order book for {symbol}")
            return order_book
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            raise
    
    def fetch_trades(self, symbol: str, limit: Optional[int] = None, since: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch recent trades (tick data) from OKX.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of trades to fetch (max 500)
            since: Fetch trades after this timestamp (milliseconds)
            
        Returns:
            DataFrame with trade data
        """
        try:
            if not self.exchange:
                self._initialize_exchange()
            
            logger.info(f"Fetching trades for {symbol}")
            
            # Set default limit
            if limit is None:
                limit = 100
            
            # Fetch trades
            trades = self.exchange.fetch_trades(symbol, since=since, limit=limit)
            
            if not trades:
                logger.warning(f"No trades returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Convert timestamp
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('datetime', inplace=True)
                df.drop('timestamp', axis=1, inplace=True)
            
            # Ensure numeric types
            numeric_columns = ['amount', 'price', 'cost']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add symbol info
            df['symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(df)} trades for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            raise
    
    def fetch_instrument_info(self, symbol: str) -> Dict:
        """
        Fetch instrument information including listing time.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with instrument information
        """
        try:
            if not self.exchange:
                self._initialize_exchange()
            
            logger.info(f"Fetching instrument info for {symbol}")
            
            # Get markets
            markets = self.get_markets()
            
            if symbol not in markets:
                logger.warning(f"Symbol {symbol} not found in markets")
                return {}
            
            market_info = markets[symbol]
            
            # Extract key information
            info = {
                'symbol': symbol,
                'id': market_info.get('id'),
                'base': market_info.get('base'),
                'quote': market_info.get('quote'),
                'type': market_info.get('type'),  # spot, swap, future, option
                'spot': market_info.get('spot'),
                'margin': market_info.get('margin'),
                'future': market_info.get('future'),
                'option': market_info.get('option'),
                'active': market_info.get('active'),
                'contract': market_info.get('contract'),
                'linear': market_info.get('linear'),
                'inverse': market_info.get('inverse'),
                'expiry': market_info.get('expiry'),
                'precision': market_info.get('precision'),
                'limits': market_info.get('limits'),
                'info': market_info.get('info', {})
            }
            
            # Extract listing time if available
            raw_info = market_info.get('info', {})
            if 'listTime' in raw_info:
                try:
                    list_time_ms = int(raw_info['listTime'])
                    info['listing_time'] = datetime.fromtimestamp(list_time_ms / 1000, tz=timezone.utc)
                    info['listing_timestamp'] = list_time_ms
                except (ValueError, TypeError):
                    logger.warning(f"Invalid listing time format for {symbol}")
            
            # Extract auction end time if available
            if 'auctionEndTime' in raw_info and raw_info['auctionEndTime']:
                try:
                    auction_end_ms = int(raw_info['auctionEndTime'])
                    info['auction_end_time'] = datetime.fromtimestamp(auction_end_ms / 1000, tz=timezone.utc)
                    info['auction_end_timestamp'] = auction_end_ms
                except (ValueError, TypeError):
                    logger.warning(f"Invalid auction end time format for {symbol}")
            
            logger.info(f"Successfully fetched instrument info for {symbol}")
            return info
            
        except Exception as e:
            logger.error(f"Error fetching instrument info for {symbol}: {e}")
            raise
    
    def get_derivatives_symbols(self, inst_type: str = 'SWAP') -> List[str]:
        """
        Get all derivatives symbols of specified type.
        
        Args:
            inst_type: Instrument type ('SWAP', 'FUTURES', 'OPTION')
            
        Returns:
            List of symbols
        """
        try:
            markets = self.get_markets()
            symbols = []
            
            for symbol, market_info in markets.items():
                if inst_type.lower() == 'swap' and market_info.get('swap'):
                    symbols.append(symbol)
                elif inst_type.lower() == 'future' and market_info.get('future'):
                    symbols.append(symbol)
                elif inst_type.lower() == 'option' and market_info.get('option'):
                    symbols.append(symbol)
            
            logger.info(f"Found {len(symbols)} {inst_type} symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting {inst_type} symbols: {e}")
            return []
    
    def get_top_volume_symbols(self, limit: int = 100, quote_currency: str = 'USDT') -> List[Dict]:
        """
        Get top symbols by trading volume.
        
        Args:
            limit: Number of symbols to return
            quote_currency: Quote currency filter
            
        Returns:
            List of symbol dictionaries with volume info
        """
        try:
            logger.info(f"Fetching top {limit} symbols by volume")
            
            # Get all tickers
            tickers = self.fetch_tickers()
            
            # Filter and sort by volume
            volume_data = []
            for symbol, ticker in tickers.items():
                if quote_currency and not symbol.endswith(f'/{quote_currency}'):
                    continue
                
                volume_data.append({
                    'symbol': symbol,
                    'volume': ticker.get('quoteVolume', 0),
                    'baseVolume': ticker.get('baseVolume', 0),
                    'price': ticker.get('last', 0),
                    'change': ticker.get('change', 0),
                    'percentage': ticker.get('percentage', 0)
                })
            
            # Sort by quote volume
            volume_data.sort(key=lambda x: x['volume'], reverse=True)
            
            # Return top symbols
            result = volume_data[:limit]
            logger.info(f"Successfully fetched top {len(result)} symbols by volume")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching top volume symbols: {e}")
            return []

    def get_markets(self) -> Dict:
        """Get available markets from OKX."""
        try:
            if not self.exchange:
                self._initialize_exchange()
            
            markets = self.exchange.load_markets()
            logger.info(f"Successfully loaded {len(markets)} markets")
            return markets
            
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            raise
    
    def get_timeframes(self) -> List[str]:
        """Get available timeframes."""
        if not self.exchange:
            self._initialize_exchange()
        
        return list(self.exchange.timeframes.keys())
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is available on OKX."""
        try:
            markets = self.get_markets()
            return symbol in markets
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if timeframe is supported."""
        try:
            timeframes = self.get_timeframes()
            return timeframe in timeframes
        except Exception as e:
            logger.error(f"Error validating timeframe {timeframe}: {e}")
            return False
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch 24hr ticker statistics for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary with ticker data including volume, price change, etc.
        """
        try:
            if not self.exchange:
                self._initialize_exchange()
            
            logger.info(f"Fetching ticker data for {symbol}")
            
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Add timestamp
            ticker['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Successfully fetched ticker for {symbol}")
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise
    
    def fetch_tickers(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Fetch 24hr ticker statistics for multiple symbols.
        
        Args:
            symbols: List of trading pair symbols. If None, fetch all symbols.
            
        Returns:
            Dictionary with ticker data for multiple symbols
        """
        try:
            if not self.exchange:
                self._initialize_exchange()
            
            logger.info(f"Fetching tickers for {len(symbols) if symbols else 'all'} symbols")
            
            tickers = self.exchange.fetch_tickers(symbols)
            
            # Add timestamp
            timestamp = datetime.now(timezone.utc).isoformat()
            for symbol, ticker in tickers.items():
                ticker['timestamp'] = timestamp
            
            logger.info(f"Successfully fetched {len(tickers)} tickers")
            return tickers
            
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            raise