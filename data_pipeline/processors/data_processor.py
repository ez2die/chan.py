import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    数据处理器：负责加载原始数据、生成不同时间框架数据、计算技术指标
    """
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.spot_path = self.data_root / "okx" / "spot"
        self.swap_path = self.data_root / "okx" / "swap"
        self.derivatives_path = self.data_root / "derivatives_history"
        
    def load_spot_data(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """加载现货数据"""
        base, quote = symbol.split('/')
        data_path = self.spot_path / base / quote / timeframe
        
        dfs = []
        for year in ['2024', '2025']:
            try:
                file_path = data_path / f"{year}.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                    logger.info(f"Loaded {len(df)} records from {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load {year} data: {e}")
                
        if dfs:
            df = pd.concat(dfs).sort_index()
            logger.info(f"Total {len(df)} records loaded for {symbol} {timeframe}")
            return df
        else:
            logger.error(f"No data found for {symbol} {timeframe}")
            return pd.DataFrame()
    
    def resample_to_daily(self, df_hourly: pd.DataFrame) -> pd.DataFrame:
        """将小时数据重采样为日线数据"""
        if df_hourly.empty:
            return pd.DataFrame()
            
        df_daily = df_hourly.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        logger.info(f"Resampled to daily: {len(df_daily)} records")
        return df_daily
    
    def calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """简单移动平均线"""
        return series.rolling(window=period).mean()
    
    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """指数移动平均线"""
        return series.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """相对强弱指标 RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """平均趋向指数 ADX"""
        # 计算真实范围 TR
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # 计算方向性运动 +DM 和 -DM
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=low.index)
        
        # 计算平滑的TR、+DM和-DM
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # 计算DX和ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD指标"""
        ema_fast = self.calculate_ema(series, fast)
        ema_slow = self.calculate_ema(series, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有需要的技术指标"""
        df = df.copy()
        
        # 趋势指标
        df['SMA_20'] = self.calculate_sma(df['close'], 20)
        df['SMA_50'] = self.calculate_sma(df['close'], 50)
        df['EMA_20'] = self.calculate_ema(df['close'], 20)
        df['EMA_50'] = self.calculate_ema(df['close'], 50)
        df['EMA_100'] = self.calculate_ema(df['close'], 100)
        
        # 震荡指标
        df['RSI_14'] = self.calculate_rsi(df['close'], 14)
        
        # 市场状态指标
        df['ADX_14'] = self.calculate_adx(df['high'], df['low'], df['close'], 14)
        
        # 额外的有用指标
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(24).std()
        df['volume_ma'] = df['volume'].rolling(24).mean()
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = self.calculate_macd(df['close'])
        
        logger.info(f"Calculated {len(df.columns)} indicators")
        return df
    
    def load_derivatives_data(self) -> Dict[str, pd.DataFrame]:
        """加载衍生品数据（资金费率和持仓量）"""
        derivatives_data = {}
        
        # 加载资金费率 (从新的完整历史文件加载)
        funding_path = self.data_root / "processed" / "BTC_USDT_SWAP_funding_rate_full_history.parquet"
        if funding_path.exists():
            df_funding = pd.read_parquet(funding_path)
            # 新文件索引已经是datetime，无需转换
            derivatives_data['funding_rates'] = df_funding
            logger.info(f"Loaded {len(df_funding)} funding rate records from full history.")
        else:
            logger.warning(f"Full funding rate history file not found at {funding_path}, skipping.")
        
        # 加载持仓量 (逻辑保留)
        oi_path = self.derivatives_path / "BTC_USDT_SWAP_open_interest_3y.csv"
        if oi_path.exists():
            df_oi = pd.read_csv(oi_path)
            df_oi['datetime'] = pd.to_datetime(df_oi['datetime'])
            df_oi = df_oi.set_index('datetime')
            derivatives_data['open_interest'] = df_oi
            logger.info(f"Loaded {len(df_oi)} open interest records")
            
        return derivatives_data
    
    def process_and_save(self, symbol: str = "BTC/USDT", 
                        output_dir: str = "data/processed") -> Dict[str, pd.DataFrame]:
        """
        处理数据并保存到文件
        返回处理后的数据字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_data = {}
        
        # 1. 加载和处理1h数据
        logger.info(f"Processing {symbol} hourly data...")
        df_1h = self.load_spot_data(symbol, '1h')
        if not df_1h.empty:
            df_1h_with_indicators = self.calculate_indicators(df_1h)
            processed_data['1h'] = df_1h_with_indicators
            
            # 保存1h数据
            output_file = output_path / f"{symbol.replace('/', '_')}_1h_processed.parquet"
            df_1h_with_indicators.to_parquet(output_file)
            logger.info(f"Saved hourly data to {output_file}")
        
        # 2. 生成和处理1d数据
        logger.info(f"Generating {symbol} daily data...")
        df_1d = self.resample_to_daily(df_1h)
        if not df_1d.empty:
            df_1d_with_indicators = self.calculate_indicators(df_1d)
            processed_data['1d'] = df_1d_with_indicators
            
            # 保存1d数据
            output_file = output_path / f"{symbol.replace('/', '_')}_1d_processed.parquet"
            df_1d_with_indicators.to_parquet(output_file)
            logger.info(f"Saved daily data to {output_file}")
        
        # 3. 加载衍生品数据
        logger.info("Loading derivatives data...")
        derivatives_data = self.load_derivatives_data()
        processed_data.update(derivatives_data)
        
        # 4. 合并现货和衍生品数据（用于综合分析）
        if '1h' in processed_data and 'funding_rates' in derivatives_data:
            # 处理资金费率数据中的重复索引
            funding_df = derivatives_data['funding_rates'].copy()
            funding_df = funding_df[~funding_df.index.duplicated(keep='first')]
            
            # 将资金费率数据重采样到1h（从8h）
            funding_1h = funding_df['fundingRate'].resample('1h').ffill()

            # 进一步对齐至 OHLCV 索引并双向填补缺口
            funding_aligned = funding_1h.reindex(processed_data['1h'].index).ffill().bfill()

            # 合并数据
            combined_df = processed_data['1h'].copy()
            combined_df = combined_df.join(funding_aligned.rename('fundingRate'))
            
            processed_data['combined'] = combined_df
            
            # 保存合并数据
            output_file = output_path / f"{symbol.replace('/', '_')}_combined_processed.parquet"
            combined_df.to_parquet(output_file)
            logger.info(f"Saved combined data to {output_file}")
        
        logger.info("Data processing completed!")
        return processed_data
    
    def get_market_regime(self, adx_value: float) -> str:
        """
        根据ADX值判断市场状态
        ADX < 20: 震荡市场
        ADX > 25: 趋势市场
        20 <= ADX <= 25: 中性市场
        """
        if adx_value < 20:
            return 'RANGING'
        elif adx_value > 25:
            return 'TREND'
        else:
            return 'NEUTRAL'


if __name__ == "__main__":
    # 测试数据处理器
    processor = DataProcessor()
    
    # 处理BTC/USDT数据
    processed_data = processor.process_and_save("BTC/USDT")
    
    # 显示处理结果
    for key, df in processed_data.items():
        if isinstance(df, pd.DataFrame):
            print(f"\n{key} data shape: {df.shape}")
            if not df.empty:
                print(f"Date range: {df.index.min()} to {df.index.max()}")
                print(f"Columns: {list(df.columns)[:10]}...")  # 显示前10列