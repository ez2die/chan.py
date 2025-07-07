import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录
sys.path.append(str(Path(__file__).parent.parent))

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE, BSP_TYPE, BI_DIR, FX_TYPE
from Common.CTime import CTime
from KLine.KLine_Unit import CKLine_Unit
from tools.data_converter import ChanDataConverter

class EnhancedChanFeatureCalculator:
    """增强版Chan.py特征计算器 - 基于现有数据大幅扩展特征"""
    
    def __init__(self):
        self.feature_config = {
            'basic_features': True,
            'technical_indicators': True,
            'volume_features': True,
            'multi_timeframe_features': True,  # 多时间窗口特征
            'historical_features': True,       # 历史回望特征
            'time_based_features': True,       # 时间特征
            'price_structure_features': True,  # 价格结构特征
            'volatility_features': True,       # 波动率特征
            'momentum_features': True,         # 动量特征
            'chan_features': True,             # 缠论特征
            'advanced_chan_features': True,    # 高级缠论特征
            'interaction_features': True,      # 特征交互
            'funding_rate_features': True,    # 资金费率特征
        }
        
        # 时间窗口配置
        self.time_windows = {
            'short': [3, 5, 8, 12],
            'medium': [20, 30, 50, 100],
            'long': [200, 300, 500]
        }
        
        # 历史回望周期
        self.lookback_periods = [1, 2, 3, 6, 12, 24, 48, 72, 168]  # 小时
        
        # Chan.py配置
        self.chan_config = CChanConfig({
            "trigger_step": True,  # 必须为True以支持迭代计算
            "bi_strict": True,
            "skip_step": 0,
            "divergence_rate": float("inf"),
            "bsp2_follow_1": False,
            "bsp3_follow_1": False,
            "min_zs_cnt": 0,
            "bs1_peak": False,
            "macd_algo": "peak",
            "bs_type": '1,2,3a,1p,2s,3b',
            "print_warning": False,
            "zs_algo": "normal",
            "zs_combine": True,
            "one_bi_zs": True,
        })
        
        self.data_converter = ChanDataConverter()
        
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有增强特征"""
        print(f"开始计算增强特征，原始数据: {len(df)} 行")
        features_df = df.copy()
        
        # 1. 基础特征
        if self.feature_config['basic_features']:
            print("计算基础特征...")
            features_df = self._add_basic_features(features_df)
            
        # 2. 技术指标特征
        if self.feature_config['technical_indicators']:
            print("计算技术指标特征...")
            features_df = self._add_technical_features(features_df)
            
        # 3. 成交量特征
        if self.feature_config['volume_features']:
            print("计算成交量特征...")
            features_df = self._add_volume_features(features_df)
            
        # 4. 多时间窗口特征
        if self.feature_config['multi_timeframe_features']:
            print("计算多时间窗口特征...")
            features_df = self._add_multi_timeframe_features(features_df)
            
        # 5. 历史回望特征
        if self.feature_config['historical_features']:
            print("计算历史回望特征...")
            features_df = self._add_historical_features(features_df)
            
        # 6. 时间特征
        if self.feature_config['time_based_features']:
            print("计算时间特征...")
            features_df = self._add_time_features(features_df)
            
        # 7. 价格结构特征
        if self.feature_config['price_structure_features']:
            print("计算价格结构特征...")
            features_df = self._add_price_structure_features(features_df)
            
        # 8. 波动率特征
        if self.feature_config['volatility_features']:
            print("计算波动率特征...")
            features_df = self._add_volatility_features(features_df)
            
        # 8.1 资金费率特征（依赖已完成的 rolling 计算基础列）
        if self.feature_config.get('funding_rate_features', False) and 'fundingRate' in features_df.columns:
            print("计算资金费率特征...")
            features_df = self._add_funding_rate_features(features_df)
            
        # 9. 动量特征
        if self.feature_config['momentum_features']:
            print("计算动量特征...")
            features_df = self._add_momentum_features(features_df)
            
        # 10. 缠论特征
        if self.feature_config['chan_features']:
            print("计算缠论特征...")
            features_df = self._add_chan_features(features_df)
            
        # 11. 高级缠论特征
        if self.feature_config['advanced_chan_features']:
            print("计算高级缠论特征...")
            features_df = self._add_advanced_chan_features(features_df)
            
        # 12. 特征交互
        if self.feature_config['interaction_features']:
            print("计算特征交互...")
            features_df = self._add_interaction_features(features_df)
            
        # 清理无效值
        features_df = self._clean_features(features_df)
        
        print(f"特征计算完成，最终特征数: {len(features_df.columns)}")
        return features_df
        
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基础价格特征 - 扩展版"""
        # 基础收益率
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['abs_returns'] = abs(df['returns'])
        
        # 价格位置特征
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # K线形态特征
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']
        df['total_shadow'] = df['upper_shadow'] + df['lower_shadow']
        df['shadow_ratio'] = df['total_shadow'] / df['body_size']
        
        # 价格范围特征
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = df['price_range'] / df['close']
        
        # 价格变化特征
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = df['price_change'] / df['open']
        df['high_change'] = df['high'] - df['close'].shift(1)
        df['low_change'] = df['low'] - df['close'].shift(1)
        df['gap_up'] = np.maximum(0, df['low'] - df['high'].shift(1))
        df['gap_down'] = np.maximum(0, df['low'].shift(1) - df['high'])
        
        # 价格强度特征
        df['bullish_strength'] = np.where(df['close'] > df['open'], 
                                         (df['close'] - df['open']) / df['price_range'], 0)
        df['bearish_strength'] = np.where(df['close'] < df['open'], 
                                         (df['open'] - df['close']) / df['price_range'], 0)
        
        return df
        
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """技术指标特征 - 扩展版"""
        # 移动平均线 - 多周期
        ma_periods = [5, 10, 15, 20, 30, 50, 100, 200]
        for period in ma_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff(5) / df[f'sma_{period}'].shift(5)
            
        # 指数移动平均
        ema_periods = [12, 26, 50, 100]
        for period in ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
            
        # MACD系列
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_histogram_slope'] = df['macd_histogram'].diff()
        df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        df['macd_cross_change'] = df['macd_cross'].diff()
        
        # RSI系列
        rsi_periods = [14, 21, 30]
        for period in rsi_periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
            df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
            
        # 布林带系列
        bb_periods = [20, 50]
        for period in bb_periods:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_{period}_upper'] = sma + (std * 2)
            df[f'bb_{period}_lower'] = sma - (std * 2)
            df[f'bb_{period}_position'] = (df['close'] - df[f'bb_{period}_lower']) / (df[f'bb_{period}_upper'] - df[f'bb_{period}_lower'])
            df[f'bb_{period}_width'] = (df[f'bb_{period}_upper'] - df[f'bb_{period}_lower']) / sma
            df[f'bb_{period}_squeeze'] = (df[f'bb_{period}_width'] < df[f'bb_{period}_width'].rolling(20).mean()).astype(int)
            
        # KDJ指标
        kdj_periods = [9, 14]
        for period in kdj_periods:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            rsv = (df['close'] - low_min) / (high_max - low_min) * 100
            df[f'kdj_{period}_k'] = rsv.ewm(com=2).mean()
            df[f'kdj_{period}_d'] = df[f'kdj_{period}_k'].ewm(com=2).mean()
            df[f'kdj_{period}_j'] = 3 * df[f'kdj_{period}_k'] - 2 * df[f'kdj_{period}_d']
            
        # 其他技术指标
        df['williams_r'] = (df['high'].rolling(14).max() - df['close']) / (df['high'].rolling(14).max() - df['low'].rolling(14).min()) * -100
        df['atr'] = df['true_range'].rolling(14).mean()
        df['atr_ratio'] = df['true_range'] / df['atr']
        
        # 动量指标
        momentum_periods = [5, 10, 20]
        for period in momentum_periods:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'roc_{period}'] = df['close'].pct_change(period)
            
        return df
        
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量特征 - 扩展版"""
        if 'volume' not in df.columns:
            return df
            
        # 成交量移动平均
        volume_periods = [5, 10, 20, 50]
        for period in volume_periods:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
            
        # 价量关系
        df['price_volume_trend'] = df['returns'] * df['volume_ratio_20']
        df['volume_price_trend'] = df['volume'] * df['price_change_pct']
        df['volume_weighted_price'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
        
        # 成交量分布
        df['volume_std_20'] = df['volume'].rolling(20).std()
        df['volume_cv'] = df['volume_std_20'] / df['volume_sma_20']
        df['volume_percentile'] = df['volume'].rolling(100).rank(pct=True)
        
        # OBV相关
        df['obv'] = (df['volume'] * np.sign(df['returns'])).cumsum()
        df['obv_sma_20'] = df['obv'].rolling(20).mean()
        df['obv_ratio'] = df['obv'] / df['obv_sma_20']
        
        # 成交量强度
        df['volume_surge'] = (df['volume'] > df['volume_sma_20'] * 2).astype(int)
        df['volume_dry'] = (df['volume'] < df['volume_sma_20'] * 0.5).astype(int)
        
        return df
        
    def _add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """多时间窗口特征"""
        all_windows = self.time_windows['short'] + self.time_windows['medium'] + self.time_windows['long']
        
        # 收集需要新增的列，最后一次性 concat，避免 DataFrame 高度碎片化
        new_cols = {}

        for window in all_windows:
            if window >= len(df):
                continue

            # 价格统计
            high_max = df['high'].rolling(window).max()
            low_min = df['low'].rolling(window).min()
            price_pos = (df['close'] - low_min) / (high_max - low_min)

            new_cols[f'high_max_{window}'] = high_max
            new_cols[f'low_min_{window}'] = low_min
            new_cols[f'price_position_{window}'] = price_pos

            # 波动率
            vol_window = df['returns'].rolling(window).std()
            if 'volatility' in df.columns:
                vol_ratio = vol_window / df['volatility']
            else:
                vol_ratio = vol_window / df['returns'].rolling(24).std()

            new_cols[f'volatility_{window}'] = vol_window
            new_cols[f'volatility_ratio_{window}'] = vol_ratio

            # 趋势强度
            new_cols[f'trend_strength_{window}'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)

            # 价格范围
            price_range_avg = df['price_range'].rolling(window).mean()
            new_cols[f'price_range_avg_{window}'] = price_range_avg
            new_cols[f'price_range_ratio_{window}'] = df['price_range'] / price_range_avg

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df
        
    def _add_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """历史回望特征"""
        for period in self.lookback_periods:
            if period >= len(df):
                continue
                
            # 价格变化
            df[f'price_change_{period}h'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
            df[f'high_change_{period}h'] = (df['high'] - df['high'].shift(period)) / df['high'].shift(period)
            df[f'low_change_{period}h'] = (df['low'] - df['low'].shift(period)) / df['low'].shift(period)
            
            # 成交量变化
            if 'volume' in df.columns:
                df[f'volume_change_{period}h'] = (df['volume'] - df['volume'].shift(period)) / df['volume'].shift(period)
                
            # 波动率变化
            vol_current = df['returns'].rolling(24).std()
            vol_past = df['returns'].shift(period).rolling(24).std()
            df[f'volatility_change_{period}h'] = (vol_current - vol_past) / vol_past
            
            # 技术指标变化
            df[f'rsi_change_{period}h'] = df['rsi_14'] - df['rsi_14'].shift(period)
            df[f'macd_change_{period}h'] = df['macd'] - df['macd'].shift(period)
            
        return df
        
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """时间特征"""
        if 'timestamp' in df.columns:
            # 确保 timestamp 列是 datetime 类型
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['datetime'] = pd.to_datetime(df['timestamp'])
            else:
                df['datetime'] = df['timestamp']
        elif isinstance(df.index, pd.DatetimeIndex):
            df['datetime'] = df.index
        else:
            return df
            
        # 基础时间特征
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)
        df['month'] = df['datetime'].dt.month
        
        # 时间周期特征
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # 交易时段特征（UTC时间）
        df['asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        # 新增：周末/节假日效应
        df['weekend_effect'] = df['is_weekend']  # 二值同 is_weekend
        holidays_mmdd = {"01-01", "12-25", "07-04"}  # 可按需扩展
        df['holiday_effect'] = df['datetime'].dt.strftime('%m-%d').isin(holidays_mmdd).astype(int)
        
        # 时间周期性特征
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
        
    def _add_price_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格结构特征"""
        # 支撑阻力水平
        for window in [50, 100, 200]:
            df[f'resistance_{window}'] = df['high'].rolling(window).max()
            df[f'support_{window}'] = df['low'].rolling(window).min()
            df[f'distance_to_resistance_{window}'] = (df[f'resistance_{window}'] - df['close']) / df['close']
            df[f'distance_to_support_{window}'] = (df['close'] - df[f'support_{window}']) / df['close']
            
        # 价格分位数
        for window in [50, 100]:
            df[f'price_percentile_{window}'] = df['close'].rolling(window).rank(pct=True)
            
        # 突破特征
        df['breakout_resistance_50'] = (df['close'] > df['resistance_50'].shift(1)).astype(int)
        df['breakdown_support_50'] = (df['close'] < df['support_50'].shift(1)).astype(int)
        
        # 价格密度
        for window in [20, 50]:
            price_std = df['close'].rolling(window).std()
            df[f'price_density_{window}'] = 1 / (price_std + 1e-8)
            
        # 新增：支撑/阻力强度 & 突破概率（基于50周期）
        window_sr = 50
        # 支撑/阻力触及次数占比
        support_touch = (df['low'] <= df[f'support_{window_sr}']).rolling(window_sr).sum()
        resistance_touch = (df['high'] >= df[f'resistance_{window_sr}']).rolling(window_sr).sum()
        df['support_strength'] = support_touch / window_sr
        df['resistance_strength'] = resistance_touch / window_sr
        # 近20根突破阻力的比例估算突破概率
        df['breakout_probability'] = df['breakout_resistance_50'].rolling(20).mean()
        
        # 体积分布特征：Volume Profile POC 及 Value Area (50周期)
        if 'volume' in df.columns:
            vol_window = 50
            vwap = (df['close'] * df['volume']).rolling(vol_window).sum() / df['volume'].rolling(vol_window).sum()
            df['volume_profile_poc_50'] = vwap
        # Value area 高/低 (30%~70% 分位)
        df['value_area_high_50'] = df['close'].rolling(50).quantile(0.7)
        df['value_area_low_50'] = df['close'].rolling(50).quantile(0.3)
        
        # 支撑阻力区间宽度 & 中心
        df['support_resistance_levels'] = df[f'resistance_{window_sr}'] - df[f'support_{window_sr}']
        df['support_resistance_mid'] = (df[f'resistance_{window_sr}'] + df[f'support_{window_sr}']) / 2
        
        return df
        
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """波动率特征"""
        # 已实现波动率
        for window in [12, 24, 48, 168]:
            df[f'realized_vol_{window}'] = df['returns'].rolling(window).std() * np.sqrt(24)
            
        # 波动率比率
        df['vol_ratio_short_long'] = df['realized_vol_12'] / df['realized_vol_168']
        df['vol_ratio_medium_long'] = df['realized_vol_48'] / df['realized_vol_168']
        
        # 波动率分位数
        df['vol_percentile_100'] = df['realized_vol_24'].rolling(100).rank(pct=True)
        
        # GARCH风格的波动率
        df['vol_ewm'] = df['returns'].ewm(span=24).std()
        df['vol_expansion'] = df['realized_vol_24'] / df['vol_ewm']
        
        # 波动率聚集性
        df['vol_clustering'] = (df['realized_vol_24'] > df['realized_vol_24'].rolling(50).mean()).astype(int)
        
        # 极端波动
        df['extreme_vol'] = (df['realized_vol_24'] > df['realized_vol_24'].rolling(100).quantile(0.95)).astype(int)
        
        # 新增：GARCH风格波动率 & 波动率状态
        df['garch_volatility'] = df['returns'].ewm(span=10).std() * np.sqrt(24)
        vol_thresh_high = df['realized_vol_24'].rolling(100).quantile(0.75)
        df['volatility_regime'] = (df['realized_vol_24'] > vol_thresh_high).astype(int)
        
        return df
        
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """动量特征 - 优化写入方式以避免 DataFrame 碎片化"""

        momentum_periods = [3, 5, 10, 20, 50]
        new_cols: dict[str, pd.Series] = {}

        # 基础动量及其 10 日均值
        for period in momentum_periods:
            mom = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
            new_cols[f'momentum_{period}'] = mom
            new_cols[f'momentum_{period}_ma'] = mom.rolling(10).mean()

        # 动量加速度
        new_cols['momentum_acceleration_5'] = new_cols['momentum_5'] - new_cols['momentum_5'].shift(5)
        new_cols['momentum_acceleration_20'] = new_cols['momentum_20'] - new_cols['momentum_20'].shift(5)

        # 相对强弱
        new_cols['relative_strength'] = new_cols['momentum_20'] / new_cols['momentum_20'].rolling(50).std()

        # 动量背离
        volume_momentum = df['volume'].pct_change(20) if 'volume' in df.columns else 0
        new_cols['momentum_divergence'] = new_cols['momentum_20'] - volume_momentum

        # --- 新增自适应与强度指标 ---
        cand_periods = [3, 5, 10, 20, 50]
        # 组装绝对动量 DataFrame 以选择最佳周期 —— 使用新生成的列而非 df 中可能尚不存在的列
        mom_abs_df = pd.concat([new_cols[f'momentum_{p}'].abs().rename(str(p)) for p in cand_periods], axis=1)
        df['adaptive_ma_period'] = mom_abs_df.idxmax(axis=1).astype(float)  # 以周期数字表示
        if 'realized_vol_24' in df.columns:
            df['trend_strength_adaptive'] = new_cols['momentum_20'] / (df['realized_vol_24'] + 1e-8)
        
        # 统一追加 / 覆盖已存在列
        for col, series in new_cols.items():
            df[col] = series

        return df

    def _add_chan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [重构] 缠论特征 - 使用迭代计算避免数据泄露
        该方法通过逐根K线模拟，在每个时间点上运行完整的缠论分析，
        确保每个时间点的特征仅基于历史数据。
        """
        # 1. 初始化所有缠论特征列为默认值 (0.0)
        self._init_chan_feature_columns(df)
        
        # 2. 转换整个数据集为Chan格式
        try:
            chan_data = self._convert_to_chan_format(df)
            if len(chan_data) < 20: # 保证最小计算量
                print("警告: 数据量不足，跳过缠论特征计算。")
                return df
        except Exception as e:
            print(f"错误: K线数据转换失败: {e}。跳过缠论特征。")
            return df
            
        # 3. 创建CChan对象
        try:
            chan_obj = CChan(
                code="TEMP",
                begin_time="2000-01-01",
                end_time="2100-01-01",
                data_src=DATA_SRC.CSV,  # 使用轻量数据源避免外部依赖
                lv_list=[KL_TYPE.K_60M],
                config=self.chan_config,
            )
        except Exception as e:
            print(f"错误: CChan对象初始化失败: {e}。跳过缠论特征。")
            return df

        # 3.2 逐条注入 K 线并提取特征 —— 避免 step_load 中的外部数据请求
        print(f"开始迭代计算 {len(df)} 根K线的缠论特征...")
        for i, kl_unit in enumerate(chan_data):
            try:
                # 只注入当前一根 K 线，保证不泄露未来信息
                chan_obj.trigger_load({KL_TYPE.K_60M: [kl_unit]})
            except Exception as exc:
                # 若内部一致性检查报错，则跳过该行，防御性处理
                print(f"⚠️  trigger_load 失败 (idx={i}): {exc}")
                continue

            current_timestamp = df.index[i]
            features_at_step = self._extract_chan_features_at_step(chan_obj)
            for feature, value in features_at_step.items():
                if feature in df.columns:
                    df.loc[current_timestamp, feature] = value

            if (i + 1) % 500 == 0:
                print(f"  ...已处理 {i+1}/{len(df)} K线")

        print("✅ 缠论特征迭代计算完成。")
        return df

    def _extract_chan_features_at_step(self, chan: CChan) -> Dict[str, Any]:
        """在单个时间点提取所有缠论特征"""
        # 获取当前级别的分析结果
        kl_data = chan[KL_TYPE.K_60M]
        if not kl_data or not kl_data.bi_list:
            return {}

        # CKLine 对象没有 close 属性，最后一根 K 线收盘价位于其 lst[-1].close
        try:
            current_price = kl_data[-1].lst[-1].close
        except Exception:
            current_price = 0.0  # 防御性回退
        bi_list = kl_data.bi_list
        seg_list = kl_data.seg_list

        # 合并所有特征提取器的结果
        features = {}
        features.update(self._extract_bi_features(bi_list))
        features.update(self._extract_seg_features(seg_list))
        features.update(self._extract_zs_features(seg_list, current_price))
        features.update(self._extract_bsp_features(chan))
        features.update(self._extract_macd_features(bi_list, seg_list))
        features.update(self._extract_pattern_features(kl_data))
        features.update(self._extract_trend_features(seg_list, bi_list))
        
        return features
        
    def _add_advanced_chan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级缠论特征"""
        # 这里可以添加更复杂的缠论特征
        # 比如多级别联动、分形特征等
        
        # 分形维度代理
        df['fractal_dimension'] = self._calculate_fractal_dimension(df)
        
        # 市场状态识别
        df['market_regime'] = self._identify_market_regime(df)
        
        # --- 新增高级缠论 / 价格结构衍生特征 ---
        # 跨时间框架背离：短长期动量差值
        if {'momentum_20', 'momentum_50'}.issubset(df.columns):
            df['cross_timeframe_divergence'] = df['momentum_20'] - df['momentum_50']
        # 艾略特波浪位置：近10根涨跌和
        price_diff = df['close'].diff()
        wave_seq = pd.Series(np.where(price_diff > 0, 1, -1), index=df.index)
        df['elliott_wave_position'] = wave_seq.rolling(10).sum()
        # 斐波那契水平：当前价在100周期高低区间中的位置
        win_fb = 100
        high_max = df['high'].rolling(win_fb).max()
        low_min = df['low'].rolling(win_fb).min()
        df['fibonacci_levels'] = (df['close'] - low_min) / (high_max - low_min + 1e-8)
        
        return df
        
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征交互"""
        # 价量交互
        if 'volume' in df.columns:
            df['price_volume_interaction'] = df['returns'] * df['volume_ratio_20']
            df['volatility_volume_interaction'] = df['realized_vol_24'] * df['volume_ratio_20']
            
        # 趋势波动率交互
        df['trend_vol_interaction'] = df['momentum_20'] * df['realized_vol_24']
        
        # 技术指标交互
        df['rsi_macd_interaction'] = df['rsi_14'] * df['macd']
        df['bb_rsi_interaction'] = df['bb_20_position'] * df['rsi_14']
        
        # 新增：价量相关性特征
        if 'volume' in df.columns:
            df['cross_corr_returns_volume_20'] = df['returns'].rolling(20).corr(df['volume'])
        
        return df
        
    def _calculate_fractal_dimension(self, df: pd.DataFrame) -> pd.Series:
        """计算分形维度代理"""
        # 简化的分形维度计算
        window = 50
        returns = df['returns'].rolling(window)
        
        def hurst_exponent(ts):
            try:
                lags = range(2, min(20, len(ts)//2))
                tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            except:
                return 0.5
                
        return returns.apply(hurst_exponent)
        
    def _identify_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """识别市场状态"""
        # 基于波动率和趋势的简单状态识别
        vol_threshold = df['realized_vol_24'].rolling(100).quantile(0.7)
        trend_threshold = 0.02
        
        conditions = [
            (df['realized_vol_24'] > vol_threshold) & (df['momentum_20'] > trend_threshold),
            (df['realized_vol_24'] > vol_threshold) & (df['momentum_20'] < -trend_threshold),
            (df['realized_vol_24'] <= vol_threshold) & (abs(df['momentum_20']) < trend_threshold/2),
        ]
        
        choices = [2, 0, 1]  # 2: 高波动上涨, 0: 高波动下跌, 1: 低波动横盘
        
        return np.select(conditions, choices, default=1)
        
    def _convert_to_chan_format(self, df: pd.DataFrame) -> List[CKLine_Unit]:
        """转换为Chan格式 - 优化版本"""
        from Common.CEnum import DATA_FIELD
        from Common.CTime import CTime
        
        chan_data = []
        # 去重：确保时间索引严格递增，删除重复索引行
        if isinstance(df.index, pd.DatetimeIndex):
            df_dedup = df[~df.index.duplicated(keep='first')]
        else:
            df_dedup = df

        for i, row in df_dedup.iterrows():
            try:
                # 改进时间处理
                if isinstance(i, pd.Timestamp):
                    dt = i
                elif 'timestamp' in row and pd.notna(row['timestamp']):
                    dt = pd.to_datetime(row['timestamp'])
                else:
                    # 使用序号生成时间，确保时间递增
                    base_time = pd.Timestamp('2024-01-01 01:00:00')  # 从1点开始，避免00:00
                    dt = base_time + pd.Timedelta(hours=len(chan_data))
                
                # ✅ 设置auto=False避免时间自动调整，确保时间严格递增
                time_obj = CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, auto=False)
                
                kl_dict = {
                    DATA_FIELD.FIELD_TIME: time_obj,
                    DATA_FIELD.FIELD_OPEN: float(row['open']),
                    DATA_FIELD.FIELD_HIGH: float(row['high']),
                    DATA_FIELD.FIELD_LOW: float(row['low']),
                    DATA_FIELD.FIELD_CLOSE: float(row['close']),
                    DATA_FIELD.FIELD_VOLUME: float(row.get('volume', 0)),
                    DATA_FIELD.FIELD_TURNOVER: float(row.get('turnover', 0)),
                    DATA_FIELD.FIELD_TURNRATE: float(row.get('turnrate', 0))
                }
                chan_data.append(CKLine_Unit(kl_dict))
                
            except Exception as e:
                print(f"警告: K线数据转换失败 (行 {i}): {e}")
                continue
                
        return chan_data
        
    def _init_chan_feature_columns(self, df: pd.DataFrame):
        """初始化缠论特征列"""
        chan_features = [
            # 笔特征
            'bi_count', 'bi_up_count', 'bi_down_count', 'bi_up_ratio',
            'current_bi_dir', 'current_bi_amp', 'current_bi_length', 'current_bi_slope',
            'last_bi_amp', 'last_bi_length', 'bi_amp_ratio',
            'avg_bi_amp', 'max_bi_amp', 'min_bi_amp', 'bi_amp_std',
            'bi_trend_strength', 'bi_reversal_signal',
            
            # 线段特征
            'seg_count', 'seg_up_count', 'seg_down_count', 'seg_up_ratio',
            'current_seg_dir', 'current_seg_amp', 'current_seg_length', 'current_seg_slope',
            'last_seg_amp', 'last_seg_length', 'seg_amp_ratio', 'seg_len_ratio',
            'avg_seg_amp', 'max_seg_amp', 'min_seg_amp', 'seg_amp_std',
            'seg_trend_strength', 'current_seg_bi_count',
            
            # 中枢特征
            'zs_count', 'last_zs_height', 'last_zs_width', 'last_zs_volume',
            'zs_avg_height', 'zs_max_height',
            'price_vs_zs_high', 'price_vs_zs_low', 'is_in_zs',
            'zs_level_change', 'is_zs_overlap',
            
            # 买卖点特征
            'bsp_count', 'buy_point_count', 'sell_point_count',
            'klines_since_last_bsp', 'klines_since_last_buy_bsp', 'klines_since_last_sell_bsp',
            'last_bsp_type', 'last_bsp_is_buy', 'bsp_density_100',
            
            # MACD特征
            'bi_macd_divergence', 'seg_macd_divergence', 'macd_area_in_zs',
            'macd_at_bsp', 'bi_crosses_zero_axis', 'macd_peak_count', 'macd_valley_count',
            
            # 形态特征
            'fx_top_count', 'fx_bottom_count', 'is_double_top', 'is_double_bottom',
            'consolidation_strength',
            
            # 趋势特征
            'trend_direction', 'trend_strength', 'trend_consistency',
            'trend_duration',
        ]
        
        for feature in chan_features:
            df[feature] = 0.0
            
    def _extract_bi_features(self, bi_list) -> Dict[str, Any]:
        """提取笔特征"""
        if not bi_list:
            return {}
        
        features = {}
        features['bi_count'] = len(bi_list)
        up_bis = [bi for bi in bi_list if bi.is_up()]
        down_bis = [bi for bi in bi_list if bi.is_down()]
        
        features['bi_up_count'] = len(up_bis)
        features['bi_down_count'] = len(down_bis)
        features['bi_up_ratio'] = len(up_bis) / len(bi_list) if bi_list else 0
        
        # 当前笔特征
        current_bi = bi_list[-1]
        features['current_bi_dir'] = 1 if current_bi.is_up() else -1
        features['current_bi_amp'] = current_bi.amp()
        features['current_bi_length'] = current_bi.get_klu_cnt()
        features['current_bi_slope'] = features['current_bi_amp'] / features['current_bi_length'] if features['current_bi_length'] > 0 else 0

        # 上一笔特征
        if len(bi_list) > 1:
            last_bi = bi_list[-2]
            features['last_bi_amp'] = last_bi.amp()
            features['last_bi_length'] = last_bi.get_klu_cnt()
            features['bi_amp_ratio'] = current_bi.amp() / last_bi.amp() if last_bi.amp() > 0 else 1
            
        # 统计特征
        amps = [bi.amp() for bi in bi_list]
        features['avg_bi_amp'] = np.mean(amps) if amps else 0
        features['max_bi_amp'] = np.max(amps) if amps else 0
        features['min_bi_amp'] = np.min(amps) if amps else 0
        features['bi_amp_std'] = np.std(amps) if amps else 0
        
        # 趋势强度
        up_amp_sum = sum(bi.amp() for bi in up_bis)
        down_amp_sum = sum(bi.amp() for bi in down_bis)
        total_amp = up_amp_sum + down_amp_sum
        features['bi_trend_strength'] = abs(up_amp_sum - down_amp_sum) / total_amp if total_amp > 0 else 0
            
        # 反转信号
        if len(bi_list) >= 3 and len(set(bi.dir for bi in bi_list[-3:])) > 1:
            features['bi_reversal_signal'] = 1
                
        return features
        
    def _extract_seg_features(self, seg_list) -> Dict[str, Any]:
        """提取线段特征"""
        if not seg_list:
            return {}

        features = {}
        features['seg_count'] = len(seg_list)
        up_segs = [s for s in seg_list if s.is_up()]
        down_segs = [s for s in seg_list if s.is_down()]

        features['seg_up_count'] = len(up_segs)
        features['seg_down_count'] = len(down_segs)
        features['seg_up_ratio'] = len(up_segs) / len(seg_list) if seg_list else 0

        # 当前线段
        current_seg = seg_list[-1]
        features['current_seg_dir'] = 1 if current_seg.is_up() else -1
        features['current_seg_amp'] = current_seg.amp()
        features['current_seg_length'] = current_seg.get_klu_cnt()
        features['current_seg_slope'] = current_seg.amp() / current_seg.get_klu_cnt() if current_seg.get_klu_cnt() > 0 else 0
        features['current_seg_bi_count'] = len(current_seg.bi_list)

        # 与上一同向线段比较
        if len(seg_list) > 2:
            last_seg = seg_list[-2]
            prev_same_dir_seg = seg_list[-3]
            if current_seg.dir == prev_same_dir_seg.dir:
                features['seg_amp_ratio'] = current_seg.amp() / prev_same_dir_seg.amp() if prev_same_dir_seg.amp() > 0 else 1
                features['seg_len_ratio'] = current_seg.get_klu_cnt() / prev_same_dir_seg.get_klu_cnt() if prev_same_dir_seg.get_klu_cnt() > 0 else 1

        # 统计特征
        amps = [s.amp() for s in seg_list]
        features['avg_seg_amp'] = np.mean(amps) if amps else 0
        features['max_seg_amp'] = np.max(amps) if amps else 0
        features['min_seg_amp'] = np.min(amps) if amps else 0
        features['seg_amp_std'] = np.std(amps) if amps else 0

        # 趋势强度
        up_amp_sum = sum(s.amp() for s in up_segs)
        down_amp_sum = sum(s.amp() for s in down_segs)
        total_amp = up_amp_sum + down_amp_sum
        features['seg_trend_strength'] = abs(up_amp_sum - down_amp_sum) / total_amp if total_amp > 0 else 0

        return features
        
    def _extract_zs_features(self, seg_list, current_price: float) -> Dict[str, Any]:
        """提取中枢特征"""
        all_zs = [zs for seg in seg_list for zs in seg.zs_lst]
        if not all_zs:
            return {}

        features = {}
        features['zs_count'] = len(all_zs)

        last_zs = all_zs[-1]
        features['last_zs_height'] = last_zs.high - last_zs.low
        # CZS 无 get_klu_cnt，这里用 KLine_Unit idx 差值 +1 近似表示宽度（含端点）
        try:
            features['last_zs_width'] = last_zs.end.idx - last_zs.begin.idx + 1
        except Exception:
            features['last_zs_width'] = 0
        # 计算中枢内成交量（如有 volume 字段）
        try:
            volume_sum = 0.0
            for bi in getattr(last_zs, 'bi_lst', []):
                # bi.klc_lst 返回 CKLine 序列
                for klc in bi.klc_lst:
                    for klu in klc.lst:
                        if hasattr(klu, 'volume') and klu.volume is not None:
                            volume_sum += klu.volume
            features['last_zs_volume'] = volume_sum
        except Exception as exc:
            print(f"⚠️  计算中枢成交量失败: {exc}")

        heights = [zs.high - zs.low for zs in all_zs]
        features['zs_avg_height'] = np.mean(heights) if heights else 0
        features['zs_max_height'] = np.max(heights) if heights else 0
        
        # 价格与最后一个中枢的位置关系
        if features['last_zs_height'] > 1e-8:
            features['price_vs_zs_high'] = (current_price - last_zs.high) / features['last_zs_height']
            features['price_vs_zs_low'] = (current_price - last_zs.low) / features['last_zs_height']
            features['is_in_zs'] = 1 if last_zs.low <= current_price <= last_zs.high else 0
        
        # 中枢趋势
        if len(all_zs) > 1:
            second_last_zs = all_zs[-2]
            # 使用 mid 属性替代
            features['zs_level_change'] = last_zs.mid - second_last_zs.mid
            features['is_zs_overlap'] = 1 if max(last_zs.low, second_last_zs.low) < min(last_zs.high, second_last_zs.high) else 0

        return features
        
    def _extract_bsp_features(self, chan: CChan) -> Dict[str, Any]:
        """提取买卖点特征"""
        kl_data = chan[KL_TYPE.K_60M]
        bsp_container = kl_data.bs_point_lst
        all_bsp = list(bsp_container.bsp_iter()) if hasattr(bsp_container, 'bsp_iter') else []
        if not all_bsp:
            return {}

        # 按 bi.idx 升序排序，确保时序一致
        all_bsp.sort(key=lambda p: p.bi.idx)

        features = {}
        features['bsp_count'] = len(all_bsp)
        buy_points = [p for p in all_bsp if p.is_buy]
        sell_points = [p for p in all_bsp if not p.is_buy]
        
        last_bsp = all_bsp[-1]
        features['klines_since_last_bsp'] = len(kl_data) - last_bsp.klu.idx - 1
        features['last_bsp_is_buy'] = 1 if last_bsp.is_buy else 0
        # last_bsp.type 可能是 List[BSP_TYPE]
        if isinstance(last_bsp.type, list):
            features['last_bsp_type'] = ",".join(str(t.value) for t in last_bsp.type)
        else:
            features['last_bsp_type'] = str(getattr(last_bsp.type, 'value', last_bsp.type))

        if buy_points:
            features['klines_since_last_buy_bsp'] = len(kl_data) - buy_points[-1].klu.idx - 1
        if sell_points:
            features['klines_since_last_sell_bsp'] = len(kl_data) - sell_points[-1].klu.idx - 1

        bsp_indices = [p.klu.idx for p in all_bsp]
        features['bsp_density_100'] = sum(1 for idx in bsp_indices if idx >= len(kl_data) - 100)

        return features

    def _extract_macd_features(self, bi_list, seg_list) -> Dict[str, Any]:
        """提取缠论相关的MACD特征"""
        features = {}
        
        # --- 替换为 Cal_MACD_peak 指标 ---
        try:
            if len(bi_list) >= 3 and bi_list[-1].dir == bi_list[-3].dir:
                peak_curr = bi_list[-1].Cal_MACD_peak()
                peak_prev = bi_list[-3].Cal_MACD_peak()
                if peak_prev > 1e-8:
                    features['bi_macd_divergence'] = peak_curr / peak_prev
        except Exception as exc:
            print(f"⚠️  计算笔 MACD 背驰失败: {exc}")

        try:
            if len(seg_list) >= 3 and seg_list[-1].dir == seg_list[-3].dir:
                # CSeg 没有 Cal_MACD_peak，改用 Cal_MACD_amp() 表征线段 MACD 强度
                peak_curr = seg_list[-1].Cal_MACD_amp()
                peak_prev = seg_list[-3].Cal_MACD_amp()
                if peak_prev > 1e-8:
                    features['seg_macd_divergence'] = peak_curr / peak_prev
        except Exception as exc:
            print(f"⚠️  计算线段 MACD 背驰失败: {exc}")

        return features
        
    def _extract_pattern_features(self, kl_data) -> Dict[str, Any]:
        """提取形态学特征（不依赖 kl_data.fx_list）"""
        features: Dict[str, Any] = {}

        # 遍历所有合并后的 K 线，统计分型（顶部/底部）
        top_cnt = 0
        bottom_cnt = 0
        for klc in kl_data:  # CKLine_List.__iter__ => yield CKLine
            if klc.fx == FX_TYPE.TOP:
                top_cnt += 1
            elif klc.fx == FX_TYPE.BOTTOM:
                bottom_cnt += 1

        features['fx_top_count'] = top_cnt
        features['fx_bottom_count'] = bottom_cnt

        # 近 4 笔双顶/双底形态检测
        bi_list = kl_data.bi_list
        if len(bi_list) >= 4:
            b1, b2, b3, b4 = bi_list[-4:]
            # 双底: down-up-down, 且两个底差不多
            if b1.is_down() and b2.is_up() and b3.is_down():
                if abs(b1._low() - b3._low()) / max(1e-8, b1._low()) < 0.03: # 3% 容忍度
                    features['is_double_bottom'] = 1
            # 双顶: up-down-up, 且两个顶差不多
            if b1.is_up() and b2.is_down() and b3.is_up():
                if abs(b1._high() - b3._high()) / max(1e-8, b1._high()) < 0.03:
                    features['is_double_top'] = 1

        return features

    def _extract_trend_features(self, seg_list, bi_list) -> Dict[str, Any]:
        """提取趋势特征"""
        features = {}
        
        # 基于线段的趋势
        if len(seg_list) >= 2:
            s1, s2 = seg_list[-2:]
            if s1.is_up() and s2.is_up() and s2.high > s1.high and s2.low > s1.low:
                features['trend_direction'] = 2 # 强上涨
            elif s1.is_down() and s2.is_down() and s2.low < s1.low and s2.high < s1.high:
                features['trend_direction'] = 0 # 强下跌
            else:
                features['trend_direction'] = 1 # 震荡
        
        # 基于笔的趋势一致性
        if len(bi_list) >= 5:
            last_5_bis = bi_list[-5:]
            up_amps = [b.amp() for b in last_5_bis if b.is_up()]
            down_amps = [b.amp() for b in last_5_bis if b.is_down()]
            if np.std(up_amps) > 0 and np.std(down_amps) > 0:
                features['trend_consistency'] = np.std(up_amps) / np.std(down_amps)

        return features

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理特征"""
        # 处理无穷大和NaN值
        df = df.replace([np.inf, -np.inf], np.nan)
        # 前向填充再以0填补
        df = df.fillna(method='ffill').fillna(0)
        # 删除整列仍全为0或NaN的冗余列（如分形维度计算失败导致全NaN）
        df = df.dropna(axis=1, how='all')
        
        return df
        
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """获取特征名称 - 修复：排除标签列"""
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime', 
                       'future_returns', 'direction', 'direction_3cls', 'binary_direction']
        return [col for col in df.columns if col not in exclude_cols]
        
    def get_feature_count(self, df: pd.DataFrame) -> int:
        """获取特征数量"""
        return len(self.get_feature_names(df))
        
    def create_labels(self, df: pd.DataFrame, horizon: int = 24, threshold: float = 0.02) -> pd.DataFrame:
        """创建预测标签 - 完全修复数据泄露问题"""
        # 首先移除最后horizon行（这些行没有未来数据可用于标签）
        if len(df) <= horizon:
            raise ValueError(f"数据长度 {len(df)} 必须大于预测时间窗口 {horizon}")
        
        # 只保留可以计算标签的数据
        labels_df = df.iloc[:-horizon].copy()
        
        # 计算未来收益率（使用正确的索引方式，避免数据泄露）
        future_returns = []
        for i in range(len(labels_df)):
            current_price = df.iloc[i]['close']
            future_price = df.iloc[i + horizon]['close']  # 使用正确的未来价格
            future_return = (future_price / current_price) - 1
            future_returns.append(future_return)
        
        labels_df['future_returns'] = future_returns
        
        # 分类标签
        labels_df['direction'] = 0  # 横盘
        labels_df.loc[labels_df['future_returns'] > threshold, 'direction'] = 1  # 上涨
        labels_df.loc[labels_df['future_returns'] < -threshold, 'direction'] = -1  # 下跌
        
        # 三分类标签：0=下跌,1=横盘,2=上涨（兼容各ML库需要非负整数）
        labels_df['direction_3cls'] = labels_df['direction'] + 1
        
        # 二分类标签
        labels_df['binary_direction'] = (labels_df['future_returns'] > 0).astype(float)
        
        print(f"📊 标签创建完成: {len(labels_df)} 条记录")
        if 'binary_direction' in labels_df.columns:
            print(f"📈 正样本比例: {labels_df['binary_direction'].mean():.3f}")
            print(f"🎯 标签分布: {labels_df['binary_direction'].value_counts().to_dict()}")
        
        return labels_df

    def _add_funding_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """资金费率相关特征 (专为已对齐的 fundingRate 列设计)"""
        if 'fundingRate' not in df.columns:
            # 如果列缺失直接返回原 df
            return df

        fr = df['fundingRate']

        # 当前资金费率
        df['fr_current'] = fr

        # 近 24 根 (24h) 平均资金费率
        df['fr_mean_24h'] = fr.rolling(window=24, min_periods=1).mean()

        # 与 24h 均值的偏差
        df['fr_delta'] = fr - df['fr_mean_24h']

        # 30 日 (约 30*24 根) 95% 分位阈值
        extreme_threshold = fr.rolling(window=24 * 30, min_periods=24).quantile(0.95)
        df['fr_extreme_threshold'] = extreme_threshold
        df['fr_extreme_flag'] = (fr > extreme_threshold).astype(int)

        # 简化版 AR(1) 预测：使用上一根 fundingRate 作为预测
        df['fr_ar1_pred'] = fr.shift(1)

        return df 