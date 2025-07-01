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
            "trigger_step": False,
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
        
        for window in all_windows:
            if window >= len(df):
                continue
                
            # 价格统计
            df[f'high_max_{window}'] = df['high'].rolling(window).max()
            df[f'low_min_{window}'] = df['low'].rolling(window).min()
            df[f'price_position_{window}'] = (df['close'] - df[f'low_min_{window}']) / (df[f'high_max_{window}'] - df[f'low_min_{window}'])
            
            # 波动率
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            if 'volatility' in df.columns:
                df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df['volatility']
            else:
                df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df['returns'].rolling(24).std()
            
            # 趋势强度
            df[f'trend_strength_{window}'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
            
            # 价格范围
            df[f'price_range_avg_{window}'] = df['price_range'].rolling(window).mean()
            df[f'price_range_ratio_{window}'] = df['price_range'] / df[f'price_range_avg_{window}']
            
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
            df['datetime'] = pd.to_datetime(df['timestamp'])
        elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            df['datetime'] = df.index
        else:
            # 如果没有时间信息，跳过时间特征
            return df
            
        # 基础时间特征
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['week_of_year'] = df['datetime'].dt.isocalendar().week
        df['month'] = df['datetime'].dt.month
        
        # 时间周期特征
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # 交易时段特征（UTC时间）
        df['asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
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
        
        return df
        
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """动量特征"""
        # 多周期动量
        momentum_periods = [3, 5, 10, 20, 50]
        for period in momentum_periods:
            df[f'momentum_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
            df[f'momentum_{period}_ma'] = df[f'momentum_{period}'].rolling(10).mean()
            
        # 动量加速度
        df['momentum_acceleration_5'] = df['momentum_5'] - df['momentum_5'].shift(5)
        df['momentum_acceleration_20'] = df['momentum_20'] - df['momentum_20'].shift(5)
        
        # 相对强弱
        df['relative_strength'] = df['momentum_20'] / df['momentum_20'].rolling(50).std()
        
        # 动量背离
        price_momentum = df['momentum_20']
        volume_momentum = df['volume'].pct_change(20) if 'volume' in df.columns else 0
        df['momentum_divergence'] = price_momentum - volume_momentum
        
        return df
        
    def _add_chan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """缠论特征 - 增强版错误处理"""
        import warnings
        
        # 抑制性能警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            
            try:
                # 转换数据格式
                chan_data = self._convert_to_chan_format(df)
                
                if len(chan_data) < 10:  # 数据太少，无法进行缠论分析
                    print(f"警告: 数据量不足 ({len(chan_data)} < 10)，跳过缠论特征计算")
                    df = self._add_empty_chan_features(df)
                    return df
                
                # 创建Chan对象并传入数据
                chan = CChan(
                    code="TEMP",
                    begin_time=None,
                    end_time=None,
                    data_src="custom:temp.TempDataSource",  # 使用字符串形式
                    lv_list=[KL_TYPE.K_60M],  # 使用1小时级别
                    config=self.chan_config,
                    autype=AUTYPE.NONE
                )
                
                # 使用trigger_load方法传入数据
                chan.trigger_load({KL_TYPE.K_60M: chan_data})
                
                # 提取缠论特征
                df = self._extract_chan_features(df, chan)
                print(f"✅ 缠论特征计算成功，数据量: {len(chan_data)}")
                
            except Exception as e:
                print(f"⚠️ 缠论特征计算失败，使用降级处理: {e}")
                df = self._add_empty_chan_features(df)
                
        return df
        
    def _add_advanced_chan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级缠论特征"""
        # 这里可以添加更复杂的缠论特征
        # 比如多级别联动、分形特征等
        
        # 分形维度代理
        df['fractal_dimension'] = self._calculate_fractal_dimension(df)
        
        # 市场状态识别
        df['market_regime'] = self._identify_market_regime(df)
        
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
        """转换为Chan格式 - 修复时间转换问题"""
        from Common.CEnum import DATA_FIELD
        
        chan_data = []
        for i, row in df.iterrows():
            try:
                # 修复时间转换 - 使用正确的CTime构造方法
                if 'timestamp' in row and pd.notna(row['timestamp']):
                    # 如果有timestamp，从时间戳创建
                    dt = pd.to_datetime(row['timestamp'])
                    time_obj = CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
                elif hasattr(i, 'year'):
                    # 如果索引是datetime类型
                    time_obj = CTime(i.year, i.month, i.day, i.hour, i.minute, 0)
                else:
                    # 使用默认时间
                    base_year = 2024
                    base_month = 1
                    base_day = 1
                    hour = len(chan_data) % 24
                    time_obj = CTime(base_year, base_month, base_day, hour, 0, 0)
                
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
                # 如果转换失败，跳过这一行
                print(f"警告: K线数据转换失败 (行 {i}): {e}")
                continue
                
        return chan_data
        
    def _extract_chan_features(self, df: pd.DataFrame, chan: CChan) -> pd.DataFrame:
        """提取缠论特征"""
        # 初始化缠论特征列
        self._init_chan_feature_columns(df)
        
        # 获取缠论结构
        kl_data = chan[KL_TYPE.K_1H]
        bi_list = kl_data.bi_list
        seg_list = kl_data.seg_list
        
        # 提取各类特征
        self._extract_bi_features(df, bi_list)
        self._extract_seg_features(df, seg_list)
        self._extract_zs_features(df, seg_list)
        self._extract_bsp_features(df, chan)
        self._extract_macd_features(df, bi_list)
        self._extract_pattern_features(df, kl_data)
        self._extract_trend_features(df, seg_list)
        
        return df
        
    def _init_chan_feature_columns(self, df: pd.DataFrame):
        """初始化缠论特征列"""
        chan_features = [
            # 笔特征
            'bi_count', 'bi_up_count', 'bi_down_count', 'bi_up_ratio',
            'current_bi_dir', 'current_bi_amp', 'current_bi_length',
            'last_bi_amp', 'last_bi_length', 'bi_amp_ratio',
            'avg_bi_amp', 'max_bi_amp', 'min_bi_amp', 'bi_amp_std',
            'bi_trend_strength', 'bi_reversal_signal',
            
            # 线段特征
            'seg_count', 'seg_up_count', 'seg_down_count', 'seg_up_ratio',
            'current_seg_dir', 'current_seg_amp', 'current_seg_length',
            'last_seg_amp', 'last_seg_length', 'seg_amp_ratio',
            'avg_seg_amp', 'max_seg_amp', 'min_seg_amp', 'seg_amp_std',
            'seg_trend_strength', 'seg_bi_count',
            
            # 中枢特征
            'zs_count', 'current_zs_height', 'current_zs_width', 'current_zs_strength',
            'zs_avg_height', 'zs_max_height', 'zs_min_height', 'zs_height_std',
            'zs_position', 'zs_break_strength', 'zs_oscillation_count',
            'distance_to_zs', 'zs_relative_position',
            
            # 买卖点特征
            'bsp_count', 'buy_point_count', 'sell_point_count',
            'bsp1_count', 'bsp2_count', 'bsp3_count',
            'last_bsp_type', 'last_bsp_strength', 'bsp_density',
            'bsp_success_rate', 'bsp_avg_amplitude',
            
            # MACD特征
            'macd_divergence_strength', 'macd_trend_consistency',
            'macd_peak_count', 'macd_valley_count',
            'macd_area_ratio', 'macd_slope_strength',
            
            # 形态特征
            'fx_top_count', 'fx_bottom_count', 'fx_strength',
            'consolidation_strength', 'breakout_strength',
            'support_resistance_strength',
            
            # 趋势特征
            'trend_direction', 'trend_strength', 'trend_consistency',
            'trend_duration', 'trend_acceleration'
        ]
        
        for feature in chan_features:
            df[feature] = 0.0
            
    def _extract_bi_features(self, df: pd.DataFrame, bi_list):
        """提取笔特征"""
        if not bi_list or len(bi_list) == 0:
            return
            
        # 基础统计
        df.loc[df.index[-1], 'bi_count'] = len(bi_list)
        up_bis = [bi for bi in bi_list if bi.is_up()]
        down_bis = [bi for bi in bi_list if bi.is_down()]
        
        df.loc[df.index[-1], 'bi_up_count'] = len(up_bis)
        df.loc[df.index[-1], 'bi_down_count'] = len(down_bis)
        df.loc[df.index[-1], 'bi_up_ratio'] = len(up_bis) / len(bi_list) if bi_list else 0
        
        # 当前笔特征
        if bi_list:
            current_bi = bi_list[-1]
            df.loc[df.index[-1], 'current_bi_dir'] = 1 if current_bi.is_up() else -1
            df.loc[df.index[-1], 'current_bi_amp'] = current_bi.amp()
            df.loc[df.index[-1], 'current_bi_length'] = current_bi.get_klu_cnt()
            
            # 上一笔特征
            if len(bi_list) > 1:
                last_bi = bi_list[-2]
                df.loc[df.index[-1], 'last_bi_amp'] = last_bi.amp()
                df.loc[df.index[-1], 'last_bi_length'] = last_bi.get_klu_cnt()
                df.loc[df.index[-1], 'bi_amp_ratio'] = current_bi.amp() / last_bi.amp() if last_bi.amp() > 0 else 1
                
        # 统计特征
        if bi_list:
            amps = [bi.amp() for bi in bi_list]
            df.loc[df.index[-1], 'avg_bi_amp'] = np.mean(amps)
            df.loc[df.index[-1], 'max_bi_amp'] = np.max(amps)
            df.loc[df.index[-1], 'min_bi_amp'] = np.min(amps)
            df.loc[df.index[-1], 'bi_amp_std'] = np.std(amps)
            
            # 趋势强度
            up_amp_sum = sum(bi.amp() for bi in up_bis)
            down_amp_sum = sum(bi.amp() for bi in down_bis)
            total_amp = up_amp_sum + down_amp_sum
            df.loc[df.index[-1], 'bi_trend_strength'] = abs(up_amp_sum - down_amp_sum) / total_amp if total_amp > 0 else 0
            
        # 反转信号
        if len(bi_list) >= 3:
            recent_bis = bi_list[-3:]
            if len(set(bi.dir for bi in recent_bis)) > 1:
                df.loc[df.index[-1], 'bi_reversal_signal'] = 1
                
    def _extract_seg_features(self, df: pd.DataFrame, seg_list):
        """提取线段特征"""
        # 简化实现，类似笔特征
        if not seg_list or len(seg_list) == 0:
            return
            
        df.loc[df.index[-1], 'seg_count'] = len(seg_list)
        
    def _extract_zs_features(self, df: pd.DataFrame, seg_list):
        """提取中枢特征"""
        # 简化实现
        pass
        
    def _extract_bsp_features(self, df: pd.DataFrame, chan: CChan):
        """提取买卖点特征"""
        # 简化实现
        pass
        
    def _extract_macd_features(self, df: pd.DataFrame, bi_list):
        """提取MACD特征"""
        # 简化实现
        pass
        
    def _extract_pattern_features(self, df: pd.DataFrame, kl_data):
        """提取形态特征"""
        # 简化实现
        pass
        
    def _extract_trend_features(self, df: pd.DataFrame, seg_list):
        """提取趋势特征"""
        # 简化实现
        pass
        
    def _add_empty_chan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加空的缠论特征"""
        self._init_chan_feature_columns(df)
        return df
        
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理特征"""
        # 处理无穷大和NaN值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(0)
        
        return df
        
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """获取特征名称 - 修复：排除标签列"""
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime', 
                       'future_returns', 'direction', 'binary_direction']
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
        
        # 二分类标签
        labels_df['binary_direction'] = (labels_df['future_returns'] > 0).astype(float)
        
        print(f"📊 标签创建完成: {len(labels_df)} 条记录")
        if 'binary_direction' in labels_df.columns:
            print(f"📈 正样本比例: {labels_df['binary_direction'].mean():.3f}")
            print(f"🎯 标签分布: {labels_df['binary_direction'].value_counts().to_dict()}")
        
        return labels_df 