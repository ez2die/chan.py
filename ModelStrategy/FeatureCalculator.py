import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import sys
from pathlib import Path
import json

# 添加项目根目录
sys.path.append(str(Path(__file__).parent.parent))

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE, BSP_TYPE, BI_DIR, FX_TYPE
from Common.CTime import CTime
from KLine.KLine_Unit import CKLine_Unit
from tools.data_converter import ChanDataConverter

class ChanFeatureCalculator:
    """Chan.py完整特征计算器 - 集成所有缠论特征"""
    
    def __init__(self):
        self.feature_config = {
            'basic_features': True,        # 基础价格特征
            'technical_indicators': True,  # 技术指标特征
            'volume_features': True,       # 成交量特征
            'chan_features': True,         # 缠论特征 (新增)
            'bi_features': True,           # 笔特征
            'seg_features': True,          # 线段特征
            'zs_features': True,           # 中枢特征
            'bsp_features': True,          # 买卖点特征
            'macd_features': True,         # MACD相关特征
            'pattern_features': True,      # 形态特征
            'trend_features': True,        # 趋势特征
        }
        
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
        """计算所有特征"""
        features_df = df.copy()
        
        # 基础特征
        if self.feature_config['basic_features']:
            features_df = self._add_basic_features(features_df)
            
        # 技术指标特征
        if self.feature_config['technical_indicators']:
            features_df = self._add_technical_features(features_df)
            
        # 成交量特征
        if self.feature_config['volume_features']:
            features_df = self._add_volume_features(features_df)
            
        # 缠论特征
        if self.feature_config['chan_features']:
            features_df = self._add_chan_features(features_df)
            
        return features_df
        
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加基础价格特征"""
        # 收益率特征
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(24).std()
        
        # 价格相对位置
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # K线形态特征
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']
        
        # 价格范围特征
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # 价格变化特征
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = df['price_change'] / df['open']
        df['high_change'] = df['high'] - df['close'].shift(1)
        df['low_change'] = df['low'] - df['close'].shift(1)
        
        return df
        
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标特征"""
        # 移动平均线
        for window in [5, 10, 20, 50, 100]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            
        # 指数移动平均
        for span in [12, 26, 50]:
            df[f'ema_{span}'] = df['close'].ewm(span=span).mean()
            
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
        
        # KDJ指标
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        df['kdj_k'] = rsv.ewm(com=2).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        # 威廉指标
        df['williams_r'] = (high_max - df['close']) / (high_max - low_min) * -100
        
        # ATR (平均真实波幅)
        df['atr'] = df['true_range'].rolling(14).mean()
        
        # 动量指标
        df['momentum'] = df['close'] / df['close'].shift(10) - 1
        df['roc'] = df['close'].pct_change(10)
        
        return df
        
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量特征"""
        if 'volume' not in df.columns:
            return df
            
        # 成交量移动平均
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # 价量关系
        df['price_volume_trend'] = df['returns'] * df['volume_ratio']
        df['volume_price_trend'] = df['volume'] * df['price_change_pct']
        
        # 成交量率
        df['volume_rate'] = df['volume'].pct_change()
        
        # OBV (能量潮)
        df['obv'] = (df['volume'] * np.sign(df['returns'])).cumsum()
        
        # 成交量分布
        df['volume_std'] = df['volume'].rolling(20).std()
        df['volume_cv'] = df['volume_std'] / df['volume_sma_20']
        
        return df
        
    def _add_chan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加缠论特征 - 这是核心功能"""
        try:
            # 转换数据格式为chan.py格式
            chan_data = self._convert_to_chan_format(df)
            
            # 创建Chan对象并计算缠论结构
            chan = CChan(
                code="TEMP",
                begin_time=None,
                end_time=None,
                data_src=DATA_SRC.CUSTOM,
                lv_list=[KL_TYPE.K_1H],
                config=self.chan_config,
                autype=AUTYPE.NONE,
                data=chan_data
            )
            
            # 提取缠论特征
            chan_features_df = self._extract_chan_features(df, chan)
            
            # 合并特征
            for col in chan_features_df.columns:
                if col not in df.columns:
                    df[col] = chan_features_df[col]
                    
        except Exception as e:
            print(f"Warning: 缠论特征计算失败: {e}")
            # 如果缠论计算失败，添加空特征避免后续错误
            df = self._add_empty_chan_features(df)
            
        return df
        
    def _convert_to_chan_format(self, df: pd.DataFrame) -> List[CKLine_Unit]:
        """将DataFrame转换为chan.py格式"""
        kline_units = []
        
        for idx, row in df.iterrows():
            # 创建时间对象
            if 'timestamp' in row:
                time_obj = CTime.from_timestamp(row['timestamp'])
            elif hasattr(idx, 'strftime'):
                time_obj = CTime.from_str(idx.strftime('%Y/%m/%d %H:%M'))
            else:
                time_obj = CTime.from_str(f"2024/01/01 {idx:02d}:00")
            
            # 创建K线单元
            klu = CKLine_Unit(
                time_key=time_obj,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0)),
                turnover=float(row.get('turnover', 0)),
                turnover_rate=float(row.get('turnover_rate', 0))
            )
            kline_units.append(klu)
            
        return kline_units
        
    def _extract_chan_features(self, df: pd.DataFrame, chan: CChan) -> pd.DataFrame:
        """从Chan对象中提取所有缠论特征"""
        features_df = pd.DataFrame(index=df.index)
        
        # 获取缠论结构
        kl_data = chan[0]  # 获取第一级别数据
        bi_list = kl_data.bi_list if hasattr(kl_data, 'bi_list') else []
        seg_list = kl_data.seg_list if hasattr(kl_data, 'seg_list') else []
        
        # 初始化特征列
        self._init_chan_feature_columns(features_df)
        
        # 提取笔特征
        if self.feature_config['bi_features']:
            self._extract_bi_features(features_df, bi_list)
            
        # 提取线段特征
        if self.feature_config['seg_features']:
            self._extract_seg_features(features_df, seg_list)
            
        # 提取中枢特征
        if self.feature_config['zs_features']:
            self._extract_zs_features(features_df, seg_list)
            
        # 提取买卖点特征
        if self.feature_config['bsp_features']:
            self._extract_bsp_features(features_df, chan)
            
        # 提取MACD特征
        if self.feature_config['macd_features']:
            self._extract_macd_features(features_df, bi_list)
            
        # 提取形态特征
        if self.feature_config['pattern_features']:
            self._extract_pattern_features(features_df, kl_data)
            
        # 提取趋势特征
        if self.feature_config['trend_features']:
            self._extract_trend_features(features_df, seg_list)
            
        return features_df
        
    def _init_chan_feature_columns(self, df: pd.DataFrame):
        """初始化缠论特征列"""
        # 笔特征
        bi_features = [
            'bi_count', 'bi_up_count', 'bi_down_count', 'bi_up_ratio',
            'current_bi_dir', 'current_bi_amp', 'current_bi_length',
            'last_bi_amp', 'last_bi_length', 'bi_amp_ratio',
            'avg_bi_amp', 'max_bi_amp', 'min_bi_amp', 'bi_amp_std',
            'bi_trend_strength', 'bi_reversal_signal'
        ]
        
        # 线段特征
        seg_features = [
            'seg_count', 'seg_up_count', 'seg_down_count', 'seg_up_ratio',
            'current_seg_dir', 'current_seg_amp', 'current_seg_length',
            'last_seg_amp', 'last_seg_length', 'seg_amp_ratio',
            'avg_seg_amp', 'max_seg_amp', 'min_seg_amp', 'seg_amp_std',
            'seg_trend_strength', 'seg_bi_count'
        ]
        
        # 中枢特征
        zs_features = [
            'zs_count', 'current_zs_height', 'current_zs_width', 'current_zs_strength',
            'zs_avg_height', 'zs_max_height', 'zs_min_height', 'zs_height_std',
            'zs_position', 'zs_break_strength', 'zs_oscillation_count',
            'distance_to_zs', 'zs_relative_position'
        ]
        
        # 买卖点特征
        bsp_features = [
            'bsp_count', 'buy_point_count', 'sell_point_count',
            'bsp1_count', 'bsp2_count', 'bsp3_count',
            'last_bsp_type', 'last_bsp_strength', 'bsp_density',
            'bsp_success_rate', 'bsp_avg_amplitude'
        ]
        
        # MACD特征
        macd_features = [
            'macd_divergence_strength', 'macd_trend_consistency',
            'macd_peak_count', 'macd_valley_count',
            'macd_area_ratio', 'macd_slope_strength'
        ]
        
        # 形态特征
        pattern_features = [
            'fx_top_count', 'fx_bottom_count', 'fx_strength',
            'consolidation_strength', 'breakout_strength',
            'support_resistance_strength'
        ]
        
        # 趋势特征
        trend_features = [
            'trend_direction', 'trend_strength', 'trend_consistency',
            'trend_duration', 'trend_acceleration'
        ]
        
        # 初始化所有特征为0
        all_features = (bi_features + seg_features + zs_features + 
                       bsp_features + macd_features + pattern_features + trend_features)
        
        for feature in all_features:
            df[feature] = 0.0
            
    def _extract_bi_features(self, df: pd.DataFrame, bi_list):
        """提取笔特征"""
        if not bi_list or len(bi_list) == 0:
            return
            
        # 基础统计
        df['bi_count'] = len(bi_list)
        up_bis = [bi for bi in bi_list if bi.is_up()]
        down_bis = [bi for bi in bi_list if bi.is_down()]
        
        df['bi_up_count'] = len(up_bis)
        df['bi_down_count'] = len(down_bis)
        df['bi_up_ratio'] = len(up_bis) / len(bi_list) if bi_list else 0
        
        # 当前笔特征
        if bi_list:
            current_bi = bi_list[-1]
            df['current_bi_dir'] = 1 if current_bi.is_up() else -1
            df['current_bi_amp'] = current_bi.amp()
            df['current_bi_length'] = current_bi.get_klu_cnt()
            
            # 上一笔特征
            if len(bi_list) > 1:
                last_bi = bi_list[-2]
                df['last_bi_amp'] = last_bi.amp()
                df['last_bi_length'] = last_bi.get_klu_cnt()
                df['bi_amp_ratio'] = current_bi.amp() / last_bi.amp() if last_bi.amp() > 0 else 1
                
        # 统计特征
        if bi_list:
            amps = [bi.amp() for bi in bi_list]
            df['avg_bi_amp'] = np.mean(amps)
            df['max_bi_amp'] = np.max(amps)
            df['min_bi_amp'] = np.min(amps)
            df['bi_amp_std'] = np.std(amps)
            
            # 趋势强度
            up_amp_sum = sum(bi.amp() for bi in up_bis)
            down_amp_sum = sum(bi.amp() for bi in down_bis)
            total_amp = up_amp_sum + down_amp_sum
            df['bi_trend_strength'] = abs(up_amp_sum - down_amp_sum) / total_amp if total_amp > 0 else 0
            
        # 反转信号
        if len(bi_list) >= 3:
            recent_bis = bi_list[-3:]
            if len(set(bi.dir for bi in recent_bis)) > 1:
                df['bi_reversal_signal'] = 1
                
    def _extract_seg_features(self, df: pd.DataFrame, seg_list):
        """提取线段特征"""
        if not seg_list or len(seg_list) == 0:
            return
            
        # 基础统计
        df['seg_count'] = len(seg_list)
        up_segs = [seg for seg in seg_list if seg.is_up()]
        down_segs = [seg for seg in seg_list if seg.is_down()]
        
        df['seg_up_count'] = len(up_segs)
        df['seg_down_count'] = len(down_segs)
        df['seg_up_ratio'] = len(up_segs) / len(seg_list) if seg_list else 0
        
        # 当前线段特征
        if seg_list:
            current_seg = seg_list[-1]
            df['current_seg_dir'] = 1 if current_seg.is_up() else -1
            df['current_seg_amp'] = current_seg.amp()
            df['current_seg_length'] = current_seg.cal_bi_cnt()
            df['seg_bi_count'] = current_seg.cal_bi_cnt()
            
            # 上一线段特征
            if len(seg_list) > 1:
                last_seg = seg_list[-2]
                df['last_seg_amp'] = last_seg.amp()
                df['last_seg_length'] = last_seg.cal_bi_cnt()
                df['seg_amp_ratio'] = current_seg.amp() / last_seg.amp() if last_seg.amp() > 0 else 1
                
        # 统计特征
        if seg_list:
            amps = [seg.amp() for seg in seg_list]
            df['avg_seg_amp'] = np.mean(amps)
            df['max_seg_amp'] = np.max(amps)
            df['min_seg_amp'] = np.min(amps)
            df['seg_amp_std'] = np.std(amps)
            
            # 趋势强度
            up_amp_sum = sum(seg.amp() for seg in up_segs)
            down_amp_sum = sum(seg.amp() for seg in down_segs)
            total_amp = up_amp_sum + down_amp_sum
            df['seg_trend_strength'] = abs(up_amp_sum - down_amp_sum) / total_amp if total_amp > 0 else 0
            
    def _extract_zs_features(self, df: pd.DataFrame, seg_list):
        """提取中枢特征"""
        if not seg_list:
            return
            
        # 收集所有中枢
        all_zs = []
        for seg in seg_list:
            if hasattr(seg, 'zs_lst') and seg.zs_lst:
                all_zs.extend(seg.zs_lst)
                
        if not all_zs:
            return
            
        # 基础统计
        df['zs_count'] = len(all_zs)
        
        # 当前中枢特征
        if all_zs:
            current_zs = all_zs[-1]
            df['current_zs_height'] = current_zs.high - current_zs.low
            df['current_zs_width'] = current_zs.end.idx - current_zs.begin.idx
            df['current_zs_strength'] = df['current_zs_height'] / current_zs.mid if current_zs.mid > 0 else 0
            
        # 统计特征
        heights = [zs.high - zs.low for zs in all_zs]
        df['zs_avg_height'] = np.mean(heights)
        df['zs_max_height'] = np.max(heights)
        df['zs_min_height'] = np.min(heights)
        df['zs_height_std'] = np.std(heights)
        
        # 中枢位置特征
        if all_zs:
            last_zs = all_zs[-1]
            df['zs_position'] = (last_zs.mid - last_zs.low) / (last_zs.high - last_zs.low) if last_zs.high > last_zs.low else 0.5
            
            # 突破强度
            if hasattr(last_zs, 'bi_out') and last_zs.bi_out:
                if last_zs.bi_out.is_up():
                    df['zs_break_strength'] = (last_zs.bi_out.get_end_val() - last_zs.high) / last_zs.high
                else:
                    df['zs_break_strength'] = (last_zs.low - last_zs.bi_out.get_end_val()) / last_zs.low
                    
        # 震荡次数
        if all_zs:
            oscillations = 0
            for zs in all_zs:
                if hasattr(zs, 'bi_lst'):
                    oscillations += len(zs.bi_lst)
            df['zs_oscillation_count'] = oscillations
            
    def _extract_bsp_features(self, df: pd.DataFrame, chan: CChan):
        """提取买卖点特征"""
        try:
            bsp_list = chan.get_latest_bsp(number=0)  # 获取所有买卖点
            
            if not bsp_list:
                return
                
            # 基础统计
            df['bsp_count'] = len(bsp_list)
            buy_points = [bsp for bsp in bsp_list if bsp.is_buy]
            sell_points = [bsp for bsp in bsp_list if not bsp.is_buy]
            
            df['buy_point_count'] = len(buy_points)
            df['sell_point_count'] = len(sell_points)
            
            # 按类型统计
            bsp1_count = len([bsp for bsp in bsp_list if BSP_TYPE.T1 in bsp.type or BSP_TYPE.T1P in bsp.type])
            bsp2_count = len([bsp for bsp in bsp_list if BSP_TYPE.T2 in bsp.type or BSP_TYPE.T2S in bsp.type])
            bsp3_count = len([bsp for bsp in bsp_list if BSP_TYPE.T3A in bsp.type or BSP_TYPE.T3B in bsp.type])
            
            df['bsp1_count'] = bsp1_count
            df['bsp2_count'] = bsp2_count
            df['bsp3_count'] = bsp3_count
            
            # 最近买卖点特征
            if bsp_list:
                last_bsp = bsp_list[-1]
                df['last_bsp_type'] = 1 if last_bsp.is_buy else -1
                
                # 买卖点强度（基于特征）
                if hasattr(last_bsp, 'features') and last_bsp.features:
                    strength = 0
                    for feat_name, feat_value in last_bsp.features.items():
                        if 'amp' in feat_name.lower():
                            strength += feat_value
                    df['last_bsp_strength'] = strength
                    
            # 买卖点密度
            if len(df) > 0:
                df['bsp_density'] = len(bsp_list) / len(df)
                
        except Exception as e:
            print(f"Warning: 买卖点特征提取失败: {e}")
            
    def _extract_macd_features(self, df: pd.DataFrame, bi_list):
        """提取MACD相关特征"""
        if not bi_list:
            return
            
        try:
            # MACD背离强度
            divergence_strengths = []
            for bi in bi_list:
                if hasattr(bi, 'cal_macd_metric'):
                    try:
                        metric = bi.cal_macd_metric('peak', False)
                        divergence_strengths.append(metric)
                    except:
                        pass
                        
            if divergence_strengths:
                df['macd_divergence_strength'] = np.mean(divergence_strengths)
                
            # MACD趋势一致性
            macd_trends = []
            for bi in bi_list:
                if hasattr(bi, 'Cal_MACD_area'):
                    try:
                        area = bi.Cal_MACD_area()
                        macd_trends.append(1 if area > 0 else -1)
                    except:
                        pass
                        
            if macd_trends:
                df['macd_trend_consistency'] = abs(np.mean(macd_trends))
                
        except Exception as e:
            print(f"Warning: MACD特征提取失败: {e}")
            
    def _extract_pattern_features(self, df: pd.DataFrame, kl_data):
        """提取形态特征"""
        try:
            # 分型统计
            if hasattr(kl_data, 'fx_list'):
                fx_list = kl_data.fx_list
                top_fx = [fx for fx in fx_list if fx.fx == FX_TYPE.TOP]
                bottom_fx = [fx for fx in fx_list if fx.fx == FX_TYPE.BOTTOM]
                
                df['fx_top_count'] = len(top_fx)
                df['fx_bottom_count'] = len(bottom_fx)
                df['fx_strength'] = len(fx_list) / len(df) if len(df) > 0 else 0
                
        except Exception as e:
            print(f"Warning: 形态特征提取失败: {e}")
            
    def _extract_trend_features(self, df: pd.DataFrame, seg_list):
        """提取趋势特征"""
        if not seg_list or len(seg_list) < 2:
            return
            
        # 趋势方向
        recent_segs = seg_list[-3:] if len(seg_list) >= 3 else seg_list
        up_segs = sum(1 for seg in recent_segs if seg.is_up())
        down_segs = sum(1 for seg in recent_segs if seg.is_down())
        
        if up_segs > down_segs:
            df['trend_direction'] = 1
        elif down_segs > up_segs:
            df['trend_direction'] = -1
        else:
            df['trend_direction'] = 0
            
        # 趋势强度
        if recent_segs:
            amps = [seg.amp() for seg in recent_segs]
            df['trend_strength'] = np.mean(amps)
            
        # 趋势一致性
        if len(recent_segs) > 1:
            directions = [1 if seg.is_up() else -1 for seg in recent_segs]
            df['trend_consistency'] = abs(np.mean(directions))
            
        # 趋势持续时间
        df['trend_duration'] = len(recent_segs)
        
    def _add_empty_chan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加空的缠论特征（当计算失败时）"""
        empty_features = [
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
        
        for feature in empty_features:
            df[feature] = 0.0
            
        return df
        
    def create_labels(self, df: pd.DataFrame, horizon: int = 24, threshold: float = 0.02) -> pd.DataFrame:
        """创建预测标签"""
        labels_df = df.copy()
        
        # 未来收益率
        labels_df['future_returns'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # 分类标签
        labels_df['direction'] = 0  # 横盘
        labels_df.loc[labels_df['future_returns'] > threshold, 'direction'] = 1  # 上涨
        labels_df.loc[labels_df['future_returns'] < -threshold, 'direction'] = -1  # 下跌
        
        # 三分类标签：0=下跌,1=横盘,2=上涨
        labels_df['direction_3cls'] = labels_df['direction'] + 1
        
        # 二分类标签
        labels_df['binary_direction'] = (labels_df['future_returns'] > 0).astype(int)
        
        # 波动率标签
        labels_df['future_volatility'] = df['close'].rolling(horizon).std().shift(-horizon)
        
        # 高波动率标签
        vol_threshold = df['close'].rolling(horizon).std().quantile(0.8)
        labels_df['high_volatility'] = (labels_df['future_volatility'] > vol_threshold).astype(int)
        
        return labels_df
        
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """获取特征名称列表"""
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume', 'timestamp',
            'future_returns', 'direction', 'direction_3cls', 'binary_direction',
            'future_volatility', 'high_volatility'
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
        
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """获取特征重要性分组"""
        return {
            'basic_features': [
                'returns', 'log_returns', 'volatility', 'high_low_ratio',
                'close_open_ratio', 'price_position', 'body_size',
                'upper_shadow', 'lower_shadow', 'true_range',
                'price_change', 'price_change_pct', 'high_change', 'low_change'
            ],
            'technical_indicators': [
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100',
                'price_sma_5_ratio', 'price_sma_10_ratio', 'price_sma_20_ratio',
                'price_sma_50_ratio', 'price_sma_100_ratio',
                'ema_12', 'ema_26', 'ema_50', 'macd', 'macd_signal',
                'macd_histogram', 'rsi', 'bb_upper', 'bb_lower', 'bb_position', 'bb_width',
                'kdj_k', 'kdj_d', 'kdj_j', 'williams_r', 'atr',
                'momentum', 'roc'
            ],
            'volume_features': [
                'volume_sma_20', 'volume_ratio', 'price_volume_trend',
                'volume_price_trend', 'volume_rate', 'obv',
                'volume_std', 'volume_cv'
            ],
            'bi_features': [
                'bi_count', 'bi_up_count', 'bi_down_count', 'bi_up_ratio',
                'current_bi_dir', 'current_bi_amp', 'current_bi_length',
                'last_bi_amp', 'last_bi_length', 'bi_amp_ratio',
                'avg_bi_amp', 'max_bi_amp', 'min_bi_amp', 'bi_amp_std',
                'bi_trend_strength', 'bi_reversal_signal'
            ],
            'seg_features': [
                'seg_count', 'seg_up_count', 'seg_down_count', 'seg_up_ratio',
                'current_seg_dir', 'current_seg_amp', 'current_seg_length',
                'last_seg_amp', 'last_seg_length', 'seg_amp_ratio',
                'avg_seg_amp', 'max_seg_amp', 'min_seg_amp', 'seg_amp_std',
                'seg_trend_strength', 'seg_bi_count'
            ],
            'zs_features': [
                'zs_count', 'current_zs_height', 'current_zs_width', 'current_zs_strength',
                'zs_avg_height', 'zs_max_height', 'zs_min_height', 'zs_height_std',
                'zs_position', 'zs_break_strength', 'zs_oscillation_count',
                'distance_to_zs', 'zs_relative_position'
            ],
            'bsp_features': [
                'bsp_count', 'buy_point_count', 'sell_point_count',
                'bsp1_count', 'bsp2_count', 'bsp3_count',
                'last_bsp_type', 'last_bsp_strength', 'bsp_density',
                'bsp_success_rate', 'bsp_avg_amplitude'
            ],
            'macd_features': [
                'macd_divergence_strength', 'macd_trend_consistency',
                'macd_peak_count', 'macd_valley_count',
                'macd_area_ratio', 'macd_slope_strength'
            ],
            'pattern_features': [
                'fx_top_count', 'fx_bottom_count', 'fx_strength',
                'consolidation_strength', 'breakout_strength',
                'support_resistance_strength'
            ],
            'trend_features': [
                'trend_direction', 'trend_strength', 'trend_consistency',
                'trend_duration', 'trend_acceleration'
            ]
        }
        
    def calculate_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """计算特征统计信息"""
        feature_names = self.get_feature_names(df)
        
        stats = {
            'total_features': len(feature_names),
            'feature_groups': {},
            'missing_values': {},
            'correlation_high': []
        }
        
        # 按组统计特征
        groups = self.get_feature_importance_groups()
        for group_name, group_features in groups.items():
            available_features = [f for f in group_features if f in feature_names]
            stats['feature_groups'][group_name] = {
                'count': len(available_features),
                'features': available_features
            }
            
        # 缺失值统计
        for feature in feature_names:
            if feature in df.columns:
                missing_count = df[feature].isnull().sum()
                if missing_count > 0:
                    stats['missing_values'][feature] = missing_count
                    
        # 高相关性特征
        if len(feature_names) > 1:
            feature_df = df[feature_names].select_dtypes(include=[np.number])
            if len(feature_df.columns) > 1:
                corr_matrix = feature_df.corr().abs()
                high_corr = np.where((corr_matrix > 0.9) & (corr_matrix < 1.0))
                for i, j in zip(high_corr[0], high_corr[1]):
                    if i < j:  # 避免重复
                        stats['correlation_high'].append({
                            'feature1': feature_df.columns[i],
                            'feature2': feature_df.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
                        
        return stats

    # 向后兼容的方法名
    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """向后兼容：计算基础特征"""
        return self.calculate_all_features(df) 