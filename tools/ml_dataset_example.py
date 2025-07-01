#!/usr/bin/env python3
"""
机器学习数据集使用示例
演示如何使用准备好的数据集进行chan.py机器学习
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from tools.data_converter import ChanDataConverter
from Common.CTime import CTime
from Common.CEnum import DATA_FIELD


class MLDatasetLoader:
    """机器学习数据集加载器"""
    
    def __init__(self, ml_data_dir: str = "data/ml_datasets"):
        self.ml_data_dir = ml_data_dir
        self.converter = ChanDataConverter()
    
    def load_dataset(self, dataset_file: str) -> Tuple[List, Dict]:
        """加载数据集"""
        file_path = os.path.join(self.ml_data_dir, dataset_file)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data['kline_data'], data['metadata']
    
    def load_funding_rate_data(self) -> pd.DataFrame:
        """加载资金费率数据"""
        funding_file = os.path.join(self.ml_data_dir, "funding_rate_processed.csv")
        
        if not os.path.exists(funding_file):
            raise FileNotFoundError(f"资金费率文件不存在: {funding_file}")
        
        df = pd.read_csv(funding_file, index_col=0, parse_dates=True)
        return df
    
    def kline_data_to_dataframe(self, kline_data: List[Dict]) -> pd.DataFrame:
        """将K线数据转换为DataFrame"""
        records = []
        
        for record in kline_data:
            # 解析时间
            time_str = record[DATA_FIELD.FIELD_TIME]
            if ' ' in time_str:
                dt = datetime.strptime(time_str, '%Y/%m/%d %H:%M')
            else:
                dt = datetime.strptime(time_str, '%Y/%m/%d')
            
            # 构建记录
            row = {
                'datetime': dt,
                'open': record[DATA_FIELD.FIELD_OPEN],
                'high': record[DATA_FIELD.FIELD_HIGH],
                'low': record[DATA_FIELD.FIELD_LOW],
                'close': record[DATA_FIELD.FIELD_CLOSE],
            }
            
            # 添加可选字段
            if DATA_FIELD.FIELD_VOLUME in record:
                row['volume'] = record[DATA_FIELD.FIELD_VOLUME]
            
            records.append(row)
        
        df = pd.DataFrame(records)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def create_features_dataframe(self, kline_data: List[Dict], metadata: Dict) -> pd.DataFrame:
        """创建包含所有特征的DataFrame"""
        # 基础OHLCV数据
        df = self.kline_data_to_dataframe(kline_data)
        
        # 从元数据中获取原始特征列
        original_features = metadata.get('features', [])
        
        print(f"原始特征列: {original_features}")
        print(f"数据集包含 {len(df)} 条记录")
        
        # 注意：这里的数据已经包含了技术指标，但在JSON转换过程中可能丢失
        # 实际使用时，需要重新计算技术指标或从原始数据源获取
        
        return df
    
    def prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备机器学习特征"""
        features_df = df.copy()
        
        # 基础价格特征
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        features_df['volatility'] = features_df['returns'].rolling(24).std()
        
        # 价格相对位置
        features_df['high_low_ratio'] = features_df['high'] / features_df['low']
        features_df['close_open_ratio'] = features_df['close'] / features_df['open']
        
        # 简单移动平均
        for window in [5, 10, 20, 50]:
            features_df[f'sma_{window}'] = features_df['close'].rolling(window).mean()
            features_df[f'price_sma_{window}_ratio'] = features_df['close'] / features_df[f'sma_{window}']
        
        # 指数移动平均
        for span in [12, 26, 50]:
            features_df[f'ema_{span}'] = features_df['close'].ewm(span=span).mean()
        
        # MACD
        ema_12 = features_df['close'].ewm(span=12).mean()
        ema_26 = features_df['close'].ewm(span=26).mean()
        features_df['macd'] = ema_12 - ema_26
        features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # RSI
        delta = features_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['rsi'] = 100 - (100 / (1 + rs))
        
        # 成交量特征（如果有的话）
        if 'volume' in features_df.columns:
            features_df['volume_sma_20'] = features_df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma_20']
        
        # 滞后特征
        for lag in [1, 2, 3, 6, 12, 24]:
            features_df[f'close_lag_{lag}'] = features_df['close'].shift(lag)
            features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
        
        # 删除NaN值
        features_df.dropna(inplace=True)
        
        return features_df
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 24, threshold: float = 0.02) -> pd.DataFrame:
        """创建标签（预测目标）"""
        labels_df = df.copy()
        
        # 未来收益率
        labels_df['future_returns'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # 分类标签：上涨/下跌/横盘
        labels_df['direction'] = 0  # 横盘
        labels_df.loc[labels_df['future_returns'] > threshold, 'direction'] = 1  # 上涨
        labels_df.loc[labels_df['future_returns'] < -threshold, 'direction'] = -1  # 下跌
        
        # 二分类标签：上涨/下跌
        labels_df['binary_direction'] = (labels_df['future_returns'] > 0).astype(int)
        
        # 波动率标签
        labels_df['future_volatility'] = df['close'].rolling(horizon).std().shift(-horizon)
        
        # 删除NaN值
        labels_df.dropna(inplace=True)
        
        return labels_df


def example_load_training_data():
    """示例：加载训练数据"""
    print("=" * 60)
    print("示例1: 加载主训练集数据")
    print("=" * 60)
    
    loader = MLDatasetLoader()
    
    # 加载主训练集
    kline_data, metadata = loader.load_dataset("main_training_BTC_USDT_1h_train.json")
    
    print(f"数据集描述: {metadata['description']}")
    print(f"记录数: {metadata['record_count']:,}")
    print(f"时间范围: {metadata['time_range']['start']} 至 {metadata['time_range']['end']}")
    print(f"原始特征: {len(metadata['features'])} 个")
    
    # 转换为DataFrame
    df = loader.kline_data_to_dataframe(kline_data)
    print(f"DataFrame形状: {df.shape}")
    print(f"前5行数据:")
    print(df.head())
    
    return df, metadata


def example_prepare_features():
    """示例：准备机器学习特征"""
    print("\n" + "=" * 60)
    print("示例2: 准备机器学习特征")
    print("=" * 60)
    
    loader = MLDatasetLoader()
    
    # 加载数据
    kline_data, metadata = loader.load_dataset("main_training_BTC_USDT_1h_train.json")
    df = loader.kline_data_to_dataframe(kline_data)
    
    # 准备特征
    features_df = loader.prepare_ml_features(df)
    
    print(f"特征工程后形状: {features_df.shape}")
    print(f"特征列数: {len(features_df.columns)}")
    print(f"特征列名: {features_df.columns.tolist()}")
    
    # 检查缺失值
    missing_values = features_df.isnull().sum()
    print(f"\n缺失值统计:")
    print(missing_values[missing_values > 0])
    
    return features_df


def example_create_labels():
    """示例：创建预测标签"""
    print("\n" + "=" * 60)
    print("示例3: 创建预测标签")
    print("=" * 60)
    
    loader = MLDatasetLoader()
    
    # 加载数据
    kline_data, metadata = loader.load_dataset("main_training_BTC_USDT_1h_train.json")
    df = loader.kline_data_to_dataframe(kline_data)
    
    # 创建标签
    labels_df = loader.create_labels(df, horizon=24, threshold=0.02)
    
    print(f"标签数据形状: {labels_df.shape}")
    print(f"标签分布:")
    print(labels_df['direction'].value_counts().sort_index())
    
    print(f"\n二分类标签分布:")
    print(labels_df['binary_direction'].value_counts())
    
    print(f"\n未来收益率统计:")
    print(labels_df['future_returns'].describe())
    
    return labels_df


def example_funding_rate_integration():
    """示例：集成资金费率数据"""
    print("\n" + "=" * 60)
    print("示例4: 集成资金费率数据")
    print("=" * 60)
    
    loader = MLDatasetLoader()
    
    try:
        # 加载资金费率数据
        funding_df = loader.load_funding_rate_data()
        
        print(f"资金费率数据形状: {funding_df.shape}")
        print(f"时间范围: {funding_df.index.min()} 至 {funding_df.index.max()}")
        print(f"资金费率统计:")
        print(funding_df['fundingRate'].describe())
        
        # 加载主数据
        kline_data, metadata = loader.load_dataset("main_training_BTC_USDT_1h_train.json")
        df = loader.kline_data_to_dataframe(kline_data)
        
        # 合并数据（这里需要处理时间对齐）
        print(f"\n主数据时间范围: {df.index.min()} 至 {df.index.max()}")
        
        # 简单的时间对齐示例
        common_start = max(df.index.min(), funding_df.index.min())
        common_end = min(df.index.max(), funding_df.index.max())
        
        print(f"重叠时间范围: {common_start} 至 {common_end}")
        
        return funding_df
        
    except FileNotFoundError as e:
        print(f"资金费率数据不可用: {e}")
        return None


def example_dataset_summary():
    """示例：数据集总结"""
    print("\n" + "=" * 60)
    print("示例5: 数据集总结")
    print("=" * 60)
    
    ml_data_dir = "data/ml_datasets"
    
    if not os.path.exists(ml_data_dir):
        print("ML数据集目录不存在")
        return
    
    files = [f for f in os.listdir(ml_data_dir) if f.endswith('.json')]
    
    print(f"可用数据集文件: {len(files)}")
    
    for filename in sorted(files):
        filepath = os.path.join(ml_data_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            metadata = data['metadata']
            
            print(f"\n📄 {filename}")
            print(f"   描述: {metadata.get('description', 'N/A')}")
            print(f"   记录数: {metadata.get('record_count', 0):,}")
            print(f"   交易对: {metadata.get('symbol', 'N/A')}")
            print(f"   时间级别: {metadata.get('kl_type', 'N/A')}")
            
            if 'time_range' in metadata:
                time_range = metadata['time_range']
                print(f"   时间范围: {time_range.get('start', 'N/A')} 至 {time_range.get('end', 'N/A')}")
                print(f"   天数: {time_range.get('days', 'N/A')}")
            
            if 'features' in metadata:
                print(f"   特征数: {len(metadata['features'])}")
                
        except Exception as e:
            print(f"读取文件失败 {filename}: {e}")


def main():
    """主函数"""
    print("Chan.py机器学习数据集使用示例")
    print("=" * 80)
    
    # 运行示例
    try:
        df, metadata = example_load_training_data()
        features_df = example_prepare_features()
        labels_df = example_create_labels()
        funding_df = example_funding_rate_integration()
        example_dataset_summary()
        
        print("\n" + "=" * 80)
        print("✅ 所有示例运行完成！")
        print("=" * 80)
        
        print("\n💡 下一步建议:")
        print("1. 基于features_df进行特征选择和工程")
        print("2. 使用labels_df训练预测模型")
        print("3. 集成资金费率等外部数据")
        print("4. 实现chan.py缠论特征计算")
        print("5. 开发完整的机器学习pipeline")
        
    except Exception as e:
        print(f"❌ 示例运行失败: {e}")


if __name__ == "__main__":
    main() 