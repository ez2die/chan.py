import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# 添加项目根目录
sys.path.append(str(Path(__file__).parent.parent))
from tools.ml_dataset_example import MLDatasetLoader

class ChanDatasetManager:
    """Chan.py数据集管理器"""
    
    def __init__(self, ml_data_dir: str = "data/ml_datasets"):
        self.ml_data_dir = ml_data_dir
        self.loader = MLDatasetLoader(ml_data_dir)
        self.datasets_cache = {}
        
    def load_training_dataset(self, dataset_name: str = "main_training") -> Tuple[pd.DataFrame, Dict]:
        """加载训练数据集"""
        cache_key = f"{dataset_name}_train"
        if cache_key in self.datasets_cache:
            return self.datasets_cache[cache_key]
            
        # 加载训练集
        train_file = f"{dataset_name}_BTC_USDT_1h_train.json"
        kline_data, metadata = self.loader.load_dataset(train_file)
        
        # 转换为DataFrame
        df = self.loader.kline_data_to_dataframe(kline_data)
        
        self.datasets_cache[cache_key] = (df, metadata)
        return df, metadata
        
    def load_validation_dataset(self, dataset_name: str = "main_training") -> Tuple[pd.DataFrame, Dict]:
        """加载验证数据集"""
        cache_key = f"{dataset_name}_val"
        if cache_key in self.datasets_cache:
            return self.datasets_cache[cache_key]
            
        val_file = f"{dataset_name}_BTC_USDT_1h_val.json"
        kline_data, metadata = self.loader.load_dataset(val_file)
        df = self.loader.kline_data_to_dataframe(kline_data)
        
        self.datasets_cache[cache_key] = (df, metadata)
        return df, metadata
        
    def load_funding_rate_data(self) -> pd.DataFrame:
        """加载资金费率数据"""
        return self.loader.load_funding_rate_data()
        
    def prepare_chan_format_data(self, df: pd.DataFrame) -> List:
        """准备Chan.py格式的数据"""
        chan_data = []
        for timestamp, row in df.iterrows():
            kl_dict = {
                'time': timestamp.strftime('%Y/%m/%d %H:%M'),
                'open': row['open'],
                'high': row['high'], 
                'low': row['low'],
                'close': row['close'],
                'volume': row.get('volume', 0)
            }
            chan_data.append(kl_dict)
        return chan_data
        
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        info = {
            'available_datasets': [],
            'total_records': 0,
            'dataset_details': {}
        }
        
        if not os.path.exists(self.ml_data_dir):
            return info
            
        files = [f for f in os.listdir(self.ml_data_dir) if f.endswith('.json')]
        
        for filename in files:
            try:
                filepath = os.path.join(self.ml_data_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                metadata = data['metadata']
                record_count = metadata.get('record_count', 0)
                
                info['available_datasets'].append(filename)
                info['total_records'] += record_count
                info['dataset_details'][filename] = {
                    'records': record_count,
                    'description': metadata.get('description', ''),
                    'symbol': metadata.get('symbol', ''),
                    'kl_type': metadata.get('kl_type', ''),
                    'time_range': metadata.get('time_range', {})
                }
                
            except Exception as e:
                print(f"读取数据集失败 {filename}: {e}")
                
        return info
        
    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """验证数据集质量"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # 基础统计
        validation_result['stats'] = {
            'total_records': len(df),
            'date_range': {
                'start': df.index.min().strftime('%Y-%m-%d %H:%M'),
                'end': df.index.max().strftime('%Y-%m-%d %H:%M')
            },
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"缺少必需列: {missing_cols}")
            
        # 检查价格逻辑
        if 'high' in df.columns and 'low' in df.columns:
            invalid_hl = df[df['high'] < df['low']]
            if len(invalid_hl) > 0:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"发现 {len(invalid_hl)} 条高价小于低价的记录")
                
        # 检查时间连续性
        time_gaps = df.index.to_series().diff()
        expected_interval = time_gaps.mode().iloc[0] if len(time_gaps.mode()) > 0 else pd.Timedelta(hours=1)
        large_gaps = time_gaps[time_gaps > expected_interval * 2]
        if len(large_gaps) > 0:
            validation_result['warnings'].append(f"发现 {len(large_gaps)} 个时间间隔异常")
            
        return validation_result 