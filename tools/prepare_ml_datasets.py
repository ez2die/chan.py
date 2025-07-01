#!/usr/bin/env python3
"""
机器学习数据集准备脚本
基于chan.py框架准备训练集和验证集
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from tools.data_converter import ChanDataConverter


class MLDatasetPreparer:
    """机器学习数据集准备器"""
    
    def __init__(self):
        self.converter = ChanDataConverter()
        self.ml_data_dir = "data/ml_datasets"
        
        # 关键数据文件配置
        self.key_datasets = {
            # 主训练集 - 3年综合数据
            "main_training": {
                "file": "data/processed/BTC_USDT_combined_1h_3y.parquet",
                "symbol": "BTC_USDT",
                "kl_type": "1h",
                "description": "主训练集：3年1小时综合数据，包含现货+合约特征",
                "priority": 1
            },
            
            # 验证集 - 最新处理数据
            "validation": {
                "file": "data/processed/BTC_USDT_1h_processed.parquet", 
                "symbol": "BTC_USDT",
                "kl_type": "1h",
                "description": "验证集：2024-2025年最新数据",
                "priority": 1
            },
            
            # 日线数据 - 长期趋势
            "daily_training": {
                "file": "data/processed/BTC_USDT_combined_1d_3y.parquet",
                "symbol": "BTC_USDT", 
                "kl_type": "1d",
                "description": "日线训练集：3年日线数据，长期趋势分析",
                "priority": 2
            },
            
            # 现货数据 - 对比分析
            "spot_training": {
                "file": "data/processed/BTC_USDT_spot_1h_3y.parquet",
                "symbol": "BTC_USDT_SPOT",
                "kl_type": "1h", 
                "description": "现货训练集：3年现货1小时数据",
                "priority": 3
            },
            
            # 合约数据 - 对比分析
            "swap_training": {
                "file": "data/processed/BTC_USDT_swap_1h_3y.parquet",
                "symbol": "BTC_USDT_SWAP",
                "kl_type": "1h",
                "description": "合约训练集：3年合约1小时数据", 
                "priority": 3
            }
        }
    
    def prepare_dataset(self, dataset_key: str) -> bool:
        """准备单个数据集"""
        if dataset_key not in self.key_datasets:
            print(f"未知的数据集: {dataset_key}")
            return False
        
        config = self.key_datasets[dataset_key]
        input_file = config["file"]
        
        if not os.path.exists(input_file):
            print(f"文件不存在: {input_file}")
            return False
        
        print(f"\n{'='*60}")
        print(f"准备数据集: {dataset_key}")
        print(f"描述: {config['description']}")
        print(f"{'='*60}")
        
        try:
            # 加载数据
            print("正在加载数据...")
            df = self.converter.load_data(input_file)
            print(f"数据形状: {df.shape}")
            print(f"时间范围: {df.index.min()} 至 {df.index.max()}")
            print(f"特征列: {df.columns.tolist()}")
            
            # 转换为chan.py格式
            print("正在转换为chan.py格式...")
            chan_data = self.converter.convert_to_chan_format(
                df, config["symbol"], config["kl_type"]
            )
            
            # 保存数据
            os.makedirs(self.ml_data_dir, exist_ok=True)
            output_file = os.path.join(
                self.ml_data_dir, 
                f"{dataset_key}_{config['symbol']}_{config['kl_type']}_chan.json"
            )
            
            metadata = {
                'dataset_key': dataset_key,
                'symbol': config['symbol'],
                'kl_type': config['kl_type'],
                'description': config['description'],
                'priority': config['priority'],
                'source_file': input_file,
                'convert_time': datetime.now().isoformat(),
                'record_count': len(chan_data),
                'features': df.columns.tolist(),
                'time_range': {
                    'start': str(df.index.min()),
                    'end': str(df.index.max()),
                    'days': (df.index.max() - df.index.min()).days
                }
            }
            
            self.converter.save_chan_data(chan_data, output_file, metadata)
            
            print(f"✅ 数据集准备完成: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ 数据集准备失败: {e}")
            return False
    
    def prepare_funding_rate_data(self) -> bool:
        """准备资金费率数据"""
        print(f"\n{'='*60}")
        print("准备资金费率数据")
        print(f"{'='*60}")
        
        funding_file = "data/derivatives_history/BTC_USDT_SWAP_funding_rate_3y.csv"
        
        if not os.path.exists(funding_file):
            print(f"资金费率文件不存在: {funding_file}")
            return False
        
        try:
            # 加载资金费率数据
            df = pd.read_csv(funding_file)
            print(f"资金费率数据形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            
            # 处理时间戳
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"时间范围: {df.index.min()} 至 {df.index.max()}")
            print(f"资金费率范围: {df['fundingRate'].min():.6f} 至 {df['fundingRate'].max():.6f}")
            
            # 保存处理后的资金费率数据
            output_file = os.path.join(self.ml_data_dir, "funding_rate_processed.csv")
            df.to_csv(output_file)
            
            # 也保存为JSON格式便于集成
            funding_data = []
            for timestamp, row in df.iterrows():
                funding_data.append({
                    'timestamp': timestamp.isoformat(),
                    'funding_rate': float(row['fundingRate']),
                    'symbol': row['symbol']
                })
            
            import json
            json_output = os.path.join(self.ml_data_dir, "funding_rate_data.json")
            with open(json_output, 'w') as f:
                json.dump({
                    'metadata': {
                        'description': '资金费率数据',
                        'record_count': len(funding_data),
                        'time_range': {
                            'start': str(df.index.min()),
                            'end': str(df.index.max())
                        }
                    },
                    'funding_data': funding_data
                }, f, indent=2)
            
            print(f"✅ 资金费率数据准备完成:")
            print(f"   CSV格式: {output_file}")
            print(f"   JSON格式: {json_output}")
            return True
            
        except Exception as e:
            print(f"❌ 资金费率数据准备失败: {e}")
            return False
    
    def create_train_val_split(self, dataset_key: str, split_ratio: float = 0.8) -> bool:
        """创建训练/验证集分割"""
        print(f"\n{'='*60}")
        print(f"创建训练/验证集分割: {dataset_key}")
        print(f"分割比例: {split_ratio:.1%} 训练 / {1-split_ratio:.1%} 验证")
        print(f"{'='*60}")
        
        # 查找数据集文件
        dataset_file = None
        for filename in os.listdir(self.ml_data_dir):
            if filename.startswith(dataset_key) and filename.endswith('_chan.json'):
                dataset_file = os.path.join(self.ml_data_dir, filename)
                break
        
        if not dataset_file:
            print(f"未找到数据集文件: {dataset_key}")
            return False
        
        try:
            # 加载数据
            import json
            with open(dataset_file, 'r') as f:
                data = json.load(f)
            
            kline_data = data['kline_data']
            metadata = data['metadata']
            
            print(f"总记录数: {len(kline_data)}")
            
            # 按时间顺序分割
            split_point = int(len(kline_data) * split_ratio)
            train_data = kline_data[:split_point]
            val_data = kline_data[split_point:]
            
            print(f"训练集: {len(train_data)} 条记录")
            print(f"验证集: {len(val_data)} 条记录")
            
            # 保存训练集
            train_metadata = metadata.copy()
            train_metadata.update({
                'split_type': 'train',
                'split_ratio': split_ratio,
                'record_count': len(train_data),
                'original_file': dataset_file
            })
            
            train_file = dataset_file.replace('_chan.json', '_train.json')
            with open(train_file, 'w') as f:
                json.dump({
                    'metadata': train_metadata,
                    'kline_data': train_data
                }, f, indent=2)
            
            # 保存验证集
            val_metadata = metadata.copy()
            val_metadata.update({
                'split_type': 'validation',
                'split_ratio': 1 - split_ratio,
                'record_count': len(val_data),
                'original_file': dataset_file
            })
            
            val_file = dataset_file.replace('_chan.json', '_val.json')
            with open(val_file, 'w') as f:
                json.dump({
                    'metadata': val_metadata,
                    'kline_data': val_data
                }, f, indent=2)
            
            print(f"✅ 训练/验证集分割完成:")
            print(f"   训练集: {train_file}")
            print(f"   验证集: {val_file}")
            return True
            
        except Exception as e:
            print(f"❌ 训练/验证集分割失败: {e}")
            return False
    
    def prepare_all_datasets(self, priority_level: int = 1) -> Dict[str, bool]:
        """准备所有指定优先级的数据集"""
        print(f"\n🚀 开始准备机器学习数据集 (优先级 ≤ {priority_level})")
        print(f"{'='*80}")
        
        results = {}
        
        # 创建输出目录
        os.makedirs(self.ml_data_dir, exist_ok=True)
        
        # 准备资金费率数据
        results['funding_rate'] = self.prepare_funding_rate_data()
        
        # 准备K线数据集
        for dataset_key, config in self.key_datasets.items():
            if config['priority'] <= priority_level:
                results[dataset_key] = self.prepare_dataset(dataset_key)
                
                # 如果是主要数据集，创建训练/验证分割
                if results[dataset_key] and config['priority'] == 1:
                    split_result = self.create_train_val_split(dataset_key)
                    results[f"{dataset_key}_split"] = split_result
        
        return results
    
    def generate_dataset_summary(self):
        """生成数据集准备总结"""
        if not os.path.exists(self.ml_data_dir):
            print("ML数据集目录不存在")
            return
        
        print(f"\n📊 机器学习数据集准备总结")
        print(f"{'='*80}")
        
        files = [f for f in os.listdir(self.ml_data_dir) if f.endswith('.json')]
        
        total_records = 0
        datasets_info = []
        
        for filename in sorted(files):
            filepath = os.path.join(self.ml_data_dir, filename)
            try:
                import json
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                metadata = data['metadata']
                record_count = metadata.get('record_count', 0)
                total_records += record_count
                
                datasets_info.append({
                    'file': filename,
                    'records': record_count,
                    'description': metadata.get('description', 'N/A'),
                    'symbol': metadata.get('symbol', 'N/A'),
                    'kl_type': metadata.get('kl_type', 'N/A')
                })
                
            except Exception as e:
                print(f"读取文件失败 {filename}: {e}")
        
        print(f"总数据集文件: {len(files)}")
        print(f"总记录数: {total_records:,}")
        print(f"\n数据集详情:")
        print("-" * 80)
        
        for info in datasets_info:
            print(f"📄 {info['file']}")
            print(f"   记录数: {info['records']:,}")
            print(f"   描述: {info['description']}")
            print(f"   交易对: {info['symbol']} | 时间级别: {info['kl_type']}")
            print()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='准备机器学习数据集')
    parser.add_argument('--priority', type=int, default=1, 
                       help='数据集优先级 (1=最高优先级)')
    parser.add_argument('--dataset', type=str, 
                       help='准备特定数据集')
    
    args = parser.parse_args()
    
    preparer = MLDatasetPreparer()
    
    if args.dataset:
        # 准备特定数据集
        success = preparer.prepare_dataset(args.dataset)
        if success:
            preparer.create_train_val_split(args.dataset)
    else:
        # 准备所有数据集
        results = preparer.prepare_all_datasets(args.priority)
        
        print(f"\n📋 准备结果:")
        print("-" * 40)
        for dataset, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            print(f"{dataset}: {status}")
    
    # 生成总结
    preparer.generate_dataset_summary()


if __name__ == "__main__":
    main() 