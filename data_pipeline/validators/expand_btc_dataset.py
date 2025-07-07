#!/usr/bin/env python3
"""
BTC现货和swap数据集扩展脚本
扩展到完整的三年历史数据，小时级别
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
import time
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from data_pipeline.collectors.collector import DataCollector
from data_pipeline.processors.data_processor import DataProcessor

class BTCDatasetExpander:
    """BTC数据集扩展器"""
    
    def __init__(self):
        self.collector = DataCollector()
        self.processor = DataProcessor()
        self.data_root = Path("data")
        
        # 目标时间范围：三年历史数据
        self.end_date = datetime.now(timezone.utc)
        self.start_date = self.end_date - timedelta(days=3*365)  # 3年
        
        logger.info(f"目标时间范围: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")
    
    def check_data_completeness(self, symbol: str, instrument_type: str) -> dict:
        """检查数据完整性"""
        logger.info(f"检查 {symbol} {instrument_type} 数据完整性...")
        
        # 构建数据路径
        if instrument_type == "spot":
            base_path = self.data_root / "okx" / "spot" / "BTC" / "USDT" / "1h"
        else:  # swap
            base_path = self.data_root / "okx" / "swap" / "BTC" / "USDT" / "1h"
        
        data_info = {
            'total_records': 0,
            'date_range': None,
            'missing_periods': [],
            'files_info': {}
        }
        
        all_data = []
        
        # 检查每年的数据文件
        for year in range(2022, 2026):  # 2022-2025
            file_path = base_path / f"{year}.parquet"
            
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    df.index = pd.to_datetime(df.index)
                    
                    data_info['files_info'][year] = {
                        'records': len(df),
                        'start_date': df.index.min(),
                        'end_date': df.index.max(),
                        'file_size': f"{file_path.stat().st_size / 1024:.1f} KB"
                    }
                    
                    all_data.append(df)
                    logger.info(f"  {year}: {len(df)} 条记录, {df.index.min()} 至 {df.index.max()}")
                    
                except Exception as e:
                    logger.error(f"  {year}: 读取失败 - {e}")
                    data_info['files_info'][year] = {'error': str(e)}
            else:
                logger.warning(f"  {year}: 文件不存在")
                data_info['files_info'][year] = {'missing': True}
        
        if all_data:
            # 合并所有数据
            combined_df = pd.concat(all_data).sort_index()
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            
            data_info['total_records'] = len(combined_df)
            data_info['date_range'] = (combined_df.index.min(), combined_df.index.max())
            
            # 检查缺失的时间段
            expected_range = pd.date_range(
                start=max(self.start_date, combined_df.index.min()),
                end=min(self.end_date, combined_df.index.max()),
                freq='H'
            )
            
            missing_timestamps = expected_range.difference(combined_df.index)
            if len(missing_timestamps) > 0:
                # 将连续的缺失时间段分组
                missing_periods = self._group_missing_periods(missing_timestamps)
                data_info['missing_periods'] = missing_periods
                logger.warning(f"  发现 {len(missing_timestamps)} 个缺失时间点，{len(missing_periods)} 个缺失时间段")
            else:
                logger.info(f"  数据完整，无缺失时间点")
        
        return data_info
    
    def _group_missing_periods(self, missing_timestamps):
        """将缺失的时间戳分组为连续的时间段"""
        if len(missing_timestamps) == 0:
            return []
        
        periods = []
        current_start = missing_timestamps[0]
        current_end = missing_timestamps[0]
        
        for i in range(1, len(missing_timestamps)):
            if missing_timestamps[i] == current_end + timedelta(hours=1):
                current_end = missing_timestamps[i]
            else:
                periods.append((current_start, current_end))
                current_start = missing_timestamps[i]
                current_end = missing_timestamps[i]
        
        periods.append((current_start, current_end))
        return periods
    
    def fetch_missing_data(self, symbol: str, instrument_type: str, missing_periods: list):
        """获取缺失的数据"""
        if not missing_periods:
            logger.info(f"无需获取缺失数据")
            return
        
        logger.info(f"开始获取 {len(missing_periods)} 个缺失时间段的数据...")
        
        for i, (start_time, end_time) in enumerate(missing_periods, 1):
            logger.info(f"获取缺失数据 {i}/{len(missing_periods)}: {start_time} 至 {end_time}")
            
            try:
                # 获取数据
                missing_data = self.collector.fetch_historical_data(
                    symbol=symbol,
                    timeframe="1h",
                    start_date=start_time,
                    end_date=end_time + timedelta(hours=1),  # 包含结束时间
                    instrument_type=instrument_type
                )
                
                if not missing_data.empty:
                    # 保存数据
                    self._save_missing_data(missing_data, symbol, instrument_type)
                    logger.info(f"  成功获取 {len(missing_data)} 条记录")
                else:
                    logger.warning(f"  未获取到数据")
                    
            except Exception as e:
                logger.error(f"  获取失败: {e}")
    
    def _save_missing_data(self, data: pd.DataFrame, symbol: str, instrument_type: str):
        """保存缺失的数据到对应的年份文件"""
        # 按年份分组数据
        data_by_year = data.groupby(data.index.year)
        
        for year, year_data in data_by_year:
            # 构建文件路径
            if instrument_type == "spot":
                file_path = self.data_root / "okx" / "spot" / "BTC" / "USDT" / "1h" / f"{year}.parquet"
            else:  # swap
                file_path = self.data_root / "okx" / "swap" / "BTC" / "USDT" / "1h" / f"{year}.parquet"
            
            # 确保目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.exists():
                # 读取现有数据
                existing_data = pd.read_parquet(file_path)
                existing_data.index = pd.to_datetime(existing_data.index)
                
                # 合并数据
                combined_data = pd.concat([existing_data, year_data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data.sort_index(inplace=True)
                
                # 保存合并后的数据
                combined_data.to_parquet(file_path)
                logger.info(f"  更新 {file_path}: 添加 {len(year_data)} 条记录")
            else:
                # 创建新文件
                year_data.to_parquet(file_path)
                logger.info(f"  创建 {file_path}: {len(year_data)} 条记录")
    
    def extend_to_full_range(self, symbol: str, instrument_type: str):
        """扩展数据到完整的三年范围"""
        logger.info(f"扩展 {symbol} {instrument_type} 数据到完整三年范围...")
        
        # 获取当前数据的时间范围
        data_info = self.check_data_completeness(symbol, instrument_type)
        
        if data_info['date_range']:
            current_start, current_end = data_info['date_range']
            
            # 检查是否需要向前扩展
            if current_start > self.start_date:
                logger.info(f"向前扩展数据: {self.start_date} 至 {current_start}")
                try:
                    early_data = self.collector.fetch_historical_data(
                        symbol=symbol,
                        timeframe="1h",
                        start_date=self.start_date,
                        end_date=current_start,
                        instrument_type=instrument_type
                    )
                    
                    if not early_data.empty:
                        self._save_missing_data(early_data, symbol, instrument_type)
                        logger.info(f"  成功扩展 {len(early_data)} 条早期记录")
                except Exception as e:
                    logger.error(f"  向前扩展失败: {e}")
            
            # 检查是否需要向后扩展
            if current_end < self.end_date:
                logger.info(f"向后扩展数据: {current_end} 至 {self.end_date}")
                try:
                    recent_data = self.collector.fetch_historical_data(
                        symbol=symbol,
                        timeframe="1h",
                        start_date=current_end,
                        end_date=self.end_date,
                        instrument_type=instrument_type
                    )
                    
                    if not recent_data.empty:
                        self._save_missing_data(recent_data, symbol, instrument_type)
                        logger.info(f"  成功扩展 {len(recent_data)} 条最新记录")
                except Exception as e:
                    logger.error(f"  向后扩展失败: {e}")
        else:
            # 没有现有数据，获取完整的三年数据
            logger.info(f"获取完整的三年历史数据...")
            try:
                full_data = self.collector.fetch_historical_data(
                    symbol=symbol,
                    timeframe="1h",
                    start_date=self.start_date,
                    end_date=self.end_date,
                    instrument_type=instrument_type
                )
                
                if not full_data.empty:
                    self._save_missing_data(full_data, symbol, instrument_type)
                    logger.info(f"  成功获取 {len(full_data)} 条完整历史记录")
            except Exception as e:
                logger.error(f"  获取完整数据失败: {e}")
    
    def process_and_generate_indicators(self, symbol: str):
        """处理数据并生成技术指标"""
        logger.info(f"处理 {symbol} 数据并生成技术指标...")
        
        try:
            # 使用数据处理器处理数据
            processed_data = self.processor.process_and_save(
                symbol=symbol,
                output_dir="data/processed"
            )
            
            if processed_data:
                for timeframe, df in processed_data.items():
                    logger.info(f"  {timeframe}: {len(df)} 条记录，{len(df.columns)} 个指标")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"处理数据失败: {e}")
            return None
    
    def generate_summary_report(self):
        """生成数据扩展总结报告"""
        logger.info("生成数据扩展总结报告...")
        
        report = {
            'timestamp': datetime.now(),
            'target_range': (self.start_date, self.end_date),
            'symbols': {}
        }
        
        # 检查BTC现货和swap数据
        for symbol, instrument_type in [("BTC/USDT", "spot"), ("BTC/USDT", "swap")]:
            data_info = self.check_data_completeness(symbol, instrument_type)
            report['symbols'][f"{symbol}_{instrument_type}"] = data_info
        
        # 保存报告
        report_path = Path("data") / "btc_dataset_expansion_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BTC数据集扩展报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {report['timestamp']}\n")
            f.write(f"目标时间范围: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}\n\n")
            
            for symbol_key, data_info in report['symbols'].items():
                f.write(f"{symbol_key}:\n")
                f.write(f"  总记录数: {data_info['total_records']}\n")
                
                if data_info['date_range']:
                    start, end = data_info['date_range']
                    f.write(f"  数据范围: {start} 至 {end}\n")
                    f.write(f"  覆盖天数: {(end - start).days} 天\n")
                
                f.write(f"  缺失时间段: {len(data_info['missing_periods'])} 个\n")
                
                f.write("  文件信息:\n")
                for year, file_info in data_info['files_info'].items():
                    if 'error' in file_info:
                        f.write(f"    {year}: 错误 - {file_info['error']}\n")
                    elif 'missing' in file_info:
                        f.write(f"    {year}: 文件缺失\n")
                    else:
                        f.write(f"    {year}: {file_info['records']} 条记录 ({file_info['file_size']})\n")
                
                f.write("\n")
        
        logger.info(f"报告已保存至: {report_path}")
        return report

def main():
    """主函数"""
    logger.info("开始BTC数据集扩展任务...")
    
    expander = BTCDatasetExpander()
    
    # 要处理的数据类型
    datasets = [
        ("BTC/USDT", "spot"),
        ("BTC/USDT", "swap")
    ]
    
    for symbol, instrument_type in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"处理 {symbol} {instrument_type} 数据")
        logger.info(f"{'='*60}")
        
        # 1. 检查数据完整性
        data_info = expander.check_data_completeness(symbol, instrument_type)
        
        # 2. 获取缺失数据
        if data_info['missing_periods']:
            expander.fetch_missing_data(symbol, instrument_type, data_info['missing_periods'])
        
        # 3. 扩展到完整范围
        expander.extend_to_full_range(symbol, instrument_type)
        
        # 4. 重新检查完整性
        logger.info("重新检查数据完整性...")
        final_data_info = expander.check_data_completeness(symbol, instrument_type)
        
        logger.info(f"最终数据统计:")
        logger.info(f"  总记录数: {final_data_info['total_records']}")
        if final_data_info['date_range']:
            start, end = final_data_info['date_range']
            logger.info(f"  数据范围: {start} 至 {end}")
            logger.info(f"  覆盖天数: {(end - start).days} 天")
        logger.info(f"  缺失时间段: {len(final_data_info['missing_periods'])} 个")
    
    # 5. 处理数据并生成技术指标
    logger.info(f"\n{'='*60}")
    logger.info("处理数据并生成技术指标")
    logger.info(f"{'='*60}")
    
    processed_data = expander.process_and_generate_indicators("BTC/USDT")
    
    # 6. 生成总结报告
    logger.info(f"\n{'='*60}")
    logger.info("生成总结报告")
    logger.info(f"{'='*60}")
    
    report = expander.generate_summary_report()
    
    logger.info("BTC数据集扩展任务完成！")

if __name__ == "__main__":
    main() 