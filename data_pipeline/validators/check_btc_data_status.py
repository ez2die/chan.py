#!/usr/bin/env python3
"""
BTC数据状态检查脚本
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_btc_data_status():
    """检查BTC现货和swap数据状态"""
    
    data_root = Path("data")
    
    # 目标时间范围：三年历史数据
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=3*365)  # 3年
    
    logger.info(f"检查时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    
    # 检查现货和swap数据
    for instrument_type in ["spot", "swap"]:
        logger.info(f"\n{'='*50}")
        logger.info(f"检查 BTC/USDT {instrument_type} 数据")
        logger.info(f"{'='*50}")
        
        base_path = data_root / "okx" / instrument_type / "BTC" / "USDT" / "1h"
        
        total_records = 0
        all_data = []
        
        # 检查每年的数据文件
        for year in range(2022, 2026):  # 2022-2025
            file_path = base_path / f"{year}.parquet"
            
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    df.index = pd.to_datetime(df.index)
                    
                    file_size = file_path.stat().st_size / 1024  # KB
                    
                    logger.info(f"  {year}: {len(df)} 条记录, {file_size:.1f} KB")
                    logger.info(f"    时间范围: {df.index.min()} 至 {df.index.max()}")
                    
                    total_records += len(df)
                    all_data.append(df)
                    
                except Exception as e:
                    logger.error(f"  {year}: 读取失败 - {e}")
            else:
                logger.warning(f"  {year}: 文件不存在")
        
        if all_data:
            # 合并所有数据
            combined_df = pd.concat(all_data).sort_index()
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            
            logger.info(f"\n总计:")
            logger.info(f"  总记录数: {len(combined_df)}")
            logger.info(f"  数据范围: {combined_df.index.min()} 至 {combined_df.index.max()}")
            logger.info(f"  覆盖天数: {(combined_df.index.max() - combined_df.index.min()).days} 天")
            
            # 检查数据完整性
            expected_hours = (combined_df.index.max() - combined_df.index.min()).total_seconds() / 3600
            completeness = len(combined_df) / expected_hours * 100
            logger.info(f"  数据完整性: {completeness:.1f}%")
            
            # 检查最近的数据
            hours_since_last = (datetime.now(timezone.utc) - combined_df.index.max()).total_seconds() / 3600
            logger.info(f"  最新数据距今: {hours_since_last:.1f} 小时")
            
            # 检查数据质量
            logger.info(f"\n数据质量检查:")
            logger.info(f"  价格范围: ${combined_df['close'].min():.2f} - ${combined_df['close'].max():.2f}")
            logger.info(f"  平均成交量: {combined_df['volume'].mean():.2f}")
            logger.info(f"  缺失值: {combined_df.isnull().sum().sum()} 个")
        else:
            logger.warning(f"  没有找到任何数据文件")

def check_processed_data():
    """检查处理后的数据"""
    logger.info(f"\n{'='*50}")
    logger.info("检查处理后的数据")
    logger.info(f"{'='*50}")
    
    processed_path = Path("data/processed")
    
    if processed_path.exists():
        for file_path in processed_path.glob("*.parquet"):
            try:
                df = pd.read_parquet(file_path)
                logger.info(f"  {file_path.name}: {len(df)} 条记录, {len(df.columns)} 个字段")
                
                if hasattr(df.index, 'min'):
                    logger.info(f"    时间范围: {df.index.min()} 至 {df.index.max()}")
                
            except Exception as e:
                logger.error(f"  {file_path.name}: 读取失败 - {e}")
    else:
        logger.warning("  processed 目录不存在")

def main():
    """主函数"""
    logger.info("开始检查BTC数据状态...")
    
    check_btc_data_status()
    check_processed_data()
    
    logger.info("\n数据状态检查完成！")

if __name__ == "__main__":
    main() 