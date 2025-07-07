#!/usr/bin/env python3
"""
验证三年BTC数据集质量
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_3y_dataset():
    """验证三年数据集的质量"""
    
    logger.info("开始验证三年BTC数据集质量...")
    
    processed_path = Path("data/processed")
    
    # 要验证的文件
    files_to_validate = [
        "BTC_USDT_spot_1h_3y.parquet",
        "BTC_USDT_swap_1h_3y.parquet", 
        "BTC_USDT_combined_1h_3y.parquet",
        "BTC_USDT_spot_1d_3y.parquet",
        "BTC_USDT_swap_1d_3y.parquet",
        "BTC_USDT_combined_1d_3y.parquet"
    ]
    
    validation_results = {}
    
    for filename in files_to_validate:
        file_path = processed_path / filename
        
        if not file_path.exists():
            logger.error(f"文件不存在: {filename}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"验证文件: {filename}")
        logger.info(f"{'='*60}")
        
        try:
            # 读取数据
            df = pd.read_parquet(file_path)
            
            # 基础信息
            logger.info(f"📊 基础信息:")
            logger.info(f"  记录数: {len(df):,}")
            logger.info(f"  字段数: {len(df.columns)}")
            logger.info(f"  文件大小: {file_path.stat().st_size / 1024 / 1024:.1f} MB")
            
            # 时间范围
            if hasattr(df.index, 'min'):
                start_time = df.index.min()
                end_time = df.index.max()
                duration = end_time - start_time
                
                logger.info(f"⏰ 时间信息:")
                logger.info(f"  开始时间: {start_time}")
                logger.info(f"  结束时间: {end_time}")
                logger.info(f"  覆盖天数: {duration.days} 天")
                logger.info(f"  覆盖年数: {duration.days / 365.25:.2f} 年")
            
            # 数据完整性检查
            logger.info(f"🔍 数据完整性:")
            
            # 检查缺失值
            missing_values = df.isnull().sum()
            total_missing = missing_values.sum()
            logger.info(f"  总缺失值: {total_missing}")
            
            if total_missing > 0:
                logger.info(f"  缺失值分布:")
                for col, missing in missing_values.items():
                    if missing > 0:
                        logger.info(f"    {col}: {missing} ({missing/len(df)*100:.2f}%)")
            
            # 检查重复索引
            if hasattr(df.index, 'duplicated'):
                duplicated_count = df.index.duplicated().sum()
                logger.info(f"  重复时间戳: {duplicated_count}")
            
            # 价格数据质量检查
            if 'close' in df.columns:
                logger.info(f"💰 价格数据质量:")
                logger.info(f"  价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                logger.info(f"  平均价格: ${df['close'].mean():.2f}")
                logger.info(f"  价格标准差: ${df['close'].std():.2f}")
                
                # 检查异常价格变动
                if len(df) > 1:
                    price_changes = df['close'].pct_change().abs()
                    extreme_changes = price_changes > 0.1  # 10%以上变动
                    logger.info(f"  极端价格变动(>10%): {extreme_changes.sum()} 次")
                    
                    if extreme_changes.sum() > 0:
                        max_change = price_changes.max()
                        max_change_time = price_changes.idxmax()
                        logger.info(f"  最大价格变动: {max_change*100:.2f}% 在 {max_change_time}")
            
            # 成交量数据质量检查
            if 'volume' in df.columns:
                logger.info(f"📈 成交量数据质量:")
                logger.info(f"  成交量范围: {df['volume'].min():.2f} - {df['volume'].max():.2f}")
                logger.info(f"  平均成交量: {df['volume'].mean():.2f}")
                logger.info(f"  零成交量记录: {(df['volume'] == 0).sum()}")
            
            # 技术指标质量检查
            technical_indicators = ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14', 'MACD', 'ADX_14']
            available_indicators = [col for col in technical_indicators if col in df.columns]
            
            if available_indicators:
                logger.info(f"📊 技术指标质量:")
                logger.info(f"  可用指标: {len(available_indicators)} 个")
                
                for indicator in available_indicators[:5]:  # 只显示前5个
                    if indicator in df.columns:
                        indicator_data = df[indicator].dropna()
                        if len(indicator_data) > 0:
                            logger.info(f"  {indicator}: 范围 {indicator_data.min():.2f} - {indicator_data.max():.2f}")
            
            # 特殊字段检查（合并数据集）
            if 'swap_volume' in df.columns:
                logger.info(f"🔄 合并数据特殊字段:")
                logger.info(f"  swap成交量范围: {df['swap_volume'].min():.2f} - {df['swap_volume'].max():.2f}")
                
                if 'volume_ratio' in df.columns:
                    ratio_data = df['volume_ratio'].dropna()
                    if len(ratio_data) > 0:
                        logger.info(f"  成交量比率范围: {ratio_data.min():.2f} - {ratio_data.max():.2f}")
            
            # 数据连续性检查（仅对小时数据）
            if '1h' in filename and hasattr(df.index, 'freq'):
                logger.info(f"⏱️  数据连续性:")
                
                # 检查时间间隔
                if len(df) > 1:
                    time_diffs = df.index.to_series().diff().dropna()
                    expected_interval = pd.Timedelta(hours=1)
                    
                    normal_intervals = (time_diffs == expected_interval).sum()
                    total_intervals = len(time_diffs)
                    continuity_rate = normal_intervals / total_intervals * 100
                    
                    logger.info(f"  正常间隔: {normal_intervals}/{total_intervals} ({continuity_rate:.1f}%)")
                    
                    if continuity_rate < 99:
                        abnormal_intervals = time_diffs[time_diffs != expected_interval]
                        logger.info(f"  异常间隔数量: {len(abnormal_intervals)}")
            
            # 保存验证结果
            validation_results[filename] = {
                'records': len(df),
                'columns': len(df.columns),
                'missing_values': total_missing,
                'file_size_mb': file_path.stat().st_size / 1024 / 1024,
                'status': 'PASS' if total_missing < len(df) * 0.01 else 'WARNING'  # 1%阈值
            }
            
            logger.info(f"✅ 验证状态: {validation_results[filename]['status']}")
            
        except Exception as e:
            logger.error(f"❌ 验证失败: {e}")
            validation_results[filename] = {
                'status': 'FAILED',
                'error': str(e)
            }
    
    # 生成验证总结
    logger.info(f"\n{'='*60}")
    logger.info("验证总结")
    logger.info(f"{'='*60}")
    
    passed_files = sum(1 for result in validation_results.values() if result.get('status') == 'PASS')
    warning_files = sum(1 for result in validation_results.values() if result.get('status') == 'WARNING')
    failed_files = sum(1 for result in validation_results.values() if result.get('status') == 'FAILED')
    
    logger.info(f"📊 验证统计:")
    logger.info(f"  通过: {passed_files} 个文件")
    logger.info(f"  警告: {warning_files} 个文件")
    logger.info(f"  失败: {failed_files} 个文件")
    
    # 计算总数据量
    total_records = sum(result.get('records', 0) for result in validation_results.values() if 'records' in result)
    total_size_mb = sum(result.get('file_size_mb', 0) for result in validation_results.values() if 'file_size_mb' in result)
    
    logger.info(f"📈 数据统计:")
    logger.info(f"  总记录数: {total_records:,}")
    logger.info(f"  总文件大小: {total_size_mb:.1f} MB")
    
    # 保存验证报告
    report_path = Path("data") / "btc_3y_validation_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BTC三年数据集验证报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"验证时间: {datetime.now()}\n\n")
        
        f.write("验证结果:\n")
        for filename, result in validation_results.items():
            f.write(f"\n{filename}:\n")
            f.write(f"  状态: {result.get('status', 'UNKNOWN')}\n")
            
            if 'records' in result:
                f.write(f"  记录数: {result['records']:,}\n")
                f.write(f"  字段数: {result['columns']}\n")
                f.write(f"  缺失值: {result['missing_values']}\n")
                f.write(f"  文件大小: {result['file_size_mb']:.1f} MB\n")
            
            if 'error' in result:
                f.write(f"  错误: {result['error']}\n")
        
        f.write(f"\n总结:\n")
        f.write(f"  通过: {passed_files} 个文件\n")
        f.write(f"  警告: {warning_files} 个文件\n")
        f.write(f"  失败: {failed_files} 个文件\n")
        f.write(f"  总记录数: {total_records:,}\n")
        f.write(f"  总文件大小: {total_size_mb:.1f} MB\n")
    
    logger.info(f"📄 验证报告已保存至: {report_path}")
    
    return validation_results

def main():
    """主函数"""
    validation_results = validate_3y_dataset()
    
    # 检查是否所有文件都通过验证
    all_passed = all(result.get('status') == 'PASS' for result in validation_results.values())
    
    if all_passed:
        logger.info("\n🎉 所有数据集验证通过！三年BTC数据集已准备就绪。")
    else:
        logger.warning("\n⚠️  部分数据集存在问题，请检查验证报告。")

if __name__ == "__main__":
    main() 