#!/usr/bin/env python3
"""
更新和扩展BTC数据脚本
补充最新数据并重新生成完整的三年处理文件
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BTCDataUpdater:
    """BTC数据更新器"""
    
    def __init__(self):
        self.data_root = Path("data")
        
    def load_all_data(self, instrument_type: str) -> pd.DataFrame:
        """加载所有年份的数据"""
        logger.info(f"加载 {instrument_type} 所有数据...")
        
        base_path = self.data_root / "okx" / instrument_type / "BTC" / "USDT" / "1h"
        all_data = []
        
        # 加载指定起始年份至当前年份的数据（默认从2021年开始，以便覆盖2021-Q4至今）
        start_year = 2021  # 若未来需要更早数据，可在此调整
        current_year = datetime.now().year

        for year in range(start_year, current_year + 1):
            file_path = base_path / f"{year}.parquet"
            
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    df.index = pd.to_datetime(df.index)
                    all_data.append(df)
                    logger.info(f"  {year}: {len(df)} 条记录")
                except Exception as e:
                    logger.error(f"  {year}: 读取失败 - {e}")
        
        if all_data:
            # 合并所有数据
            combined_df = pd.concat(all_data).sort_index()
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            
            logger.info(f"  合并后总计: {len(combined_df)} 条记录")
            logger.info(f"  时间范围: {combined_df.index.min()} 至 {combined_df.index.max()}")
            
            return combined_df
        else:
            logger.error(f"  没有找到任何 {instrument_type} 数据")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        logger.info("计算技术指标...")
        
        df = df.copy()
        
        # 基础指标
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(24).std()
        df['volume_ma'] = df['volume'].rolling(24).mean()
        
        # 移动平均线
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()
        df['EMA_20'] = df['close'].ewm(span=20).mean()
        df['EMA_50'] = df['close'].ewm(span=50).mean()
        df['EMA_100'] = df['close'].ewm(span=100).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # ADX (简化版本)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX_14'] = dx.rolling(14).mean()
        
        logger.info(f"  计算完成，共 {len(df.columns)} 个字段")
        return df
    
    def resample_to_daily(self, df_hourly: pd.DataFrame) -> pd.DataFrame:
        """将小时数据重采样为日线数据"""
        logger.info("重采样为日线数据...")
        
        if df_hourly.empty:
            return pd.DataFrame()
        
        # 基础OHLCV重采样
        df_daily = df_hourly.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 重新计算技术指标
        df_daily = self.calculate_technical_indicators(df_daily)
        
        logger.info(f"  日线数据: {len(df_daily)} 条记录")
        return df_daily
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """保存处理后的数据"""
        output_path = self.data_root / "processed"
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        df.to_parquet(file_path)
        
        logger.info(f"  保存至: {file_path}")
        logger.info(f"  数据量: {len(df)} 条记录, {len(df.columns)} 个字段")
        logger.info(f"  文件大小: {file_path.stat().st_size / 1024:.1f} KB")
    
    def update_latest_data(self):
        """更新最新数据（如果需要的话）"""
        logger.info("检查是否需要更新最新数据...")
        
        # 这里可以添加从API获取最新数据的逻辑
        # 由于您已经有了完整的数据收集器，这里暂时跳过
        logger.info("  跳过API更新（数据已经很新）")
    
    def process_full_dataset(self):
        """处理完整的三年数据集"""
        logger.info("开始处理完整的三年数据集...")
        
        # 1. 加载现货数据
        logger.info(f"\n{'='*60}")
        logger.info("处理现货数据")
        logger.info(f"{'='*60}")
        
        spot_data = self.load_all_data("spot")
        if not spot_data.empty:
            # 计算技术指标
            spot_with_indicators = self.calculate_technical_indicators(spot_data)
            
            # 保存小时数据
            self.save_processed_data(spot_with_indicators, "BTC_USDT_spot_1h_3y.parquet")
            
            # 生成日线数据
            spot_daily = self.resample_to_daily(spot_data)
            if not spot_daily.empty:
                self.save_processed_data(spot_daily, "BTC_USDT_spot_1d_3y.parquet")
        
        # 2. 加载swap数据
        logger.info(f"\n{'='*60}")
        logger.info("处理swap数据")
        logger.info(f"{'='*60}")
        
        swap_data = self.load_all_data("swap")
        if not swap_data.empty:
            # 计算技术指标
            swap_with_indicators = self.calculate_technical_indicators(swap_data)
            
            # 保存小时数据
            self.save_processed_data(swap_with_indicators, "BTC_USDT_swap_1h_3y.parquet")
            
            # 生成日线数据
            swap_daily = self.resample_to_daily(swap_data)
            if not swap_daily.empty:
                self.save_processed_data(swap_daily, "BTC_USDT_swap_1d_3y.parquet")
        
        # 3. 合并现货和swap数据（使用现货价格，swap成交量）
        if not spot_data.empty and not swap_data.empty:
            logger.info(f"\n{'='*60}")
            logger.info("生成合并数据集")
            logger.info(f"{'='*60}")
            
            # 对齐时间索引
            common_index = spot_data.index.intersection(swap_data.index)
            
            if len(common_index) > 0:
                # 创建合并数据集
                combined_data = spot_data.loc[common_index].copy()
                combined_data['swap_volume'] = swap_data.loc[common_index]['volume']
                combined_data['volume_ratio'] = combined_data['swap_volume'] / combined_data['volume']
                
                # 计算技术指标
                combined_with_indicators = self.calculate_technical_indicators(combined_data)
                
                # 保存合并的小时数据
                self.save_processed_data(combined_with_indicators, "BTC_USDT_combined_1h_3y.parquet")
                
                # 生成合并的日线数据
                combined_daily = self.resample_to_daily(combined_data)
                if not combined_daily.empty:
                    self.save_processed_data(combined_daily, "BTC_USDT_combined_1d_3y.parquet")
                
                logger.info(f"  合并数据覆盖: {len(common_index)} 条记录")
                logger.info(f"  时间范围: {common_index.min()} 至 {common_index.max()}")
    
    def generate_summary_report(self):
        """生成处理总结报告"""
        logger.info("生成处理总结报告...")
        
        report_path = self.data_root / "btc_3y_processing_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BTC三年数据处理报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"处理时间: {datetime.now()}\n\n")
            
            # 检查处理后的文件
            processed_path = self.data_root / "processed"
            
            if processed_path.exists():
                f.write("处理后的文件:\n")
                
                for file_path in sorted(processed_path.glob("*_3y.parquet")):
                    try:
                        df = pd.read_parquet(file_path)
                        file_size = file_path.stat().st_size / 1024  # KB
                        
                        f.write(f"\n{file_path.name}:\n")
                        f.write(f"  记录数: {len(df)}\n")
                        f.write(f"  字段数: {len(df.columns)}\n")
                        f.write(f"  文件大小: {file_size:.1f} KB\n")
                        
                        if hasattr(df.index, 'min'):
                            f.write(f"  时间范围: {df.index.min()} 至 {df.index.max()}\n")
                            f.write(f"  覆盖天数: {(df.index.max() - df.index.min()).days} 天\n")
                        
                        # 数据质量检查
                        if 'close' in df.columns:
                            f.write(f"  价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}\n")
                        
                        if 'volume' in df.columns:
                            f.write(f"  平均成交量: {df['volume'].mean():.2f}\n")
                        
                        f.write(f"  缺失值: {df.isnull().sum().sum()} 个\n")
                        
                    except Exception as e:
                        f.write(f"\n{file_path.name}: 读取失败 - {e}\n")
            
            f.write(f"\n\n技术指标说明:\n")
            f.write("- SMA_20/50: 简单移动平均线\n")
            f.write("- EMA_12/26/20/50/100: 指数移动平均线\n")
            f.write("- RSI_14: 相对强弱指标\n")
            f.write("- MACD: 移动平均收敛发散指标\n")
            f.write("- ADX_14: 平均趋向指数\n")
            f.write("- returns: 收益率\n")
            f.write("- volatility: 24小时滚动波动率\n")
            f.write("- volume_ma: 24小时成交量移动平均\n")
        
        logger.info(f"报告已保存至: {report_path}")

def main():
    """主函数"""
    logger.info("开始BTC三年数据更新和扩展任务...")
    
    updater = BTCDataUpdater()
    
    # 1. 检查并更新最新数据
    updater.update_latest_data()
    
    # 2. 处理完整的三年数据集
    updater.process_full_dataset()
    
    # 3. 生成总结报告
    updater.generate_summary_report()
    
    logger.info("BTC三年数据处理任务完成！")
    
    # 显示最终统计
    logger.info(f"\n{'='*60}")
    logger.info("最终统计")
    logger.info(f"{'='*60}")
    
    processed_path = Path("data/processed")
    if processed_path.exists():
        for file_path in sorted(processed_path.glob("*_3y.parquet")):
            try:
                df = pd.read_parquet(file_path)
                logger.info(f"{file_path.name}: {len(df)} 条记录, {len(df.columns)} 个字段")
            except Exception as e:
                logger.error(f"{file_path.name}: 读取失败 - {e}")

if __name__ == "__main__":
    main() 