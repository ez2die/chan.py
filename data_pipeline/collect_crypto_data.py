#!/usr/bin/env python3
"""
加密货币数据抓取脚本
抓取ETH-USDT-SWAP、DOGE-USDT-SWAP和SOL-USDT-SWAP的近三年数据
包括：日线数据、资金费数据和交易量数据
"""

import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import time
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data_pipeline.collectors.collector import DataCollector
from data_pipeline.collectors.derivatives_collector import DerivativesCollector
from data_pipeline.config import settings

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CryptoDataCollector:
    """加密货币数据收集器"""
    
    def __init__(self):
        """初始化收集器"""
        self.data_collector = DataCollector()
        self.derivatives_collector = DerivativesCollector()
        
        # 目标交易对 (OKX SWAP格式: BASE/QUOTE:QUOTE)
        self.target_symbols = [
            'ETH/USDT:USDT',
            'DOGE/USDT:USDT', 
            'SOL/USDT:USDT'
        ]
        
        # 时间范围：近三年
        self.end_date = datetime.now(timezone.utc)
        self.start_date = self.end_date - timedelta(days=3*365)  # 3年
        
        logger.info(f"初始化数据收集器")
        logger.info(f"目标交易对: {self.target_symbols}")
        logger.info(f"时间范围: {self.start_date.strftime('%Y-%m-%d')} 到 {self.end_date.strftime('%Y-%m-%d')}")
    
    def collect_daily_ohlcv_data(self):
        """收集日线OHLCV数据"""
        logger.info("="*60)
        logger.info("开始收集日线OHLCV数据")
        logger.info("="*60)
        
        results = {}
        
        for symbol in self.target_symbols:
            try:
                logger.info(f"\n📊 正在收集 {symbol} 的日线数据...")
                
                # 获取日线数据
                df = self.data_collector.fetch_historical_data(
                    symbol=symbol,
                    timeframe='1d',
                    start_date=self.start_date,
                    end_date=self.end_date,
                    instrument_type='swap'
                )
                
                if df.empty:
                    logger.warning(f"❌ {symbol} 没有返回数据")
                    continue
                
                # 保存数据
                self.data_collector.save_data(
                    data=df,
                    symbol=symbol,
                    timeframe='1d',
                    instrument_type='swap'
                )
                
                results[symbol] = {
                    'records': len(df),
                    'start_date': df.index.min(),
                    'end_date': df.index.max(),
                    'avg_volume': df['volume'].mean(),
                    'price_range': (df['close'].min(), df['close'].max())
                }
                
                logger.info(f"✅ {symbol} 日线数据收集完成:")
                logger.info(f"   📈 记录数: {len(df)}")
                logger.info(f"   📅 时间范围: {df.index.min()} 到 {df.index.max()}")
                logger.info(f"   💰 价格范围: ${df['close'].min():.4f} - ${df['close'].max():.4f}")
                logger.info(f"   📊 平均交易量: {df['volume'].mean():.2f}")
                
                # 添加延迟避免API限制
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ 收集 {symbol} 日线数据时出错: {e}")
                continue
        
        return results
    
    def collect_funding_rate_data(self):
        """收集资金费率数据"""
        logger.info("="*60)
        logger.info("开始收集资金费率数据")
        logger.info("="*60)
        
        try:
            # 计算天数
            days = (self.end_date - self.start_date).days
            
            # 收集资金费率数据
            funding_data = self.derivatives_collector.collect_funding_rates(
                symbols=self.target_symbols,
                days=days,
                save_data=True
            )
            
            results = {}
            for symbol, df in funding_data.items():
                if not df.empty:
                    results[symbol] = {
                        'records': len(df),
                        'start_date': df.index.min(),
                        'end_date': df.index.max(),
                        'avg_funding_rate': df['fundingRate'].mean() if 'fundingRate' in df.columns else 0,
                        'funding_rate_range': (
                            df['fundingRate'].min() if 'fundingRate' in df.columns else 0,
                            df['fundingRate'].max() if 'fundingRate' in df.columns else 0
                        )
                    }
                    
                    logger.info(f"✅ {symbol} 资金费率数据收集完成:")
                    logger.info(f"   📈 记录数: {len(df)}")
                    logger.info(f"   📅 时间范围: {df.index.min()} 到 {df.index.max()}")
                    if 'fundingRate' in df.columns:
                        logger.info(f"   💸 平均资金费率: {df['fundingRate'].mean():.6f} ({df['fundingRate'].mean()*100:.4f}%)")
                        logger.info(f"   📊 费率范围: {df['fundingRate'].min():.6f} 到 {df['fundingRate'].max():.6f}")
                
            return results
            
        except Exception as e:
            logger.error(f"❌ 收集资金费率数据时出错: {e}")
            return {}
    
    def collect_open_interest_data(self):
        """收集持仓量数据"""
        logger.info("="*60)
        logger.info("开始收集持仓量数据")
        logger.info("="*60)
        
        try:
            # 收集持仓量数据
            oi_data = self.derivatives_collector.collect_open_interest(
                symbols=self.target_symbols,
                save_data=True
            )
            
            results = {}
            for symbol, df in oi_data.items():
                if not df.empty:
                    results[symbol] = {
                        'records': len(df),
                        'latest_timestamp': df.index.max() if hasattr(df.index, 'max') else 'N/A',
                        'open_interest_info': df.to_dict() if len(df) < 10 else f"{len(df)} records"
                    }
                    
                    logger.info(f"✅ {symbol} 持仓量数据收集完成:")
                    logger.info(f"   📈 记录数: {len(df)}")
                    logger.info(f"   📅 最新时间: {df.index.max() if hasattr(df.index, 'max') else 'N/A'}")
                
            return results
            
        except Exception as e:
            logger.error(f"❌ 收集持仓量数据时出错: {e}")
            return {}
    
    def collect_hourly_data_for_volume_analysis(self):
        """收集小时线数据用于交易量分析"""
        logger.info("="*60)
        logger.info("开始收集小时线数据（用于交易量分析）")
        logger.info("="*60)
        
        # 收集最近30天的小时数据进行交易量分析
        recent_start = self.end_date - timedelta(days=30)
        
        results = {}
        
        for symbol in self.target_symbols:
            try:
                logger.info(f"\n📊 正在收集 {symbol} 的小时线数据...")
                
                # 获取小时线数据
                df = self.data_collector.fetch_historical_data(
                    symbol=symbol,
                    timeframe='1h',
                    start_date=recent_start,
                    end_date=self.end_date,
                    instrument_type='swap'
                )
                
                if df.empty:
                    logger.warning(f"❌ {symbol} 没有返回小时线数据")
                    continue
                
                # 计算交易量统计
                volume_stats = {
                    'total_volume': df['volume'].sum(),
                    'avg_hourly_volume': df['volume'].mean(),
                    'max_hourly_volume': df['volume'].max(),
                    'volume_std': df['volume'].std(),
                    'high_volume_hours': len(df[df['volume'] > df['volume'].quantile(0.9)])
                }
                
                results[symbol] = {
                    'records': len(df),
                    'start_date': df.index.min(),
                    'end_date': df.index.max(),
                    'volume_stats': volume_stats
                }
                
                logger.info(f"✅ {symbol} 小时线数据收集完成:")
                logger.info(f"   📈 记录数: {len(df)}")
                logger.info(f"   📅 时间范围: {df.index.min()} 到 {df.index.max()}")
                logger.info(f"   📊 总交易量: {volume_stats['total_volume']:.2f}")
                logger.info(f"   📊 平均小时交易量: {volume_stats['avg_hourly_volume']:.2f}")
                logger.info(f"   📊 最大小时交易量: {volume_stats['max_hourly_volume']:.2f}")
                logger.info(f"   📊 高交易量小时数: {volume_stats['high_volume_hours']}")
                
                # 添加延迟避免API限制
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ 收集 {symbol} 小时线数据时出错: {e}")
                continue
        
        return results
    
    def generate_summary_report(self, daily_results, funding_results, oi_results, hourly_results):
        """生成数据收集汇总报告"""
        logger.info("="*60)
        logger.info("生成数据收集汇总报告")
        logger.info("="*60)
        
        report = {
            'collection_time': datetime.now(timezone.utc).isoformat(),
            'target_symbols': self.target_symbols,
            'time_range': {
                'start': self.start_date.isoformat(),
                'end': self.end_date.isoformat(),
                'days': (self.end_date - self.start_date).days
            },
            'daily_data': daily_results,
            'funding_data': funding_results,
            'open_interest_data': oi_results,
            'hourly_data': hourly_results
        }
        
        # 保存报告
        report_path = Path('crypto_data_collection_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            import json
            # 自定义JSON序列化函数
            def json_serializer(obj):
                if hasattr(obj, 'isoformat'):  # datetime对象
                    return obj.isoformat()
                elif hasattr(obj, 'item'):  # numpy对象
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy数组
                    return obj.tolist()
                else:
                    return str(obj)
            
            json.dump(report, f, indent=2, ensure_ascii=False, default=json_serializer)
        
        logger.info(f"📋 数据收集报告已保存到: {report_path}")
        
        # 打印汇总信息
        logger.info("\n📊 数据收集汇总:")
        logger.info(f"   🎯 目标交易对: {len(self.target_symbols)}")
        logger.info(f"   📅 时间范围: {(self.end_date - self.start_date).days} 天")
        logger.info(f"   📈 日线数据: {len(daily_results)} 个交易对")
        logger.info(f"   💸 资金费率数据: {len(funding_results)} 个交易对")
        logger.info(f"   📊 持仓量数据: {len(oi_results)} 个交易对")
        logger.info(f"   ⏰ 小时线数据: {len(hourly_results)} 个交易对")
        
        return report
    
    def run_full_collection(self):
        """执行完整的数据收集流程"""
        logger.info("🚀 开始执行完整的加密货币数据收集流程")
        logger.info(f"📊 目标交易对: {', '.join(self.target_symbols)}")
        logger.info(f"📅 时间范围: {self.start_date.strftime('%Y-%m-%d')} 到 {self.end_date.strftime('%Y-%m-%d')}")
        
        start_time = time.time()
        
        try:
            # 1. 收集日线数据
            daily_results = self.collect_daily_ohlcv_data()
            
            # 2. 收集资金费率数据
            funding_results = self.collect_funding_rate_data()
            
            # 3. 收集持仓量数据
            oi_results = self.collect_open_interest_data()
            
            # 4. 收集小时线数据进行交易量分析
            hourly_results = self.collect_hourly_data_for_volume_analysis()
            
            # 5. 生成汇总报告
            report = self.generate_summary_report(
                daily_results, funding_results, oi_results, hourly_results
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("="*60)
            logger.info("🎉 数据收集完成!")
            logger.info(f"⏱️  总耗时: {duration:.2f} 秒")
            logger.info("="*60)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ 数据收集过程中出现错误: {e}")
            raise


def main():
    """主函数"""
    try:
        # 创建数据收集器
        collector = CryptoDataCollector()
        
        # 执行完整收集流程
        report = collector.run_full_collection()
        
        print("\n✅ 数据收集任务完成!")
        print(f"📋 详细报告已保存到: crypto_data_collection_report.json")
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 