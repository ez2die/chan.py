#!/usr/bin/env python3
"""
补充脚本：收集资金费率和持仓量数据
使用正确的symbol格式
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

from data_pipeline.collectors.derivatives_collector import DerivativesCollector
from data_pipeline.config import settings

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('funding_oi_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FundingAndOICollector:
    """资金费率和持仓量数据收集器"""
    
    def __init__(self):
        """初始化收集器"""
        self.derivatives_collector = DerivativesCollector()
        
        # 目标交易对映射
        self.symbol_mapping = {
            # OHLCV格式 -> 资金费率/持仓量格式
            'ETH/USDT:USDT': 'ETH-USDT-SWAP',
            'DOGE/USDT:USDT': 'DOGE-USDT-SWAP',
            'SOL/USDT:USDT': 'SOL-USDT-SWAP'
        }
        
        # 时间范围：近三年
        self.end_date = datetime.now(timezone.utc)
        self.start_date = self.end_date - timedelta(days=3*365)  # 3年
        
        logger.info(f"初始化资金费率和持仓量数据收集器")
        logger.info(f"目标交易对: {list(self.symbol_mapping.values())}")
        logger.info(f"时间范围: {self.start_date.strftime('%Y-%m-%d')} 到 {self.end_date.strftime('%Y-%m-%d')}")
    
    def collect_funding_rates(self):
        """收集资金费率数据"""
        logger.info("="*60)
        logger.info("开始收集资金费率数据")
        logger.info("="*60)
        
        try:
            # 计算天数
            days = (self.end_date - self.start_date).days
            
            # 使用正确的symbol格式
            swap_symbols = list(self.symbol_mapping.values())
            
            # 收集资金费率数据
            funding_data = self.derivatives_collector.collect_funding_rates(
                symbols=swap_symbols,
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
                        ),
                        'total_funding_events': len(df)
                    }
                    
                    logger.info(f"✅ {symbol} 资金费率数据收集完成:")
                    logger.info(f"   📈 记录数: {len(df)}")
                    logger.info(f"   📅 时间范围: {df.index.min()} 到 {df.index.max()}")
                    if 'fundingRate' in df.columns:
                        avg_rate = df['fundingRate'].mean()
                        min_rate = df['fundingRate'].min()
                        max_rate = df['fundingRate'].max()
                        logger.info(f"   💸 平均资金费率: {avg_rate:.6f} ({avg_rate*100:.4f}%)")
                        logger.info(f"   📊 费率范围: {min_rate:.6f} 到 {max_rate:.6f}")
                        logger.info(f"   📊 年化平均费率: {avg_rate * 3 * 365:.4f}% (每日3次)")
                
            return results
            
        except Exception as e:
            logger.error(f"❌ 收集资金费率数据时出错: {e}")
            return {}
    
    def collect_open_interest(self):
        """收集持仓量数据"""
        logger.info("="*60)
        logger.info("开始收集持仓量数据")
        logger.info("="*60)
        
        try:
            # 使用正确的symbol格式
            swap_symbols = list(self.symbol_mapping.values())
            
            # 收集持仓量数据
            oi_data = self.derivatives_collector.collect_open_interest(
                symbols=swap_symbols,
                save_data=True
            )
            
            results = {}
            for symbol, df in oi_data.items():
                if not df.empty:
                    # 解析持仓量数据
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        latest_oi = df.iloc[-1] if len(df) > 0 else None
                        
                        results[symbol] = {
                            'records': len(df),
                            'latest_timestamp': df.index.max() if hasattr(df.index, 'max') else 'N/A',
                            'latest_open_interest': latest_oi.to_dict() if latest_oi is not None else {},
                            'data_summary': f"Latest OI data available"
                        }
                        
                        logger.info(f"✅ {symbol} 持仓量数据收集完成:")
                        logger.info(f"   📈 记录数: {len(df)}")
                        logger.info(f"   📅 最新时间: {df.index.max() if hasattr(df.index, 'max') else 'N/A'}")
                        
                        # 打印持仓量详情
                        if latest_oi is not None:
                            for key, value in latest_oi.items():
                                if key not in ['datetime', 'timestamp']:
                                    logger.info(f"   📊 {key}: {value}")
                
            return results
            
        except Exception as e:
            logger.error(f"❌ 收集持仓量数据时出错: {e}")
            return {}
    
    def analyze_funding_patterns(self, funding_results):
        """分析资金费率模式"""
        logger.info("="*60)
        logger.info("分析资金费率模式")
        logger.info("="*60)
        
        analysis = {}
        
        for symbol in self.symbol_mapping.values():
            try:
                # 读取已保存的资金费率数据
                base_currency = symbol.split('-')[0]
                quote_currency = symbol.split('-')[1]
                
                funding_dir = Path(settings.data_dir) / "okx" / "derivatives" / "funding_rates" / base_currency / quote_currency
                
                if funding_dir.exists():
                    # 查找最新的资金费率文件
                    funding_files = list(funding_dir.glob("funding_rates_*.parquet"))
                    if funding_files:
                        latest_file = max(funding_files, key=lambda x: x.stat().st_mtime)
                        df = pd.read_parquet(latest_file)
                        
                        if not df.empty and 'fundingRate' in df.columns:
                            # 计算统计指标
                            rates = df['fundingRate']
                            
                            analysis[symbol] = {
                                'total_periods': len(rates),
                                'positive_rate_periods': len(rates[rates > 0]),
                                'negative_rate_periods': len(rates[rates < 0]),
                                'zero_rate_periods': len(rates[rates == 0]),
                                'avg_positive_rate': rates[rates > 0].mean() if len(rates[rates > 0]) > 0 else 0,
                                'avg_negative_rate': rates[rates < 0].mean() if len(rates[rates < 0]) > 0 else 0,
                                'max_positive_rate': rates.max(),
                                'max_negative_rate': rates.min(),
                                'rate_volatility': rates.std(),
                                'annualized_avg_rate': rates.mean() * 3 * 365  # 每日3次
                            }
                            
                            logger.info(f"📊 {symbol} 资金费率分析:")
                            logger.info(f"   总周期数: {analysis[symbol]['total_periods']}")
                            logger.info(f"   正费率周期: {analysis[symbol]['positive_rate_periods']} ({analysis[symbol]['positive_rate_periods']/analysis[symbol]['total_periods']*100:.1f}%)")
                            logger.info(f"   负费率周期: {analysis[symbol]['negative_rate_periods']} ({analysis[symbol]['negative_rate_periods']/analysis[symbol]['total_periods']*100:.1f}%)")
                            logger.info(f"   平均正费率: {analysis[symbol]['avg_positive_rate']:.6f}")
                            logger.info(f"   平均负费率: {analysis[symbol]['avg_negative_rate']:.6f}")
                            logger.info(f"   年化平均费率: {analysis[symbol]['annualized_avg_rate']:.2f}%")
                            logger.info(f"   费率波动率: {analysis[symbol]['rate_volatility']:.6f}")
                
            except Exception as e:
                logger.error(f"❌ 分析 {symbol} 资金费率时出错: {e}")
                continue
        
        return analysis
    
    def generate_comprehensive_report(self, funding_results, oi_results, funding_analysis):
        """生成综合报告"""
        logger.info("="*60)
        logger.info("生成综合数据报告")
        logger.info("="*60)
        
        report = {
            'collection_time': datetime.now(timezone.utc).isoformat(),
            'target_symbols': list(self.symbol_mapping.values()),
            'symbol_mapping': self.symbol_mapping,
            'time_range': {
                'start': self.start_date.isoformat(),
                'end': self.end_date.isoformat(),
                'days': (self.end_date - self.start_date).days
            },
            'funding_rate_data': funding_results,
            'open_interest_data': oi_results,
            'funding_rate_analysis': funding_analysis,
            'data_summary': {
                'funding_symbols_collected': len(funding_results),
                'oi_symbols_collected': len(oi_results),
                'total_funding_periods': sum([r.get('total_funding_events', 0) for r in funding_results.values()]),
                'collection_success_rate': f"{len(funding_results)}/{len(self.symbol_mapping)} funding, {len(oi_results)}/{len(self.symbol_mapping)} OI"
            }
        }
        
        # 保存报告
        report_path = Path('funding_oi_collection_report.json')
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
        
        logger.info(f"📋 综合数据报告已保存到: {report_path}")
        
        # 打印汇总信息
        logger.info("\n📊 数据收集汇总:")
        logger.info(f"   🎯 目标交易对: {len(self.symbol_mapping)}")
        logger.info(f"   📅 时间范围: {(self.end_date - self.start_date).days} 天")
        logger.info(f"   💸 资金费率数据: {len(funding_results)} 个交易对")
        logger.info(f"   📊 持仓量数据: {len(oi_results)} 个交易对")
        logger.info(f"   📈 总资金费率周期: {report['data_summary']['total_funding_periods']}")
        
        return report
    
    def run_collection(self):
        """执行完整的数据收集流程"""
        logger.info("🚀 开始执行资金费率和持仓量数据收集")
        logger.info(f"📊 目标交易对: {', '.join(self.symbol_mapping.values())}")
        logger.info(f"📅 时间范围: {self.start_date.strftime('%Y-%m-%d')} 到 {self.end_date.strftime('%Y-%m-%d')}")
        
        start_time = time.time()
        
        try:
            # 1. 收集资金费率数据
            funding_results = self.collect_funding_rates()
            
            # 2. 收集持仓量数据
            oi_results = self.collect_open_interest()
            
            # 3. 分析资金费率模式
            funding_analysis = self.analyze_funding_patterns(funding_results)
            
            # 4. 生成综合报告
            report = self.generate_comprehensive_report(
                funding_results, oi_results, funding_analysis
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("="*60)
            logger.info("🎉 资金费率和持仓量数据收集完成!")
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
        collector = FundingAndOICollector()
        
        # 执行完整收集流程
        report = collector.run_collection()
        
        print("\n✅ 资金费率和持仓量数据收集任务完成!")
        print(f"📋 详细报告已保存到: funding_oi_collection_report.json")
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 