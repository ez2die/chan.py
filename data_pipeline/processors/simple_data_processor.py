"""
简化的数据处理器 - 不依赖外部库
展示实际的数据处理和指标计算逻辑
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import statistics
from pathlib import Path


class SimpleOHLCVData:
    """OHLCV数据结构"""
    def __init__(self, timestamp, open_price, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        
    def __repr__(self):
        return f"OHLCV({self.timestamp}, O:{self.open:.2f}, H:{self.high:.2f}, L:{self.low:.2f}, C:{self.close:.2f}, V:{self.volume:.2f})"


class SimpleDataProcessor:
    """
    简化的数据处理器，展示实际的数据处理逻辑
    """
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.spot_path = self.data_root / "okx" / "spot"
        self.derivatives_path = self.data_root / "derivatives_history"
        
    def load_btc_sample_data(self) -> List[SimpleOHLCVData]:
        """
        加载BTC/USDT样本数据
        由于无法使用pandas，这里模拟读取过程
        """
        print("📊 正在加载BTC/USDT数据...")
        
        # 检查数据文件是否存在
        btc_path = self.spot_path / "BTC" / "USDT" / "1h"
        
        if btc_path.exists():
            print(f"✅ 找到数据目录: {btc_path}")
            
            # 列出parquet文件
            parquet_files = list(btc_path.glob("*.parquet"))
            for f in parquet_files:
                file_size = f.stat().st_size / 1024 / 1024  # MB
                print(f"  📁 {f.name} ({file_size:.2f} MB)")
                
            # 由于无法读取parquet，生成示例数据
            print("\n⚠️  由于环境限制，使用模拟数据展示处理流程")
            return self._generate_sample_data()
        else:
            print("❌ 数据目录不存在，生成示例数据")
            return self._generate_sample_data()
            
    def _generate_sample_data(self, days: int = 30) -> List[SimpleOHLCVData]:
        """生成示例OHLCV数据"""
        data = []
        base_price = 42000
        current_time = datetime(2024, 1, 1)
        
        for i in range(days * 24):  # 小时数据
            # 模拟价格波动
            import random
            volatility = random.uniform(-0.02, 0.02)
            
            close = base_price * (1 + volatility)
            open_price = base_price
            high = max(open_price, close) * (1 + random.uniform(0, 0.005))
            low = min(open_price, close) * (1 - random.uniform(0, 0.005))
            volume = random.uniform(100, 200)
            
            data.append(SimpleOHLCVData(
                timestamp=current_time + timedelta(hours=i),
                open_price=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume
            ))
            
            base_price = close
            
        return data
    
    def calculate_sma(self, data: List[SimpleOHLCVData], period: int) -> List[float]:
        """计算简单移动平均线"""
        sma_values = []
        
        for i in range(len(data)):
            if i < period - 1:
                sma_values.append(None)
            else:
                # 计算最近period个收盘价的平均值
                closes = [data[j].close for j in range(i - period + 1, i + 1)]
                sma = sum(closes) / period
                sma_values.append(sma)
                
        return sma_values
    
    def calculate_ema(self, data: List[SimpleOHLCVData], period: int) -> List[float]:
        """计算指数移动平均线"""
        ema_values = []
        multiplier = 2 / (period + 1)
        
        # 初始EMA使用SMA
        sma = self.calculate_sma(data, period)
        
        for i in range(len(data)):
            if i < period - 1:
                ema_values.append(None)
            elif i == period - 1:
                ema_values.append(sma[i])
            else:
                # EMA = (Close - EMA_prev) * multiplier + EMA_prev
                ema = (data[i].close - ema_values[i-1]) * multiplier + ema_values[i-1]
                ema_values.append(ema)
                
        return ema_values
    
    def calculate_rsi(self, data: List[SimpleOHLCVData], period: int = 14) -> List[float]:
        """计算RSI指标"""
        rsi_values = []
        
        # 计算价格变化
        changes = []
        for i in range(1, len(data)):
            changes.append(data[i].close - data[i-1].close)
            
        for i in range(len(data)):
            if i < period:
                rsi_values.append(None)
            else:
                # 计算平均涨幅和跌幅
                gains = [c for c in changes[i-period:i] if c > 0]
                losses = [-c for c in changes[i-period:i] if c < 0]
                
                avg_gain = sum(gains) / period if gains else 0
                avg_loss = sum(losses) / period if losses else 0
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                rsi_values.append(rsi)
                
        return rsi_values
    
    def calculate_adx(self, data: List[SimpleOHLCVData], period: int = 14) -> List[float]:
        """计算ADX指标（简化版）"""
        adx_values = []
        
        for i in range(len(data)):
            if i < period * 2:
                adx_values.append(None)
            else:
                # 简化的ADX计算，基于价格波动率
                price_range = []
                for j in range(i - period, i):
                    price_range.append(data[j].high - data[j].low)
                
                avg_range = sum(price_range) / len(price_range)
                avg_price = sum([data[j].close for j in range(i - period, i)]) / period
                
                # ADX近似值（0-100）
                adx = (avg_range / avg_price) * 1000
                adx = min(100, max(0, adx))
                
                adx_values.append(adx)
                
        return adx_values
    
    def resample_to_daily(self, hourly_data: List[SimpleOHLCVData]) -> List[SimpleOHLCVData]:
        """将小时数据重采样为日线数据"""
        daily_data = []
        
        # 按日期分组
        current_date = None
        day_data = []
        
        for bar in hourly_data:
            bar_date = bar.timestamp.date()
            
            if current_date is None:
                current_date = bar_date
                
            if bar_date != current_date:
                # 新的一天，处理前一天的数据
                if day_data:
                    daily_bar = SimpleOHLCVData(
                        timestamp=datetime.combine(current_date, datetime.min.time()),
                        open_price=day_data[0].open,
                        high=max(d.high for d in day_data),
                        low=min(d.low for d in day_data),
                        close=day_data[-1].close,
                        volume=sum(d.volume for d in day_data)
                    )
                    daily_data.append(daily_bar)
                
                current_date = bar_date
                day_data = [bar]
            else:
                day_data.append(bar)
        
        # 处理最后一天
        if day_data:
            daily_bar = SimpleOHLCVData(
                timestamp=datetime.combine(current_date, datetime.min.time()),
                open_price=day_data[0].open,
                high=max(d.high for d in day_data),
                low=min(d.low for d in day_data),
                close=day_data[-1].close,
                volume=sum(d.volume for d in day_data)
            )
            daily_data.append(daily_bar)
            
        return daily_data
    
    def process_and_display(self):
        """处理数据并展示结果"""
        print("\n🚀 开始数据处理流程\n")
        
        # 1. 加载数据
        print("📌 步骤1: 加载BTC/USDT数据")
        hourly_data = self.load_btc_sample_data()
        print(f"  ✅ 加载了 {len(hourly_data)} 条小时数据")
        
        # 展示前几条数据
        print("\n  📊 数据样本:")
        for i in range(min(3, len(hourly_data))):
            print(f"    {hourly_data[i]}")
        
        # 2. 计算技术指标
        print("\n📌 步骤2: 计算技术指标")
        
        # SMA
        sma_20 = self.calculate_sma(hourly_data, 20)
        sma_50 = self.calculate_sma(hourly_data, 50)
        print(f"  ✅ 计算SMA(20)和SMA(50)")
        
        # EMA
        ema_20 = self.calculate_ema(hourly_data, 20)
        ema_50 = self.calculate_ema(hourly_data, 50)
        print(f"  ✅ 计算EMA(20)和EMA(50)")
        
        # RSI
        rsi_14 = self.calculate_rsi(hourly_data, 14)
        print(f"  ✅ 计算RSI(14)")
        
        # ADX
        adx_14 = self.calculate_adx(hourly_data, 14)
        print(f"  ✅ 计算ADX(14)")
        
        # 3. 生成日线数据
        print("\n📌 步骤3: 生成日线数据")
        daily_data = self.resample_to_daily(hourly_data)
        print(f"  ✅ 生成了 {len(daily_data)} 条日线数据")
        
        # 4. 加载衍生品数据信息
        print("\n📌 步骤4: 检查衍生品数据")
        self.check_derivatives_data()
        
        # 5. 数据质量检查
        print("\n📌 步骤5: 数据质量报告")
        self.generate_quality_report(hourly_data, sma_20, ema_20, rsi_14, adx_14)
        
        # 6. 保存处理结果
        print("\n📌 步骤6: 保存处理结果")
        self.save_processed_data(hourly_data, daily_data, {
            'SMA_20': sma_20,
            'SMA_50': sma_50,
            'EMA_20': ema_20,
            'EMA_50': ema_50,
            'RSI_14': rsi_14,
            'ADX_14': adx_14
        })
        
        print("\n✅ 数据处理完成！")
        
    def check_derivatives_data(self):
        """检查衍生品数据"""
        funding_file = self.derivatives_path / "BTC_USDT_SWAP_funding_rate_3y.csv"
        oi_file = self.derivatives_path / "BTC_USDT_SWAP_open_interest_3y.csv"
        
        if funding_file.exists():
            size_mb = funding_file.stat().st_size / 1024 / 1024
            print(f"  ✅ 资金费率数据: {funding_file.name} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ 资金费率数据不存在")
            
        if oi_file.exists():
            size_mb = oi_file.stat().st_size / 1024 / 1024
            print(f"  ✅ 持仓量数据: {oi_file.name} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ 持仓量数据不存在")
            
    def generate_quality_report(self, data, sma_20, ema_20, rsi_14, adx_14):
        """生成数据质量报告"""
        print(f"  📊 数据统计:")
        
        # 价格统计
        prices = [d.close for d in data]
        print(f"    - 价格范围: ${min(prices):,.2f} - ${max(prices):,.2f}")
        print(f"    - 平均价格: ${sum(prices)/len(prices):,.2f}")
        
        # 指标统计
        valid_sma = [v for v in sma_20 if v is not None]
        valid_rsi = [v for v in rsi_14 if v is not None]
        valid_adx = [v for v in adx_14 if v is not None]
        
        if valid_sma:
            print(f"    - SMA(20)平均值: ${sum(valid_sma)/len(valid_sma):,.2f}")
        
        if valid_rsi:
            print(f"    - RSI(14)范围: {min(valid_rsi):.2f} - {max(valid_rsi):.2f}")
            print(f"    - RSI(14)平均值: {sum(valid_rsi)/len(valid_rsi):.2f}")
            
        if valid_adx:
            print(f"    - ADX(14)平均值: {sum(valid_adx)/len(valid_adx):.2f}")
            
        # 市场状态分析
        if valid_adx:
            trend_count = sum(1 for v in valid_adx if v > 25)
            ranging_count = sum(1 for v in valid_adx if v < 20)
            neutral_count = len(valid_adx) - trend_count - ranging_count
            
            print(f"\n  🌍 市场状态分布:")
            print(f"    - 趋势市场: {trend_count/len(valid_adx)*100:.1f}%")
            print(f"    - 震荡市场: {ranging_count/len(valid_adx)*100:.1f}%")
            print(f"    - 中性市场: {neutral_count/len(valid_adx)*100:.1f}%")
            
    def save_processed_data(self, hourly_data, daily_data, indicators):
        """保存处理后的数据（JSON格式）"""
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存最后10条数据作为示例
        sample_data = {
            'metadata': {
                'symbol': 'BTC/USDT',
                'timeframe': '1h',
                'total_records': len(hourly_data),
                'indicators': list(indicators.keys()),
                'generated_at': datetime.now().isoformat()
            },
            'sample_data': []
        }
        
        # 添加最后10条数据
        for i in range(max(0, len(hourly_data) - 10), len(hourly_data)):
            record = {
                'timestamp': hourly_data[i].timestamp.isoformat(),
                'ohlcv': {
                    'open': hourly_data[i].open,
                    'high': hourly_data[i].high,
                    'low': hourly_data[i].low,
                    'close': hourly_data[i].close,
                    'volume': hourly_data[i].volume
                },
                'indicators': {}
            }
            
            # 添加指标值
            for name, values in indicators.items():
                if i < len(values) and values[i] is not None:
                    record['indicators'][name] = round(values[i], 4)
                    
            sample_data['sample_data'].append(record)
            
        # 保存到JSON文件
        output_file = output_dir / "BTC_USDT_processed_sample.json"
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
            
        print(f"  ✅ 处理结果已保存到: {output_file}")
        
        # 显示文件内容预览
        print(f"\n  📄 文件内容预览:")
        print(f"    - 符号: {sample_data['metadata']['symbol']}")
        print(f"    - 时间框架: {sample_data['metadata']['timeframe']}")
        print(f"    - 总记录数: {sample_data['metadata']['total_records']}")
        print(f"    - 包含指标: {', '.join(sample_data['metadata']['indicators'])}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🚀 BTC数据处理器 - 实际执行演示")
    print("="*60)
    
    processor = SimpleDataProcessor()
    processor.process_and_display()
    
    print("\n" + "="*60)
    print("📝 总结")
    print("="*60)
    print("1. ✅ 成功展示了数据加载流程")
    print("2. ✅ 实现了技术指标计算（SMA, EMA, RSI, ADX）")
    print("3. ✅ 完成了小时数据到日线数据的重采样")
    print("4. ✅ 生成了数据质量报告")
    print("5. ✅ 保存了处理结果（JSON格式）")
    print("\n注：由于环境限制，使用了模拟数据展示处理流程。")
    print("实际应用中，DataProcessor会直接读取parquet文件。")
    print("="*60)


if __name__ == "__main__":
    main()