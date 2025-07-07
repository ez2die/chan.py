"""
CSV数据读取器 - 用于读取和分析衍生品数据
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple


class CSVDataReader:
    """CSV格式数据读取器"""
    
    def __init__(self):
        self.data_root = Path("data")
        
    def read_csv_file(self, filepath: str) -> Tuple[List[str], List[Dict]]:
        """读取CSV文件并返回头部和数据"""
        headers = []
        data = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                for row in reader:
                    data.append(row)
                    
            return headers, data
        except FileNotFoundError:
            print(f"❌ 文件不存在: {filepath}")
            return [], []
        except Exception as e:
            print(f"❌ 读取文件错误: {e}")
            return [], []
            
    def analyze_funding_rate(self):
        """分析资金费率数据"""
        print("\n📊 分析BTC永续合约资金费率数据")
        print("="*50)
        
        # 构建文件路径
        funding_file = self.data_root / "derivatives_history" / "BTC_USDT_SWAP_funding_rate_3y.csv"
        
        # 读取数据
        headers, data = self.read_csv_file(str(funding_file))
        
        if not data:
            print("❌ 无法读取资金费率数据")
            return
            
        print(f"✅ 成功读取 {len(data)} 条记录")
        print(f"📋 数据字段: {', '.join(headers)}")
        
        # 分析数据
        print("\n📈 数据分析:")
        
        # 时间范围
        if data:
            first_date = data[0].get('datetime', 'N/A')
            last_date = data[-1].get('datetime', 'N/A')
            print(f"  时间范围: {first_date} 至 {last_date}")
            
        # 资金费率统计
        funding_rates = []
        for row in data:
            try:
                rate = float(row.get('fundingRate', 0))
                funding_rates.append(rate)
            except:
                pass
                
        if funding_rates:
            avg_rate = sum(funding_rates) / len(funding_rates)
            max_rate = max(funding_rates)
            min_rate = min(funding_rates)
            
            print(f"  平均资金费率: {avg_rate:.6f} ({avg_rate*100:.4f}%)")
            print(f"  最高资金费率: {max_rate:.6f} ({max_rate*100:.4f}%)")
            print(f"  最低资金费率: {min_rate:.6f} ({min_rate*100:.4f}%)")
            
            # 正负费率统计
            positive_count = sum(1 for r in funding_rates if r > 0)
            negative_count = sum(1 for r in funding_rates if r < 0)
            neutral_count = len(funding_rates) - positive_count - negative_count
            
            print(f"\n  资金费率分布:")
            print(f"    正费率: {positive_count} ({positive_count/len(funding_rates)*100:.1f}%)")
            print(f"    负费率: {negative_count} ({negative_count/len(funding_rates)*100:.1f}%)")
            print(f"    零费率: {neutral_count} ({neutral_count/len(funding_rates)*100:.1f}%)")
            
        # 展示示例数据
        print("\n📋 数据示例 (前5条):")
        for i, row in enumerate(data[:5]):
            print(f"  {i+1}. 时间: {row.get('datetime')}, 费率: {row.get('fundingRate')}")
            
    def analyze_open_interest(self):
        """分析持仓量数据"""
        print("\n📊 分析BTC永续合约持仓量数据")
        print("="*50)
        
        # 构建文件路径
        oi_file = self.data_root / "derivatives_history" / "BTC_USDT_SWAP_open_interest_3y.csv"
        
        # 读取数据
        headers, data = self.read_csv_file(str(oi_file))
        
        if not data:
            print("❌ 无法读取持仓量数据")
            return
            
        print(f"✅ 成功读取 {len(data)} 条记录")
        print(f"📋 数据字段: {', '.join(headers)}")
        
        # 分析数据
        print("\n📈 数据分析:")
        
        # 时间范围
        if data:
            first_date = data[0].get('datetime', data[0].get('timestamp', 'N/A'))
            last_date = data[-1].get('datetime', data[-1].get('timestamp', 'N/A'))
            print(f"  时间范围: {first_date} 至 {last_date}")
            
        # 展示示例数据
        print("\n📋 数据示例 (前5条):")
        for i, row in enumerate(data[:5]):
            print(f"  {i+1}. {row}")
            
    def create_sample_parquet_info(self):
        """创建模拟的parquet文件信息"""
        print("\n📊 模拟Parquet文件信息")
        print("="*50)
        
        # 模拟BTC/USDT的parquet文件信息
        btc_info = {
            'symbol': 'BTC/USDT',
            'files': [
                {'name': '2024.parquet', 'records': 8784, 'size_mb': 12.5},
                {'name': '2025.parquet', 'records': 3869, 'size_mb': 5.8}
            ],
            'total_records': 12653,
            'date_range': '2024-01-01 至 2025-06-11',
            'fields': ['datetime', 'open', 'high', 'low', 'close', 'volume']
        }
        
        print(f"📁 {btc_info['symbol']} 数据文件:")
        for file_info in btc_info['files']:
            print(f"  - {file_info['name']}: {file_info['records']:,} 条记录 ({file_info['size_mb']} MB)")
            
        print(f"\n📊 数据概况:")
        print(f"  总记录数: {btc_info['total_records']:,}")
        print(f"  时间范围: {btc_info['date_range']}")
        print(f"  数据字段: {', '.join(btc_info['fields'])}")
        
        # 模拟其他交易对
        other_symbols = ['ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOGE/USDT']
        print(f"\n📁 其他交易对 (共74个):")
        for symbol in other_symbols:
            print(f"  - {symbol}: ~12,000 条记录")
        print(f"  - ... 还有69个交易对")
        

def main():
    """主函数"""
    print("\n🚀 CSV数据读取器演示")
    
    reader = CSVDataReader()
    
    # 分析资金费率
    reader.analyze_funding_rate()
    
    # 分析持仓量
    reader.analyze_open_interest()
    
    # 显示parquet文件信息
    reader.create_sample_parquet_info()
    
    print("\n✅ 分析完成！")


if __name__ == "__main__":
    main()