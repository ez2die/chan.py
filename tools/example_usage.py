#!/usr/bin/env python3
"""
Chan.py数据转换工具使用示例
演示如何使用转换工具和验证工具
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from tools.data_converter import ChanDataConverter
from tools.data_validator import ChanDataValidator


def example_single_file_conversion():
    """示例：转换单个文件"""
    print("=" * 60)
    print("示例1: 转换单个文件")
    print("=" * 60)
    
    converter = ChanDataConverter()
    
    # 检查文件是否存在
    input_file = "data/processed/BTC_USDT_1h_processed.parquet"
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return
    
    # 转换数据
    print("加载数据...")
    df = converter.load_data(input_file)
    print(f"数据形状: {df.shape}")
    
    print("转换格式...")
    chan_data = converter.convert_to_chan_format(df, "BTC_USDT", "1h")
    
    # 保存数据
    output_file = "data/chan_format/example_BTC_USDT_1h_chan.json"
    metadata = {
        'symbol': 'BTC_USDT',
        'kl_type': '1h',
        'source_file': input_file,
        'record_count': len(chan_data),
        'example': True
    }
    
    converter.save_chan_data(chan_data, output_file, metadata)
    print(f"转换完成，输出文件: {output_file}")


def example_load_chan_data():
    """示例：加载chan.py格式数据"""
    print("=" * 60)
    print("示例2: 加载chan.py格式数据")
    print("=" * 60)
    
    converter = ChanDataConverter()
    
    # 检查文件是否存在
    chan_file = "data/chan_format/BTC_USDT_1h_chan.json"
    if not os.path.exists(chan_file):
        print(f"文件不存在: {chan_file}")
        return
    
    # 加载数据为CKLine_Unit对象
    print("加载chan.py格式数据...")
    kline_units = converter.load_chan_data(chan_file)
    
    print(f"加载了 {len(kline_units)} 个K线单元")
    
    # 显示前几个数据
    print("\n前3个K线数据:")
    for i, klu in enumerate(kline_units[:3]):
        print(f"  {i+1}. {klu}")
    
    print("\n后3个K线数据:")
    for i, klu in enumerate(kline_units[-3:], len(kline_units)-2):
        print(f"  {i}. {klu}")


def example_data_validation():
    """示例：数据验证"""
    print("=" * 60)
    print("示例3: 数据验证")
    print("=" * 60)
    
    validator = ChanDataValidator()
    
    # 验证单个文件
    chan_file = "data/chan_format/BTC_USDT_1h_chan.json"
    if not os.path.exists(chan_file):
        print(f"文件不存在: {chan_file}")
        return
    
    print("验证单个文件...")
    result = validator.validate_file(chan_file)
    
    print(f"文件: {os.path.basename(result['file_path'])}")
    print(f"状态: {'✓ 有效' if result['valid'] else '✗ 无效'}")
    print(f"记录数: {result['statistics']['record_count']}")
    print(f"错误数: {len(result['errors'])}")
    print(f"警告数: {len(result['warnings'])}")
    
    if result['statistics']['time_range']:
        time_range = result['statistics']['time_range']
        print(f"时间范围: {time_range['start']} ~ {time_range['end']}")
    
    if result['statistics']['price_range']:
        price_range = result['statistics']['price_range']
        print(f"价格范围: {price_range['min']:.2f} ~ {price_range['max']:.2f}")


def example_integration_with_chan():
    """示例：与chan.py框架集成"""
    print("=" * 60)
    print("示例4: 与chan.py框架集成")
    print("=" * 60)
    
    try:
        from Chan import CChan
        from Common.CEnum import KL_TYPE, DATA_SRC
        from DataAPI.CommonStockAPI import CCommonStockApi
        
        # 这里演示如何使用转换后的数据
        print("加载转换后的数据到chan.py框架...")
        
        converter = ChanDataConverter()
        chan_file = "data/chan_format/BTC_USDT_1h_chan.json"
        
        if os.path.exists(chan_file):
            # 加载数据
            kline_units = converter.load_chan_data(chan_file)
            print(f"加载了 {len(kline_units)} 个K线单元")
            
            # 这里可以进一步处理数据，如：
            # 1. 创建CChan对象
            # 2. 添加K线数据
            # 3. 进行缠论分析
            
            print("数据已准备就绪，可以进行缠论分析")
        else:
            print(f"文件不存在: {chan_file}")
            
    except ImportError as e:
        print(f"导入chan.py模块失败: {e}")
        print("请确保chan.py框架已正确安装")


def main():
    """主函数"""
    print("Chan.py数据转换工具使用示例")
    print("=" * 60)
    
    # 创建必要的目录
    os.makedirs("data/chan_format", exist_ok=True)
    
    # 运行示例
    example_single_file_conversion()
    print()
    
    example_load_chan_data()
    print()
    
    example_data_validation()
    print()
    
    example_integration_with_chan()
    print()
    
    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
    
    print("\n下一步操作建议:")
    print("1. 运行批量转换: python tools/batch_converter.py --type processed")
    print("2. 验证所有数据: python tools/data_validator.py --dir data/chan_format")
    print("3. 在你的策略中使用转换后的数据")


if __name__ == "__main__":
    main() 