#!/usr/bin/env python3
"""
批量数据转换脚本
用于批量转换data文件夹中的所有数据文件为chan.py标准格式
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from tools.data_converter import ChanDataConverter


def convert_processed_data():
    """转换processed文件夹中的数据"""
    converter = ChanDataConverter()
    
    input_dir = "data/processed"
    output_dir = "data/chan_format"
    
    print("=" * 60)
    print("批量转换processed数据文件")
    print("=" * 60)
    
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 批量转换
    converter.batch_convert(input_dir, output_dir, "*.parquet")


def convert_okx_spot_data():
    """转换OKX现货数据"""
    converter = ChanDataConverter()
    
    base_dir = "data/okx/spot"
    output_dir = "data/chan_format/okx_spot"
    
    print("=" * 60)
    print("批量转换OKX现货数据")
    print("=" * 60)
    
    if not os.path.exists(base_dir):
        print(f"输入目录不存在: {base_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有币种
    for symbol_dir in os.listdir(base_dir):
        symbol_path = os.path.join(base_dir, symbol_dir)
        if not os.path.isdir(symbol_path):
            continue
            
        print(f"处理币种: {symbol_dir}")
        
        # 遍历交易对
        for pair_dir in os.listdir(symbol_path):
            pair_path = os.path.join(symbol_path, pair_dir)
            if not os.path.isdir(pair_path):
                continue
                
            # 遍历时间级别
            for timeframe_dir in os.listdir(pair_path):
                timeframe_path = os.path.join(pair_path, timeframe_dir)
                if not os.path.isdir(timeframe_path):
                    continue
                
                print(f"  处理时间级别: {symbol_dir}_{pair_dir}_{timeframe_dir}")
                
                # 转换该时间级别的所有文件
                for file_name in os.listdir(timeframe_path):
                    if not file_name.endswith('.parquet'):
                        continue
                    
                    file_path = os.path.join(timeframe_path, file_name)
                    
                    try:
                        # 加载数据
                        df = converter.load_data(file_path)
                        
                        # 构建symbol名称
                        symbol = f"{symbol_dir}_{pair_dir}"
                        
                        # 转换数据
                        chan_data = converter.convert_to_chan_format(df, symbol, timeframe_dir)
                        
                        # 保存数据
                        year = file_name.replace('.parquet', '')
                        output_filename = f"{symbol}_{timeframe_dir}_{year}_chan.json"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        metadata = {
                            'symbol': symbol,
                            'kl_type': timeframe_dir,
                            'year': year,
                            'source_file': file_path,
                            'record_count': len(chan_data)
                        }
                        
                        converter.save_chan_data(chan_data, output_path, metadata)
                        
                    except Exception as e:
                        print(f"    转换失败 {file_path}: {e}")


def convert_okx_swap_data():
    """转换OKX合约数据"""
    converter = ChanDataConverter()
    
    base_dir = "data/okx/swap"
    output_dir = "data/chan_format/okx_swap"
    
    print("=" * 60)
    print("批量转换OKX合约数据")
    print("=" * 60)
    
    if not os.path.exists(base_dir):
        print(f"输入目录不存在: {base_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有币种
    for symbol_dir in os.listdir(base_dir):
        symbol_path = os.path.join(base_dir, symbol_dir)
        if not os.path.isdir(symbol_path):
            continue
            
        print(f"处理合约币种: {symbol_dir}")
        
        # 处理特殊格式的合约名称
        if symbol_dir.endswith('-SWAP'):
            # 处理 BTC-USDT-SWAP 格式
            for timeframe_dir in os.listdir(symbol_path):
                timeframe_path = os.path.join(symbol_path, timeframe_dir)
                if not os.path.isdir(timeframe_path):
                    continue
                
                print(f"  处理时间级别: {symbol_dir}_{timeframe_dir}")
                
                for file_name in os.listdir(timeframe_path):
                    if not file_name.endswith('.parquet'):
                        continue
                    
                    file_path = os.path.join(timeframe_path, file_name)
                    
                    try:
                        df = converter.load_data(file_path)
                        chan_data = converter.convert_to_chan_format(df, symbol_dir, timeframe_dir)
                        
                        year = file_name.replace('.parquet', '')
                        output_filename = f"{symbol_dir}_{timeframe_dir}_{year}_chan.json"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        metadata = {
                            'symbol': symbol_dir,
                            'kl_type': timeframe_dir,
                            'year': year,
                            'source_file': file_path,
                            'record_count': len(chan_data)
                        }
                        
                        converter.save_chan_data(chan_data, output_path, metadata)
                        
                    except Exception as e:
                        print(f"    转换失败 {file_path}: {e}")
        else:
            # 处理标准格式
            for pair_dir in os.listdir(symbol_path):
                pair_path = os.path.join(symbol_path, pair_dir)
                if not os.path.isdir(pair_path):
                    continue
                    
                for timeframe_dir in os.listdir(pair_path):
                    timeframe_path = os.path.join(pair_path, timeframe_dir)
                    if not os.path.isdir(timeframe_path):
                        continue
                    
                    print(f"  处理时间级别: {symbol_dir}_{pair_dir}_{timeframe_dir}")
                    
                    for file_name in os.listdir(timeframe_path):
                        if not file_name.endswith('.parquet'):
                            continue
                        
                        file_path = os.path.join(timeframe_path, file_name)
                        
                        try:
                            df = converter.load_data(file_path)
                            symbol = f"{symbol_dir}_{pair_dir}_SWAP"
                            chan_data = converter.convert_to_chan_format(df, symbol, timeframe_dir)
                            
                            year = file_name.replace('.parquet', '')
                            output_filename = f"{symbol}_{timeframe_dir}_{year}_chan.json"
                            output_path = os.path.join(output_dir, output_filename)
                            
                            metadata = {
                                'symbol': symbol,
                                'kl_type': timeframe_dir,
                                'year': year,
                                'source_file': file_path,
                                'record_count': len(chan_data)
                            }
                            
                            converter.save_chan_data(chan_data, output_path, metadata)
                            
                        except Exception as e:
                            print(f"    转换失败 {file_path}: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量转换数据为chan.py标准格式')
    parser.add_argument('--type', choices=['processed', 'spot', 'swap', 'all'], 
                       default='all', help='转换数据类型')
    
    args = parser.parse_args()
    
    print("Chan.py数据批量转换工具")
    print("=" * 60)
    
    if args.type in ['processed', 'all']:
        convert_processed_data()
        print()
    
    if args.type in ['spot', 'all']:
        convert_okx_spot_data()
        print()
    
    if args.type in ['swap', 'all']:
        convert_okx_swap_data()
        print()
    
    print("=" * 60)
    print("批量转换完成！")


if __name__ == "__main__":
    main() 