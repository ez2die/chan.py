"""
数据转换器：将各种格式的数据转换为chan.py标准格式
支持parquet、CSV等格式的数据转换
"""

import os
import sys
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Common.CTime import CTime
from Common.CEnum import DATA_FIELD, KL_TYPE
from KLine.KLine_Unit import CKLine_Unit


class ChanDataConverter:
    """Chan.py数据转换器"""
    
    def __init__(self):
        self.supported_formats = ['parquet', 'csv']
        self.kl_type_mapping = {
            '1m': KL_TYPE.K_1M,
            '3m': KL_TYPE.K_3M,
            '5m': KL_TYPE.K_5M,
            '15m': KL_TYPE.K_15M,
            '30m': KL_TYPE.K_30M,
            '1h': KL_TYPE.K_60M,
            '1d': KL_TYPE.K_DAY,
            '1w': KL_TYPE.K_WEEK,
            '1M': KL_TYPE.K_MON,
        }
    
    def load_data(self, file_path: str, format_type: str = None) -> pd.DataFrame:
        """加载数据文件"""
        if format_type is None:
            format_type = file_path.split('.')[-1].lower()
        
        if format_type not in self.supported_formats:
            raise ValueError(f"不支持的数据格式: {format_type}")
        
        if format_type == 'parquet':
            return pd.read_parquet(file_path)
        elif format_type == 'csv':
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        
    def convert_datetime_to_ctime(self, dt) -> CTime:
        """将pandas datetime转换为CTime对象"""
        if pd.isna(dt):
            raise ValueError("无效的时间数据")
        
        # 处理时区信息
        if hasattr(dt, 'tz') and dt.tz is not None:
            dt = dt.tz_convert('UTC').tz_localize(None)
        
        return CTime(
            year=dt.year,
            month=dt.month,
            day=dt.day,
            hour=dt.hour,
            minute=dt.minute,
            second=dt.second,
            auto=True
        )
    
    def convert_row_to_kline_dict(self, row: pd.Series, timestamp) -> Dict[str, Any]:
        """将数据行转换为K线字典格式"""
        kline_dict = {
            DATA_FIELD.FIELD_TIME: self.convert_datetime_to_ctime(timestamp),
            DATA_FIELD.FIELD_OPEN: float(row['open']),
            DATA_FIELD.FIELD_HIGH: float(row['high']),
            DATA_FIELD.FIELD_LOW: float(row['low']),
            DATA_FIELD.FIELD_CLOSE: float(row['close']),
        }
        
        # 添加可选字段
        if 'volume' in row and pd.notna(row['volume']):
            kline_dict[DATA_FIELD.FIELD_VOLUME] = float(row['volume'])
        
        if 'turnover' in row and pd.notna(row['turnover']):
            kline_dict[DATA_FIELD.FIELD_TURNOVER] = float(row['turnover'])
            
        if 'turnover_rate' in row and pd.notna(row['turnover_rate']):
            kline_dict[DATA_FIELD.FIELD_TURNRATE] = float(row['turnover_rate'])
        
        return kline_dict
    
    def convert_to_chan_format(self, df: pd.DataFrame, symbol: str, kl_type: str) -> List[Dict[str, Any]]:
        """将DataFrame转换为chan.py标准格式"""
        result = []
        
        # 确保数据按时间排序
        df = df.sort_index()
        
        print(f"开始转换 {symbol} {kl_type} 数据，共 {len(df)} 条记录")
        
        for timestamp, row in df.iterrows():
            try:
                kline_dict = self.convert_row_to_kline_dict(row, timestamp)
                
                # 验证数据有效性
                if self.validate_kline_data(kline_dict):
                    result.append(kline_dict)
                else:
                    print(f"跳过无效数据: {timestamp}")
                    
            except Exception as e:
                print(f"转换数据时出错 {timestamp}: {e}")
                continue
        
        print(f"转换完成，有效数据 {len(result)} 条")
        return result
    
    def validate_kline_data(self, kline_dict: Dict[str, Any]) -> bool:
        """验证K线数据的有效性"""
        try:
            # 检查必需字段
            required_fields = [
                DATA_FIELD.FIELD_TIME,
                DATA_FIELD.FIELD_OPEN,
                DATA_FIELD.FIELD_HIGH,
                DATA_FIELD.FIELD_LOW,
                DATA_FIELD.FIELD_CLOSE
            ]
            
            for field in required_fields:
                if field not in kline_dict:
                    return False
            
            # 检查价格逻辑
            open_price = kline_dict[DATA_FIELD.FIELD_OPEN]
            high_price = kline_dict[DATA_FIELD.FIELD_HIGH]
            low_price = kline_dict[DATA_FIELD.FIELD_LOW]
            close_price = kline_dict[DATA_FIELD.FIELD_CLOSE]
            
            if not (low_price <= min(open_price, close_price) and 
                    high_price >= max(open_price, close_price)):
                return False
            
            # 检查价格为正数
            if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                return False
                
            return True
            
        except Exception:
            return False
    
    def save_chan_data(self, data: List[Dict[str, Any]], output_path: str, metadata: Dict[str, Any] = None):
        """保存转换后的数据"""
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 准备保存的数据结构
        save_data = {
            'metadata': metadata or {},
            'kline_data': []
        }
        
        # 转换CTime对象为可序列化格式
        for kline_dict in data:
            serializable_dict = {}
            for key, value in kline_dict.items():
                if key == DATA_FIELD.FIELD_TIME:
                    # 将CTime转换为字符串
                    serializable_dict[key] = value.to_str()
                else:
                    serializable_dict[key] = value
            save_data['kline_data'].append(serializable_dict)
        
        # 保存为JSON格式
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"数据已保存到: {output_path}")
    
    def load_chan_data(self, file_path: str) -> List[CKLine_Unit]:
        """加载chan.py格式的数据并创建CKLine_Unit对象"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kline_units = []
        for kline_dict in data['kline_data']:
            # 重新构建CTime对象
            time_str = kline_dict[DATA_FIELD.FIELD_TIME]
            if ' ' in time_str:
                date_part, time_part = time_str.split(' ')
                year, month, day = map(int, date_part.split('/'))
                hour, minute = map(int, time_part.split(':'))
            else:
                year, month, day = map(int, time_str.split('/'))
                hour, minute = 0, 0
            
            kline_dict[DATA_FIELD.FIELD_TIME] = CTime(year, month, day, hour, minute)
            
            # 创建CKLine_Unit对象
            kline_unit = CKLine_Unit(kline_dict, autofix=True)
            kline_units.append(kline_unit)
        
        return kline_units
    
    def batch_convert(self, input_dir: str, output_dir: str, pattern: str = "*.parquet"):
        """批量转换数据文件"""
        import glob
        
        files = glob.glob(os.path.join(input_dir, pattern))
        print(f"找到 {len(files)} 个文件待转换")
        
        for file_path in files:
            try:
                # 解析文件名获取symbol和kl_type
                filename = os.path.basename(file_path)
                parts = filename.replace('.parquet', '').split('_')
                
                if len(parts) >= 3:
                    symbol = f"{parts[0]}_{parts[1]}"  # 如 BTC_USDT
                    kl_type = parts[2]  # 如 1h
                    
                    # 加载数据
                    df = self.load_data(file_path)
                    
                    # 转换数据
                    chan_data = self.convert_to_chan_format(df, symbol, kl_type)
                    
                    # 保存数据
                    output_filename = f"{symbol}_{kl_type}_chan.json"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    metadata = {
                        'symbol': symbol,
                        'kl_type': kl_type,
                        'source_file': file_path,
                        'convert_time': datetime.now().isoformat(),
                        'record_count': len(chan_data)
                    }
                    
                    self.save_chan_data(chan_data, output_path, metadata)
                    
                else:
                    print(f"无法解析文件名格式: {filename}")
                    
            except Exception as e:
                print(f"转换文件失败 {file_path}: {e}")


def main():
    """主函数 - 示例用法"""
    converter = ChanDataConverter()
    
    # 示例：转换单个文件
    input_file = "data/processed/BTC_USDT_1h_processed.parquet"
    output_file = "data/chan_format/BTC_USDT_1h_chan.json"
    
    if os.path.exists(input_file):
        print("开始转换数据...")
        df = converter.load_data(input_file)
        chan_data = converter.convert_to_chan_format(df, "BTC_USDT", "1h")
        
        metadata = {
            'symbol': 'BTC_USDT',
            'kl_type': '1h',
            'source_file': input_file,
            'convert_time': datetime.now().isoformat(),
            'record_count': len(chan_data)
        }
        
        converter.save_chan_data(chan_data, output_file, metadata)
        print("转换完成！")
    else:
        print(f"文件不存在: {input_file}")


if __name__ == "__main__":
    main() 