"""
数据验证工具：验证转换后的chan.py格式数据的质量和完整性
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from Common.CTime import CTime
from Common.CEnum import DATA_FIELD
from tools.data_converter import ChanDataConverter


class ChanDataValidator:
    """Chan.py数据验证器"""
    
    def __init__(self):
        self.converter = ChanDataConverter()
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """验证单个数据文件"""
        result = {
            'file_path': file_path,
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # 加载数据
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证文件结构
            if 'metadata' not in data:
                result['errors'].append("缺少metadata字段")
            if 'kline_data' not in data:
                result['errors'].append("缺少kline_data字段")
                return result
            
            kline_data = data['kline_data']
            metadata = data.get('metadata', {})
            
            # 基本统计信息
            result['statistics'] = {
                'record_count': len(kline_data),
                'time_range': self._get_time_range(kline_data),
                'price_range': self._get_price_range(kline_data),
                'metadata': metadata
            }
            
            # 验证数据记录
            validation_result = self._validate_kline_records(kline_data)
            result['errors'].extend(validation_result['errors'])
            result['warnings'].extend(validation_result['warnings'])
            
            # 验证时间连续性
            time_validation = self._validate_time_continuity(kline_data)
            result['errors'].extend(time_validation['errors'])
            result['warnings'].extend(time_validation['warnings'])
            
            # 验证价格合理性
            price_validation = self._validate_price_logic(kline_data)
            result['errors'].extend(price_validation['errors'])
            result['warnings'].extend(price_validation['warnings'])
            
            result['valid'] = len(result['errors']) == 0
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"文件读取错误: {str(e)}")
        
        return result
    
    def _validate_kline_records(self, kline_data: List[Dict]) -> Dict[str, List]:
        """验证K线记录的完整性"""
        errors = []
        warnings = []
        
        required_fields = [
            DATA_FIELD.FIELD_TIME,
            DATA_FIELD.FIELD_OPEN,
            DATA_FIELD.FIELD_HIGH,
            DATA_FIELD.FIELD_LOW,
            DATA_FIELD.FIELD_CLOSE
        ]
        
        for i, record in enumerate(kline_data):
            # 检查必需字段
            for field in required_fields:
                if field not in record:
                    errors.append(f"记录 {i}: 缺少必需字段 {field}")
            
            # 检查数据类型
            try:
                if DATA_FIELD.FIELD_OPEN in record:
                    float(record[DATA_FIELD.FIELD_OPEN])
                if DATA_FIELD.FIELD_HIGH in record:
                    float(record[DATA_FIELD.FIELD_HIGH])
                if DATA_FIELD.FIELD_LOW in record:
                    float(record[DATA_FIELD.FIELD_LOW])
                if DATA_FIELD.FIELD_CLOSE in record:
                    float(record[DATA_FIELD.FIELD_CLOSE])
            except (ValueError, TypeError):
                errors.append(f"记录 {i}: 价格数据类型错误")
            
            # 检查价格逻辑
            try:
                open_price = float(record[DATA_FIELD.FIELD_OPEN])
                high_price = float(record[DATA_FIELD.FIELD_HIGH])
                low_price = float(record[DATA_FIELD.FIELD_LOW])
                close_price = float(record[DATA_FIELD.FIELD_CLOSE])
                
                if low_price > min(open_price, close_price):
                    errors.append(f"记录 {i}: 最低价 {low_price} 大于开盘价或收盘价")
                
                if high_price < max(open_price, close_price):
                    errors.append(f"记录 {i}: 最高价 {high_price} 小于开盘价或收盘价")
                
                if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                    errors.append(f"记录 {i}: 存在非正数价格")
                    
            except (KeyError, ValueError, TypeError):
                pass  # 已在上面检查过
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_time_continuity(self, kline_data: List[Dict]) -> Dict[str, List]:
        """验证时间连续性"""
        errors = []
        warnings = []
        
        if len(kline_data) < 2:
            return {'errors': errors, 'warnings': warnings}
        
        time_gaps = []
        duplicate_times = []
        
        prev_time_str = None
        for i, record in enumerate(kline_data):
            if DATA_FIELD.FIELD_TIME not in record:
                continue
            
            time_str = record[DATA_FIELD.FIELD_TIME]
            
            if prev_time_str and time_str == prev_time_str:
                duplicate_times.append(f"记录 {i}: 重复时间 {time_str}")
            
            prev_time_str = time_str
        
        if duplicate_times:
            errors.extend(duplicate_times[:10])  # 只报告前10个
            if len(duplicate_times) > 10:
                warnings.append(f"发现 {len(duplicate_times)} 个重复时间，仅显示前10个")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_price_logic(self, kline_data: List[Dict]) -> Dict[str, List]:
        """验证价格逻辑和异常值"""
        errors = []
        warnings = []
        
        prices = []
        for record in kline_data:
            try:
                close_price = float(record[DATA_FIELD.FIELD_CLOSE])
                prices.append(close_price)
            except (KeyError, ValueError, TypeError):
                continue
        
        if not prices:
            return {'errors': errors, 'warnings': warnings}
        
        # 检查异常波动
        for i in range(1, len(prices)):
            if len(prices) > i:
                change_rate = abs(prices[i] - prices[i-1]) / prices[i-1]
                if change_rate > 0.5:  # 50%的异常波动
                    warnings.append(f"记录 {i}: 价格异常波动 {change_rate:.2%}")
        
        # 检查价格范围
        min_price = min(prices)
        max_price = max(prices)
        if max_price / min_price > 100:  # 价格区间过大
            warnings.append(f"价格区间过大: {min_price:.2f} - {max_price:.2f}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _get_time_range(self, kline_data: List[Dict]) -> Dict[str, str]:
        """获取时间范围"""
        if not kline_data:
            return {'start': 'N/A', 'end': 'N/A'}
        
        times = []
        for record in kline_data:
            if DATA_FIELD.FIELD_TIME in record:
                times.append(record[DATA_FIELD.FIELD_TIME])
        
        if not times:
            return {'start': 'N/A', 'end': 'N/A'}
        
        return {
            'start': min(times),
            'end': max(times),
            'count': len(times)
        }
    
    def _get_price_range(self, kline_data: List[Dict]) -> Dict[str, float]:
        """获取价格范围"""
        if not kline_data:
            return {'min': 0, 'max': 0, 'avg': 0}
        
        prices = []
        for record in kline_data:
            try:
                close_price = float(record[DATA_FIELD.FIELD_CLOSE])
                prices.append(close_price)
            except (KeyError, ValueError, TypeError):
                continue
        
        if not prices:
            return {'min': 0, 'max': 0, 'avg': 0}
        
        return {
            'min': min(prices),
            'max': max(prices),
            'avg': sum(prices) / len(prices)
        }
    
    def batch_validate(self, directory: str) -> Dict[str, Any]:
        """批量验证目录中的所有数据文件"""
        results = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'file_results': [],
            'summary': {
                'total_records': 0,
                'total_errors': 0,
                'total_warnings': 0
            }
        }
        
        if not os.path.exists(directory):
            results['error'] = f"目录不存在: {directory}"
            return results
        
        # 查找所有JSON文件
        json_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('_chan.json'):
                    json_files.append(os.path.join(root, file))
        
        results['total_files'] = len(json_files)
        
        print(f"开始验证 {len(json_files)} 个数据文件...")
        
        for file_path in json_files:
            print(f"验证: {os.path.basename(file_path)}")
            
            file_result = self.validate_file(file_path)
            results['file_results'].append(file_result)
            
            if file_result['valid']:
                results['valid_files'] += 1
            else:
                results['invalid_files'] += 1
            
            # 累计统计
            results['summary']['total_records'] += file_result['statistics'].get('record_count', 0)
            results['summary']['total_errors'] += len(file_result['errors'])
            results['summary']['total_warnings'] += len(file_result['warnings'])
        
        return results
    
    def generate_report(self, validation_results: Dict[str, Any], output_path: str = None):
        """生成验证报告"""
        report_lines = []
        
        # 标题
        report_lines.append("=" * 80)
        report_lines.append("Chan.py数据验证报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 总体统计
        report_lines.append("总体统计:")
        report_lines.append(f"  总文件数: {validation_results['total_files']}")
        report_lines.append(f"  有效文件: {validation_results['valid_files']}")
        report_lines.append(f"  无效文件: {validation_results['invalid_files']}")
        report_lines.append(f"  总记录数: {validation_results['summary']['total_records']}")
        report_lines.append(f"  总错误数: {validation_results['summary']['total_errors']}")
        report_lines.append(f"  总警告数: {validation_results['summary']['total_warnings']}")
        report_lines.append("")
        
        # 文件详情
        if validation_results['file_results']:
            report_lines.append("文件验证详情:")
            report_lines.append("-" * 80)
            
            for file_result in validation_results['file_results']:
                filename = os.path.basename(file_result['file_path'])
                status = "✓ 有效" if file_result['valid'] else "✗ 无效"
                
                report_lines.append(f"文件: {filename} - {status}")
                
                stats = file_result['statistics']
                report_lines.append(f"  记录数: {stats.get('record_count', 0)}")
                
                if 'time_range' in stats:
                    time_range = stats['time_range']
                    report_lines.append(f"  时间范围: {time_range.get('start', 'N/A')} ~ {time_range.get('end', 'N/A')}")
                
                if 'price_range' in stats:
                    price_range = stats['price_range']
                    report_lines.append(f"  价格范围: {price_range.get('min', 0):.2f} ~ {price_range.get('max', 0):.2f}")
                
                if file_result['errors']:
                    report_lines.append(f"  错误 ({len(file_result['errors'])}):")
                    for error in file_result['errors'][:5]:  # 只显示前5个错误
                        report_lines.append(f"    - {error}")
                    if len(file_result['errors']) > 5:
                        report_lines.append(f"    ... 还有 {len(file_result['errors']) - 5} 个错误")
                
                if file_result['warnings']:
                    report_lines.append(f"  警告 ({len(file_result['warnings'])}):")
                    for warning in file_result['warnings'][:3]:  # 只显示前3个警告
                        report_lines.append(f"    - {warning}")
                    if len(file_result['warnings']) > 3:
                        report_lines.append(f"    ... 还有 {len(file_result['warnings']) - 3} 个警告")
                
                report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"验证报告已保存到: {output_path}")
        
        return report_content


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='验证chan.py格式数据')
    parser.add_argument('--dir', default='data/chan_format', help='数据目录')
    parser.add_argument('--output', help='报告输出路径')
    
    args = parser.parse_args()
    
    validator = ChanDataValidator()
    
    print("开始验证数据...")
    results = validator.batch_validate(args.dir)
    
    output_path = args.output or f"data/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report = validator.generate_report(results, output_path)
    
    print("\n" + report)


if __name__ == "__main__":
    main() 