# Chan.py 数据预处理工具包

这个工具包用于将各种格式的历史数据转换为chan.py框架的标准格式，确保数据的一致性和可用性。

## 🛠️ 工具概览

### 1. `data_converter.py` - 数据转换器
核心转换模块，支持将parquet、CSV等格式的数据转换为chan.py标准格式。

### 2. `batch_converter.py` - 批量转换脚本
批量处理工具，可以一次性转换整个数据目录。

### 3. `data_validator.py` - 数据验证器
验证转换后数据的质量和完整性。

## 📋 系统要求

- Python 3.7+
- pandas
- pyarrow
- chan.py框架

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install pandas pyarrow
```

### 2. 转换单个文件

```python
from tools.data_converter import ChanDataConverter

converter = ChanDataConverter()

# 加载数据
df = converter.load_data("data/processed/BTC_USDT_1h_processed.parquet")

# 转换为chan.py格式
chan_data = converter.convert_to_chan_format(df, "BTC_USDT", "1h")

# 保存数据
metadata = {
    'symbol': 'BTC_USDT',
    'kl_type': '1h',
    'record_count': len(chan_data)
}
converter.save_chan_data(chan_data, "output/BTC_USDT_1h_chan.json", metadata)
```

### 3. 批量转换

```bash
# 转换所有数据
python tools/batch_converter.py --type all

# 只转换processed数据
python tools/batch_converter.py --type processed

# 只转换现货数据
python tools/batch_converter.py --type spot

# 只转换合约数据
python tools/batch_converter.py --type swap
```

### 4. 数据验证

```bash
# 验证转换后的数据
python tools/data_validator.py --dir data/chan_format

# 指定报告输出路径
python tools/data_validator.py --dir data/chan_format --output validation_report.txt
```

## 📊 数据格式说明

### 输入格式要求

支持的输入格式：
- **Parquet文件**: 包含OHLCV数据的parquet文件
- **CSV文件**: 包含时间索引的CSV文件

必需字段：
- `open`: 开盘价
- `high`: 最高价  
- `low`: 最低价
- `close`: 收盘价

可选字段：
- `volume`: 成交量
- `turnover`: 成交额
- `turnover_rate`: 换手率

### 输出格式

转换后的JSON文件结构：
```json
{
  "metadata": {
    "symbol": "BTC_USDT",
    "kl_type": "1h",
    "source_file": "原始文件路径",
    "convert_time": "转换时间",
    "record_count": 记录数量
  },
  "kline_data": [
    {
      "time_key": "2024/01/01 00:00",
      "open": 42284.0,
      "high": 42551.7,
      "low": 42259.7,
      "close": 42472.1,
      "volume": 1234.56
    }
  ]
}
```

## 🔄 转换流程

1. **数据加载**: 从parquet/CSV文件读取数据
2. **格式转换**: 转换为chan.py标准字典格式
3. **数据验证**: 检查价格逻辑和数据完整性
4. **时间处理**: 将pandas datetime转换为CTime格式
5. **数据保存**: 保存为JSON格式，包含元数据

## 📁 目录结构

转换后的数据将按以下结构组织：

```
data/chan_format/
├── BTC_USDT_1h_chan.json          # processed数据转换结果
├── BTC_USDT_1d_chan.json
├── okx_spot/                       # OKX现货数据
│   ├── BTC_USDT_1h_2024_chan.json
│   └── ETH_USDT_1h_2024_chan.json
└── okx_swap/                       # OKX合约数据
    ├── BTC-USDT-SWAP_1h_2025_chan.json
    └── BTC_USDT_SWAP_1d_2024_chan.json
```

## ⚙️ 配置选项

### 时间级别映射

```python
kl_type_mapping = {
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
```

### 数据验证规则

- **价格逻辑**: low ≤ min(open, close), high ≥ max(open, close)
- **正数检查**: 所有价格必须为正数
- **时间连续性**: 检查重复时间戳
- **异常波动**: 标记超过50%的价格波动

## 🐛 故障排除

### 常见问题

1. **ModuleNotFoundError**: 确保已安装所有依赖包
2. **文件路径错误**: 检查输入文件是否存在
3. **数据格式错误**: 确保输入数据包含必需字段
4. **内存不足**: 对于大文件，考虑分批处理

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据结构
df = pd.read_parquet("your_file.parquet")
print(df.columns)
print(df.head())
print(df.dtypes)
```

## 📈 性能优化

- **批量处理**: 使用batch_converter.py处理大量文件
- **内存管理**: 转换器自动处理内存优化
- **并行处理**: 未来版本将支持多进程转换

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

本项目采用与chan.py主项目相同的许可证。

## 📞 支持

如有问题或建议，请：
1. 查看本文档的故障排除部分
2. 检查现有的Issues
3. 创建新的Issue描述问题

---

**注意**: 转换过程中会自动验证数据质量，无效数据将被跳过并记录在日志中。建议转换完成后运行数据验证工具确保数据质量。 