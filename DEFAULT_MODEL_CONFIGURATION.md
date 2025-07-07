# Chan.py Phase-2 默认模型配置（2025-07-03）

> 本文件描述截至 2025-07-03 的 **Chan.py Phase-2** 默认训练配置，作为后续实验和部署的基准。

---

## 1. 数据集

| 项目 | 说明 |
| ---- | ---- |
| 数据源             | OKX Swap (BTC_USDT) |
| 周期               | 1h K 线 |
| 时间范围           | 2021-01-01 → 2024-12-31 |
| 文件               | `data/ml_datasets/btc_swap_1h_swaponly_train.json` / `…_val.json` |
| 记录数             | 训练 35 040 \| 验证 8 760 (共 43 800) |
| 重复时间戳         | **0** (已清洗) |
| 标签               | `binary_direction` (未来 1 根 K 线涨跌) |

---

## 2. 特征工程

| 项目 | 默认设置 |
| ---- | -------- |
| 计算器版本      | `EnhancedFeatureCalculator_v2` |
| 特征组开关      | `basic·technical·volume·chan_theory·multi_timeframe·volatility·momentum·pattern = True` |
| 白名单模式      | **启用** (`selected_features_only = True`) |
| 白名单文件      | `enhanced_features_list.txt` (排名好的 **Top-200** 特征) |
| 生成特征总数    | 200 + 6 基础列 = 206 列 |
| 特征缓存路径    | `data/processed/features/*_sel200_*.parquet` |

> 说明：白名单由 399 特征按平均重要性排序后截取前 200；实验结果表明与全量 399 特征性能相当但显著降低训练时间。

---

## 3. 训练器 (`ChanMLTrainer`)

```python
{
    "data_dir": "data/ml_datasets",
    "model_dir": "models",
    "log_dir"  : "logs",
    "dataset_name": "btc_swap_1h_swaponly",
    "target_column": "binary_direction",
    "test_size": 0.2,
    "random_state": 42,
    "models": ["xgb", "lgb", "cat"],
    "feature_calc_version": "v2",
    "selected_features_path": "enhanced_features_list.txt",
    "ensemble_method": "weighted_average",
    "model_params": {
        # 用户可覆盖，默认为各 ModelGenerator 内置参数
    }
}
```

---

## 4. 单模型默认超参数

### 4.1 LightGBM
* num_leaves=127, learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, min_child_samples=100, reg_alpha=0.1, reg_lambda=0.3, num_boost_round ≤ 1000 (early-stopping 20)

### 4.2 XGBoost
* max_depth=6, eta=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=1, reg_alpha=0.1, reg_lambda=1.0, num_boost_round ≤ 100 (early-stopping 10)

### 4.3 CatBoost
* iterations=300, depth=6, learning_rate=0.05, subsample=0.8, l2_leaf_reg=3, early_stopping_rounds=30

> 以上为 `ModelGenerator` 默认值；可在 `model_params` 中传入自定义 dict 进行覆盖。

---

## 5. 集成策略 (Ensemble)

* 方法：`weighted_average`
* 权重：训练后通过验证集 **AUC 最大化** 优化得到 (SLSQP 约束 w_i ≥0, Σw_i=1)。
* 典型权重示例：`{"lgb":0.40, "cat":0.35, "xgb":0.25}`（依实际训练动态调整）。

---

## 6. 性能基线 (2025-07-03)

| 模型 | Val AUC |
| ---- | ------- |
| LightGBM | 0.512 |
| XGBoost  | 0.507 |
| CatBoost | **0.536** |
| Ensemble | **0.534** |

> 采用 Top-200 特征的验证集指标；作为当前生产基线，后续优化以 **Ensemble AUC ≥ 0.58** 为目标。

---

## 7. 目录结构快照

```
├── enhanced_features_list.txt       # Top-200 排名文件
├── DEFAULT_MODEL_CONFIGURATION.md   # ← 当前文档
├── ModelStrategy/
│   ├── ChanMLTrainer.py            # 默认配置已更新
│   └── …
└── data/
    └── ml_datasets/
        ├── btc_swap_1h_swaponly_train.json
        └── btc_swap_1h_swaponly_val.json
```

---

## 8. 如何复现

```bash
# (可选) 创建虚拟环境并安装依赖
python -m venv .venv && source .venv/bin/activate
pip install -r Script/requirements.txt  # 或项目根 requirements

# 直接运行默认训练流程
python - << 'PY'
from ModelStrategy.ChanMLTrainer import ChanMLTrainer
trainer = ChanMLTrainer()        # 使用默认配置
trainer.run_full_training_pipeline()
PY
```

---

> **维护说明**：如需调整默认参数 / 数据集 / 白名单，请同时更新 `ChanMLTrainer._get_default_config()` 和本文档对应章节。 