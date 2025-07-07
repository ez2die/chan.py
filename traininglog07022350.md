# btc_swap_1h 样本数据更新后的模型优化日志 (2025-07-02 23:50)

## 1️⃣ 样本与特征
- 数据集：`btc_swap_1h`（61 864 train / 15 448 val）
- 全特征：399 列；精选特征集：Top-200（基于 Phase-1 重要性排序）
- 缓存策略：按 `dataset + 特征列表 hash` 写入 `data/processed/features/*.parquet`

## 2️⃣ 优化迭代
| 阶段 | 内容 | 主要参数 | 验证 AUC |
|------|------|---------|---------|
| 0 | 基准 – LGB (默认) + 全 399 特征 | learning_rate=0.1, 100 iter | 0.5277 |
| 1 | HPO (Time-CV 100T) + 全 399 | lr=0.161, 319 iter | 0.5252 ↓ |
| 2 | Top-200 + 旧 HPO | lr=0.133, 557 iter | 0.5282 ↑ |
| 3 | Grid (Top-200) <br/> lr ∈ {0.05,0.07}, iter ∈ {1 200,1 600} | **lr=0.07, iter=1 600** 最优 | **0.5294** |

## 3️⃣ 最终三模型 & 集成
目录：`models/btc_swap_lgb_top200_refined_20250702_234827/`

| 模型 | 核心参数 (截取) | Val AUC |
|------|----------------|--------|
| LightGBM | lr=0.07, 1 600 iter, num_leaves=107 | 0.5294 |
| XGBoost | 默认 Phase-2 配置 | 0.5371 |
| CatBoost | depth=6, lr=0.05, 600 iter | 0.5411 |
| **Ensemble (权重优化)** | w Cat=0.46, XGB=0.31, LGB=0.23 | **0.5404** |

## 4️⃣ 结论
1. **CatBoost 单模最佳**，启用后显著提升 Ensemble。
2. Top-200 特征优于全特征；Top-300 / Top-399 会稀释效果。
3. Ensemble 权重经 `optimize_weights()` 自动搜索，可维持 >0.54 AUC。

## 5️⃣ 下一步建议
- CatBoost 微调 `depth / l2_leaf_reg / bagging_temperature`，目标 0.55+。
- 尝试 Stacking 二级模型 (Logit)；或加 DART-LGB 作为第 4 模型。
- 扩充特征：资金费率、OI、波动率分解，以突破 0.58 目标。

## 6️⃣ 快速应用指引
```python
from ModelStrategy.ChanMLTrainer import ChanMLTrainer
cfg = {
    'data_dir': 'data/ml_datasets',
    'model_dir': 'models/btc_swap_lgb_top200_refined_20250702_234827',
    'log_dir':  'logs',
    'dataset_name': 'btc_swap_1h',
    'target_column': 'binary_direction',
    'models': ['lgb','xgb','cat'],
    'selected_features_path': 'enhanced_features_list.txt',
}
trainer = ChanMLTrainer(cfg)
# trainer.run_full_training_pipeline()  # 重新训练
# 或直接加载模型文件进行预测
``` 

## 7️⃣ 2025-07-05 最新 Top-100 + 数据泄漏修复实验

- 数据集：`btc_swap_1h_swaponly`（train: 35,040，val: 8,760，2020-07-03 ~ 2025-07-02）
- 特征集：Top-100（CatBoost+LGB重要性融合，见 mixed_features_top100_20250704_164132.txt）
- 标签：未来24小时收益率，`label_horizon=24`，二分类 `binary_direction`
- 数据切分：严格无泄漏，train/val 间加 purge_gap=24h

### CatBoost HPO 最优参数（AUC 0.5618）
```yaml
iterations: 1518
learning_rate: 0.03275
depth: 10
l2_leaf_reg: 3.07
bagging_temperature: 0.0238
random_strength: 0.3949
subsample: 0.7156
early_stopping_rounds: 200
```
- 验证集 AUC: **0.5618**

### LightGBM HPO 最优参数（AUC ≈ 0.556）
```yaml
num_leaves: 55
max_depth: 6
learning_rate: 0.0864
feature_fraction: 0.72
bagging_fraction: 0.77
bagging_freq: 4
min_child_samples: 277
min_child_weight: 0.012
reg_alpha: 0.71
reg_lambda: 0.24
num_iterations: 485
```
- 验证集 AUC: **0.556**

### XGBoost HPO 最优参数（AUC ≈ 0.5533）
```yaml
learning_rate: 0.1109
max_depth: 10
min_child_weight: 4.77
subsample: 0.601
colsample_bytree: 0.841
gamma: 1.894
reg_alpha: 3.40
reg_lambda: 3.42
n_estimators: 411
```
- 验证集 AUC: **0.5533**

### 集成模型表现
- 权重平均/Stacking（LGB+XGB+Cat）均未超越 CatBoost 单模，最佳集成 AUC ≈ 0.52
- 主要瓶颈：CatBoost单模波动、LGB过拟合、Stacking样本量有限

### 结论
- **CatBoost 单模（Top-100）为当前最佳，AUC 0.5618**
- 数据泄漏彻底修复后，模型性能更真实可靠
- 后续建议：
  1. 继续 CatBoost 微调 depth/l2_leaf_reg/bagging_temperature
  2. 新特征工程（资金费率、OI、波动率等）
  3. 多时间窗/滚动验证提升稳健性 