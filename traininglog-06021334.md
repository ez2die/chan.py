# Chan.py Phase-2 超参数调优报告（截至 2025-07-02）

## 1. 数据与评估设置  
* 资产 / 时间粒度  : BTC-USDT，1h K 线  
* 训练 / 验证样本量 : 21 076 / 5 252  
* 目标标签     : `binary_direction`（下一根 K 线涨跌）  
* 评估指标     : ROC-AUC（越高越好）  
* 特征集      :  
  * Full-399  – Phase-1 修复泄漏后 399 个特征  
  * Top-200  – FeatureSelector 下选 200 个  
  * Top-300  – FeatureSelector 下选 300 个（当前主力集）  
* 训练流程     : ChanMLTrainer → 早停 20 rds（LGB）/30 iters（Cat）  
* 随机种子     : 42  

---

## 2. 基线结果（未调参）

| 特征集 | 模型 | Val-AUC | 备注 |
| ------ | ---- | ------- | ---- |
| Full-399 | XGBoost | 0.490 | 深度 6，1000 rds |
|          | LightGBM | 0.520 | leaf-wise，lr 0.05 |
|          | CatBoost | 0.533 | depth 6，lr 0.1 |
|          | Equal Ensemble | 0.501 | 三模型等权平均 |
|          | Weight-Opt Ensemble | 0.542 | SLSQP 优化权重 |
|          | Logistic Stacking | **0.553** | 三模型 LR 元模型 |

> 记录：Phase-1 单模型严重过拟合（AUC 0.47）。引入集成后突破 0.55。

---

## 3. 特征选择影响

| 特征集 | XGB | LGB | Cat | Equal Ensemble | 说明 |
| ------ | --- | --- | --- | -------------- | ---- |
| Top-200 | **0.501** ↑ | 0.507 ↓ | 0.512 ↑ | 0.503 | 信息密度提升，但 LGB 降 |
| Top-300 | 0.490 | 0.520 | 0.533 | 0.501 | 与 Full-399 接近 |

结论：Top-300 保留信息量、维度适中；成为后续调参主力。

---

## 4. Hyperparameter Tuning 过程

### 4.1 LightGBM  
* Optuna 50 Trials（cv=3）  
* 初始搜索空间：  
  * num_leaves 31-255、learning_rate 0.01-0.3、num_iterations 固定 1000  
* 扩展后搜索空间：  
  * num_leaves 64-256、learning_rate 0.01-0.3、num_iterations 500-1000  
* 最佳参数 (Trial-best)：  
  ```json
  {
    "num_leaves": 136,
    "max_depth": 16,
    "learning_rate": 0.1206,
    "feature_fraction": 0.839,
    "bagging_fraction": 0.662,
    "bagging_freq": 2,
    "min_child_samples": 36,
    "min_child_weight": 2.915,
    "reg_alpha": 1.202,
    "reg_lambda": 1.416,
    "num_iterations": 510
  }
  ```
* 验证 AUC：0.5165 （+0.002 vs 旧 0.5149，+0.-009 vs Baseline 0.520）

### 4.2 CatBoost  
* Optuna 50 Trials（cv=3）  
* 搜索空间：depth 5-10、lr 0.01-0.3、l2 0-10、bagging_temperature 0-1、subsample 0.6-1 等  
* 最佳参数：  
  ```json
  {
    "depth": 8,
    "learning_rate": 0.0277,
    "l2_leaf_reg": 1.358,
    "bagging_temperature": 0.911,
    "random_strength": 9.942,
    "subsample": 0.919
  }
  ```
* 验证 AUC：0.5326 （+0.0018 vs 旧 0.5308，≈ Baseline 0.533）  
* 迭代在 20 iters 早停，存在继续提升空间。

### 4.3 结果汇总（Top-300 特征）

| 阶段 | LGB | Cat | Equal Ensemble | 备注 |
| ---- | --- | --- | -------------- | ---- |
| Baseline params | 0.5203 | 0.5329 | 0.5008 | |
| Tuned-v1 (n_trials 75) | 0.5149 | 0.5308 | 0.5318 | 小幅波动 |
| Tuned-v2 (expanded space) | 0.5165 | 0.5326 | **0.5343** | 当前最优两模集成 |

---

## 5. 当前最佳方案

* 特征    : Top-300  
* 模型    : CatBoost (tuned) + LightGBM (tuned)  
* 集成方式  : 权重优化加权平均（≈ 0.24 / 0.76）  
* Val-AUC   : **0.5343**

> 仍未超越三模型 Logistic Stacking 0.553；Cat/LGB 调优收益趋于递减。

---

## 6. 后续改进方向

1. CatBoost  
   * 放宽 early_stopping_rounds→50，iterations→2000，细化 lr 0.01-0.05。  
2. LightGBM  
   * 更低 lr（0.01-0.05）+ higher num_iterations 1000-3000。  
   * 调整 min_child_samples / reg_* 进一步防过拟合。  
3. XGBoost  
   * 参照 Cat/LGB 进行 Optuna 全局搜索，引入正则化 (gamma, lambda).  
4. 集成策略  
   * 三模型权重优化、Logistic Stacking 重新训练（5-fold 时间序列交叉验证）。  
5. 其它  
   * 采用 time-based CV 评估稳健性。  
   * 试验 Stratified GroupKFold 消除时间泄漏。  

---

## 7. 输出物

| 文件 | 说明 |
| ---- | ---- |
| models/hpo_top300_v2.json | LGB/Cat 最佳超参 |
| models/feature_top300_hpo_run_v2/* | 最新模型、元数据、训练报告 |
| logs/feature_top300_hpo_run_v2/* | 训练日志 |

---

**当前阶段成绩单**  
* 单模型最佳 AUC : CatBoost 0.533  
* 两模型集成   : 0.534  
* 三模型 stacking : **0.553** (阶段峰值)  

下一目标：> 0.58（Phase-2 KPI）。