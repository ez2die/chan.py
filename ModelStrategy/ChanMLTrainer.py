import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .DatasetManager import ChanDatasetManager
from .FeatureCalculator import ChanFeatureCalculator
from .EnhancedFeatureCalculator import EnhancedChanFeatureCalculator
try:
    # Prefer v2 if present and user requests it via config
    from .EnhancedFeatureCalculator_v2 import (
        EnhancedChanFeatureCalculator_v2 as _EnhancedCalcV2,
    )
except ImportError:
    _EnhancedCalcV2 = None
from .models.XGBModelGenerator import CXGBModelGenerator
from .models.LightGBMModelGenerator import CLightGBMModelGenerator
from .models.CatBoostModelGenerator import CCatBoostModelGenerator
from .EnsemblePredictor import ChanEnsemblePredictor

class ChanMLTrainer:
    """Chan.py机器学习训练器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.dataset_manager = ChanDatasetManager(self.config['data_dir'])
        # 提前捕获各模型的超参数 dict
        self.model_params: Dict[str, Dict] = self.config.get('model_params', {})
        # 根据配置选择使用 v1 还是 v2 特征计算器
        calc_version = self.config.get("feature_calc_version", "v1")
        if calc_version == "v2" and _EnhancedCalcV2 is not None:
            self.feature_calculator = _EnhancedCalcV2()
        else:
            self.feature_calculator = EnhancedChanFeatureCalculator()
        # 若指定了精选特征列表，则读取
        self.selected_features: Optional[List[str]] = None
        sel_path = self.config.get('selected_features_path')
        if sel_path and os.path.exists(sel_path):
            with open(sel_path, 'r', encoding='utf-8') as fp:
                self.selected_features = [ln.strip() for ln in fp if ln.strip()]
        self.models = {}
        self.training_results = {}
        
        # 设置日志
        self._setup_logging()
        
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'data_dir': 'data/ml_datasets',
            'model_dir': 'models',
            'log_dir': 'logs',
            'dataset_name': 'btc_swap_1h_swaponly',
            'target_column': 'binary_direction',
            'test_size': 0.2,
            'random_state': 42,
            'models': ['xgb', 'lgb', 'cat'],
            'feature_selection': True,
            'cross_validation': True,
            'selected_features_path': 'enhanced_features_list.txt',
            'model_params': {},
            'feature_calc_version': 'v2',
            # 集成方式: 'weighted_average' 或 'stacking'
            'ensemble_method': 'weighted_average'
        }
        
    def _setup_logging(self):
        """设置日志"""
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config['log_dir']}/training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """准备训练数据"""
        self.logger.info("开始准备训练数据...")
        
        # -------- 1. 加载原始数据集（OHLCV） --------
        train_raw, _ = self.dataset_manager.load_training_dataset(
            self.config['dataset_name']
        )
        val_raw, _ = self.dataset_manager.load_validation_dataset(
            self.config['dataset_name']
        )

        # -------- 1.1  Purge gap to avoid label leakage --------
        horizon = self.config.get('label_horizon', 24)  # hours
        purge_gap = max(horizon, self.config.get('purge_gap_hours', horizon))

        if purge_gap > 0:
            # Drop last purge_gap rows from train and first purge_gap rows from val
            if len(train_raw) > purge_gap:
                train_raw = train_raw.iloc[:-purge_gap]
            if len(val_raw) > purge_gap:
                val_raw = val_raw.iloc[purge_gap:]
            self.logger.info(f"已应用 purge_gap={purge_gap}: train_rows={len(train_raw)}, val_rows={len(val_raw)}")

        self.logger.info(f"训练集大小: {len(train_raw)}")
        self.logger.info(f"验证集大小: {len(val_raw)}")

        # -------- 2. 构造缓存路径 --------
        cache_root = Path("data/processed/features")
        cache_root.mkdir(parents=True, exist_ok=True)

        if self.selected_features:
            # 计算精选特征散列避免不同列表冲突
            sel_hash = hashlib.md5(
                ",".join(sorted(self.selected_features)).encode()
            ).hexdigest()[:8]
            tag_suffix = f"_sel{len(self.selected_features)}_{sel_hash}"
        else:
            tag_suffix = "_all"

        train_cache = cache_root / f"{self.config['dataset_name']}_train{tag_suffix}.parquet"
        val_cache = cache_root / f"{self.config['dataset_name']}_val{tag_suffix}.parquet"

        # -------- 3. 读取或生成特征 --------
        if train_cache.exists() and val_cache.exists():
            self.logger.info("检测到特征缓存文件，直接加载…")
            train_features = pd.read_parquet(train_cache)
            val_features = pd.read_parquet(val_cache)
        else:
            self.logger.info("未找到缓存，开始计算特征…")
            # 特征工程
            train_features = self.feature_calculator.calculate_all_features(train_raw)
            val_features = self.feature_calculator.calculate_all_features(val_raw)

            # 若配置了精选特征，则仅保留这些特征 (基础价格列除外)
            if self.selected_features:
                base_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
                base_available = [c for c in base_cols if c in train_features.columns]
                keep_cols = base_available + [f for f in self.selected_features if f in train_features.columns]
                train_features = train_features[keep_cols]
                val_features = val_features[keep_cols]

            # ---- NEW: 移除非数值/布尔特征，避免 Parquet 写入及模型训练报错 ----
            non_numeric_cols = train_features.select_dtypes(exclude=[np.number, 'bool']).columns
            if len(non_numeric_cols) > 0:
                self.logger.warning(
                    f"检测到 {len(non_numeric_cols)} 个非数值特征, 将予以移除: {list(non_numeric_cols)[:10]}..."
                )
                train_features = train_features.drop(columns=non_numeric_cols)
                val_features = val_features.drop(columns=non_numeric_cols)

            # 保存缓存
            train_features.to_parquet(train_cache)
            val_features.to_parquet(val_cache)
            self.logger.info(f"特征缓存已写入 {cache_root}")
        
        # 创建标签
        train_with_labels = self.feature_calculator.create_labels(train_features)
        val_with_labels = self.feature_calculator.create_labels(val_features)
        
        # 清理数据
        train_clean = train_with_labels.dropna()
        val_clean = val_with_labels.dropna()
        
        # 替换无穷大值
        train_clean = train_clean.replace([np.inf, -np.inf], np.nan).dropna()
        val_clean = val_clean.replace([np.inf, -np.inf], np.nan).dropna()
        
        self.logger.info(f"清理后训练集大小: {len(train_clean)}")
        self.logger.info(f"清理后验证集大小: {len(val_clean)}")
        
        return train_clean, val_clean
        
    def train_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """训练模型"""
        self.logger.info("开始模型训练...")
        
        # 准备特征和标签
        feature_cols = self.feature_calculator.get_feature_names(train_df)
        target_col = self.config['target_column']
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values
        
        self.logger.info(f"特征维度: {len(feature_cols)}")
        self.logger.info(f"训练样本: {len(X_train)}, 正样本比例: {y_train.mean():.3f}")
        
        results = {}
        
        # 训练各种模型
        for model_type in self.config['models']:
            self.logger.info(f"开始训练 {model_type} 模型...")
            
            if model_type == 'xgb':
                model_generator = CXGBModelGenerator(
                    model_tag=f"btc_1h_{datetime.now().strftime('%Y%m%d')}",
                    xgb_params=self.model_params.get('xgb', self._get_xgb_params())
                )
                
                # 记录列顺序
                model_generator.feature_names = feature_cols
                
                # 创建数据集
                train_dataset = model_generator.create_data_set(X_train, y_train)
                val_dataset = model_generator.create_data_set(X_val, y_val)
                
                # 训练模型
                model_generator.train(train_dataset, val_dataset)
                
                # 保存模型
                model_generator.save_model()
                
                # 评估模型
                train_pred = model_generator.predict(train_dataset)
                val_pred = model_generator.predict(val_dataset)
                
                # 计算评估指标
                train_metrics = self._calculate_metrics(y_train, train_pred)
                val_metrics = self._calculate_metrics(y_val, val_pred)
                
                results[model_type] = {
                    'model': model_generator,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'feature_names': feature_cols,
                    'val_pred': val_pred
                }
                
                self.logger.info(f"{model_type} 训练完成:")
                self.logger.info(f"  训练集AUC: {train_metrics['auc']:.4f}")
                self.logger.info(f"  验证集AUC: {val_metrics['auc']:.4f}")
                
            elif model_type == 'lgb':
                model_generator = CLightGBMModelGenerator(
                    model_tag=f"btc_1h_{datetime.now().strftime('%Y%m%d')}",
                    lgb_params=self.model_params.get('lgb')
                )

                model_generator.feature_names = feature_cols

                train_dataset = model_generator.create_data_set(X_train, y_train)
                val_dataset = model_generator.create_data_set(X_val, y_val)

                model_generator.train(train_dataset, val_dataset)
                model_generator.save_model()

                train_pred = model_generator.predict(train_dataset)
                val_pred = model_generator.predict(val_dataset)

                train_metrics = self._calculate_metrics(y_train, train_pred)
                val_metrics = self._calculate_metrics(y_val, val_pred)

                results[model_type] = {
                    'model': model_generator,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'feature_names': feature_cols,
                    'val_pred': val_pred
                }

                self.logger.info(f"{model_type} 训练完成:")
                self.logger.info(f"  训练集AUC: {train_metrics['auc']:.4f}")
                self.logger.info(f"  验证集AUC: {val_metrics['auc']:.4f}")

            elif model_type == 'cat':
                model_generator = CCatBoostModelGenerator(
                    model_tag=f"btc_1h_{datetime.now().strftime('%Y%m%d')}",
                    cb_params=self.model_params.get('cat')
                )

                model_generator.feature_names = feature_cols

                train_dataset = model_generator.create_data_set(X_train, y_train)
                val_dataset = model_generator.create_data_set(X_val, y_val)

                model_generator.train(train_dataset, val_dataset)
                model_generator.save_model()

                train_pred = model_generator.predict(train_dataset)
                val_pred = model_generator.predict(val_dataset)

                train_metrics = self._calculate_metrics(y_train, train_pred)
                val_metrics = self._calculate_metrics(y_val, val_pred)

                results[model_type] = {
                    'model': model_generator,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'feature_names': feature_cols,
                    'val_pred': val_pred
                }

                self.logger.info(f"{model_type} 训练完成:")
                self.logger.info(f"  训练集AUC: {train_metrics['auc']:.4f}")
                self.logger.info(f"  验证集AUC: {val_metrics['auc']:.4f}")
                
        # -------------- 生成集成模型 ----------------
        if len(results) > 1:
            self.logger.info("生成加权平均集成模型...")

            predictor = ChanEnsemblePredictor(method=self.config.get('ensemble_method', 'weighted_average'))

            class _GenAdapter:
                def __init__(self, gen):
                    self.gen = gen
                def predict(self, X_np):
                    ds = self.gen.create_data_set(X_np)
                    return np.array(self.gen.predict(ds))

            predictions_dict = {}
            for mtype, info in results.items():
                predictor.add_model(mtype, _GenAdapter(info['model']))
                predictions_dict[mtype] = np.array(info['val_pred'])

            # ---- 支持固定权重配置 ----
            fixed_w = self.config.get('ensemble_weights') if isinstance(self.config.get('ensemble_weights'), dict) else None
            if predictor.ensemble_method == 'weighted_average':
                if fixed_w:
                    predictor.weights = {k: float(fixed_w.get(k, 1.0)) for k in predictor.models.keys()}
                else:
                    predictor.optimize_weights(predictions_dict, y_val)
            elif predictor.ensemble_method == 'stacking':
                # 使用验证集预测作为元特征训练 Logistic Regression 二级模型
                X_meta = np.column_stack(list(predictions_dict.values()))
                predictor.fit_meta_model(X_meta, y_val)
            else:
                raise ValueError(f"未知的ensemble_method: {predictor.ensemble_method}")

            ensemble_pred, _ = predictor.predict(X_val)
            ensemble_metrics = self._calculate_metrics(y_val, ensemble_pred)

            results['ensemble'] = {
                'predictor': predictor,
                'val_metrics': ensemble_metrics,
                'weights': predictor.weights,
                'method': predictor.ensemble_method
            }

            self.logger.info(
                f"集成模型完成: 验证集AUC {ensemble_metrics['auc']:.4f}")

        self.models = results
        
        # ---------- 生成特征重要性 CSV & 图表 (以 LightGBM Gain 为基) ----------
        try:
            lgb_gen = results.get('lgb', {}).get('model')
            if lgb_gen and hasattr(lgb_gen, 'get_feature_importance'):
                # 保证输出目录存在
                out_dir = Path(self.config['model_dir'])
                out_dir.mkdir(parents=True, exist_ok=True)

                imp_dict = lgb_gen.get_feature_importance()
                imp_series = pd.Series(imp_dict).sort_values(ascending=False)
                top50 = imp_series.head(50)
                out_csv = out_dir / f'feature_importance_top50_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                top50.to_csv(out_csv, header=['gain'])

                # 绘制条形图
                plt.figure(figsize=(10, 12))
                top50[::-1].plot(kind='barh')
                plt.title('Top-50 Feature Importance (Gain) - LightGBM')
                plt.tight_layout()
                plt.savefig(out_csv.with_suffix('.png'))
                plt.close()
                self.logger.info(f"特征重要性已保存 {out_csv} & PNG")
        except Exception as e:
            self.logger.warning(f"生成特征重要性失败: {e}")

        return results
        
    def _get_xgb_params(self) -> Dict:
        """获取XGBoost参数"""
        return {
            'max_depth': 6,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.config['random_state']
        }
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred) -> Dict:
        """计算评估指标"""
        if len(np.unique(y_true)) > 2:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
            proba = np.array(y_pred)
            if proba.ndim == 1:
                # degrade to binary metrics
                pred_labels = (proba > 0.5).astype(int)
                return {
                    'accuracy': accuracy_score(y_true, pred_labels),
                    'auc': roc_auc_score(y_true, proba)
                }
            pred_labels = proba.argmax(axis=1)
            return {
                'accuracy': accuracy_score(y_true, pred_labels),
                'f1_macro': f1_score(y_true, pred_labels, average='macro'),
                'auc_macro': roc_auc_score(y_true, proba, multi_class='ovr', average='macro'),
                'confusion': confusion_matrix(y_true, pred_labels).tolist()
            }
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
            y_pred_np = np.array(y_pred)
            if y_pred_np.ndim == 2 and y_pred_np.shape[1] == 2:
                prob_pos = y_pred_np[:, 1]
            else:
                prob_pos = y_pred_np.flatten()

            y_pred_binary = (prob_pos > 0.5).astype(int)
            return {
                'accuracy': accuracy_score(y_true, y_pred_binary),
                'precision': precision_score(y_true, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true, y_pred_binary, zero_division=0),
                'f1': f1_score(y_true, y_pred_binary, zero_division=0),
                'auc': roc_auc_score(y_true, prob_pos)
            }
        
    def run_full_training_pipeline(self) -> Dict:
        """运行完整训练流程"""
        self.logger.info("="*60)
        self.logger.info("开始Chan.py机器学习训练流程")
        self.logger.info("="*60)
        
        try:
            # 1. 准备数据
            train_df, val_df = self.prepare_training_data()
            
            # 2. 训练模型
            results = self.train_models(train_df, val_df)
            
            # 3. 保存结果
            self.save_training_results(results)
            
            # 4. 生成报告
            self.generate_training_report(results)
            
            # ------------------- 集成模型 -------------------
            # 集成部分已在 train_models 中完成
            
            self.logger.info("训练流程完成!")
            return results
            
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {e}")
            raise
            
    def save_training_results(self, results: Dict):
        """保存训练结果"""
        os.makedirs(self.config['model_dir'], exist_ok=True)
        
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 保存结果摘要
        summary = {}
        for model_type, result in results.items():
            if model_type == 'ensemble':
                summary[model_type] = {
                    'val_metrics': result['val_metrics'],
                    'weights': result['weights'],
                    'method': result['method']
                }
            else:
                summary[model_type] = {
                    'train_metrics': result['train_metrics'],
                    'val_metrics': result['val_metrics'],
                    'feature_count': len(result['feature_names'])
                }
            
        with open(f"{self.config['model_dir']}/training_summary_{ts_tag}.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"训练结果已保存到 training_summary_{ts_tag}.json")
        
    def generate_training_report(self, results: Dict):
        """生成训练报告"""
        report_lines = [
            "# Chan.py机器学习训练报告",
            f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 数据集信息",
            f"- 数据集: {self.config['dataset_name']}",
            f"- 目标列: {self.config['target_column']}",
            "",
            "## 模型性能"
        ]
        
        for model_type, result in results.items():
            if model_type == 'ensemble':
                val_metrics = result['val_metrics']
                report_lines.extend([
                    f"### ENSEMBLE ({result['method']})",
                    f"- 验证集AUC: {val_metrics['auc']:.4f}",
                    f"- 权重: {json.dumps(result['weights'])}",
                    ""
                ])
            else:
                val_metrics = result['val_metrics']
                report_lines.extend([
                    f"### {model_type.upper()} 模型",
                    f"- 验证集准确率: {val_metrics['accuracy']:.4f}",
                    f"- 验证集AUC: {val_metrics['auc']:.4f}",
                    f"- 验证集F1: {val_metrics['f1']:.4f}",
                    f"- 特征数量: {len(result['feature_names'])}",
                    ""
                ])
            
        report_content = "\n".join(report_lines)
        report_file = Path(self.config['model_dir']) / f"training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        self.logger.info(f"训练报告已保存到 {report_file}") 