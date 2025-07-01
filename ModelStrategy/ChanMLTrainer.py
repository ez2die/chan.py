import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

from .DatasetManager import ChanDatasetManager
from .FeatureCalculator import ChanFeatureCalculator
from .EnhancedFeatureCalculator import EnhancedChanFeatureCalculator
from .models.XGBModelGenerator import CXGBModelGenerator

class ChanMLTrainer:
    """Chan.py机器学习训练器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.dataset_manager = ChanDatasetManager(self.config['data_dir'])
        # 使用增强版特征计算器 (336个特征)
        self.feature_calculator = EnhancedChanFeatureCalculator()
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
            'dataset_name': 'main_training',
            'target_column': 'binary_direction',
            'test_size': 0.2,
            'random_state': 42,
            'models': ['xgb'],
            'feature_selection': True,
            'cross_validation': True
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
        
        # 加载数据集
        train_df, train_meta = self.dataset_manager.load_training_dataset(
            self.config['dataset_name']
        )
        val_df, val_meta = self.dataset_manager.load_validation_dataset(
            self.config['dataset_name']
        )
        
        self.logger.info(f"训练集大小: {len(train_df)}")
        self.logger.info(f"验证集大小: {len(val_df)}")
        
        # 特征工程
        self.logger.info("开始特征工程...")
        train_features = self.feature_calculator.calculate_all_features(train_df)
        val_features = self.feature_calculator.calculate_all_features(val_df)
        
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
                    xgb_params=self._get_xgb_params()
                )
                
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
                    'feature_names': feature_cols
                }
                
                self.logger.info(f"{model_type} 训练完成:")
                self.logger.info(f"  训练集AUC: {train_metrics['auc']:.4f}")
                self.logger.info(f"  验证集AUC: {val_metrics['auc']:.4f}")
                
        self.models = results
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
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # 转换预测结果为numpy数组
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        
        # 转换预测概率为类别
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary),
            'recall': recall_score(y_true, y_pred_binary),
            'f1': f1_score(y_true, y_pred_binary),
            'auc': roc_auc_score(y_true, y_pred)
        }
        
        return metrics
        
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
            
            self.logger.info("训练流程完成!")
            return results
            
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {e}")
            raise
            
    def save_training_results(self, results: Dict):
        """保存训练结果"""
        os.makedirs(self.config['model_dir'], exist_ok=True)
        
        # 保存结果摘要
        summary = {}
        for model_type, result in results.items():
            summary[model_type] = {
                'train_metrics': result['train_metrics'],
                'val_metrics': result['val_metrics'],
                'feature_count': len(result['feature_names'])
            }
            
        with open(f"{self.config['model_dir']}/training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"训练结果已保存到 {self.config['model_dir']}/training_summary.json")
        
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
        
        with open(f"{self.config['model_dir']}/training_report.md", 'w') as f:
            f.write(report_content)
            
        self.logger.info(f"训练报告已保存到 {self.config['model_dir']}/training_report.md") 