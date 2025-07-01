import xgboost as xgb
import numpy as np
from typing import List, Tuple, Dict, Any
import os
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ModelStrategy.ModelGenerator import CModelGenerator, CDataSet

class CXGBDataSet(CDataSet):
    """XGBoost数据集"""
    
    def __init__(self, data: xgb.DMatrix, tag='tmp'):
        super().__init__(data, tag)
        
    def get_count(self) -> int:
        return self.data.num_row()
        
    def get_pos_count(self) -> int:
        labels = self.get_label()
        return int(sum(labels))
        
    def get_label(self) -> List[float]:
        return self.data.get_label().tolist()

class CXGBModelGenerator(CModelGenerator):
    """XGBoost模型生成器"""
    
    def __init__(self, model_tag: str, xgb_params: Dict = None):
        super().__init__('xgb', model_tag)
        self.xgb_params = xgb_params or self._get_default_params()
        self.model = None
        self.feature_names = None
        
    def _get_default_params(self) -> Dict:
        """获取默认XGBoost参数"""
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
            'random_state': 42
        }
        
    def create_data_set(self, X: np.ndarray, y: np.ndarray = None) -> CXGBDataSet:
        """创建XGBoost数据集"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        dmatrix = xgb.DMatrix(X, label=y)
        return CXGBDataSet(dmatrix)
        
    def create_train_test_set(self, sample_iter) -> Tuple[CXGBDataSet, CXGBDataSet]:
        """创建训练测试集"""
        # 这个方法在我们的实现中不会用到，因为我们已经有预分割的数据
        raise NotImplementedError("使用外部数据分割")
        
    def train(self, train_set: CXGBDataSet, test_set: CXGBDataSet) -> None:
        """训练XGBoost模型"""
        evals = [(train_set.data, 'train'), (test_set.data, 'eval')]
        
        self.model = xgb.train(
            params=self.xgb_params,
            dtrain=train_set.data,
            num_boost_round=100,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
    def predict(self, dataset: CXGBDataSet) -> List[float]:
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")
            
        predictions = self.model.predict(dataset.data)
        return predictions.tolist()
        
    def save_model(self) -> None:
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")
            
        model_path = self.get_model_path()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型
        self.model.save_model(model_path)
        
        # 保存元数据
        meta_path = model_path.replace('.model', '_meta.json')
        meta_info = {
            'model_type': self.model_type,
            'model_tag': self.model_tag,
            'xgb_params': self.xgb_params,
            'num_features': self.model.num_features(),
            'feature_names': self.feature_names
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta_info, f, indent=2)
            
    def load_model(self) -> int:
        """加载模型"""
        model_path = self.get_model_path()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        # 加载模型
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # 加载元数据
        meta_path = model_path.replace('.model', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta_info = json.load(f)
                self.xgb_params = meta_info.get('xgb_params', {})
                self.feature_names = meta_info.get('feature_names', [])
                
        return self.model.num_features() 