import abc
from typing import List, Tuple, Any
from pathlib import Path

class CDataSet(abc.ABC):
    """抽象数据集类"""
    def __init__(self, data, tag='tmp'):
        self.data = data
        self.tag = tag

    @abc.abstractmethod
    def get_count(self) -> int:
        """获取样本数量"""
        pass

    @abc.abstractmethod
    def get_pos_count(self) -> int:
        """获取正样本数量"""
        pass

    @abc.abstractmethod
    def get_label(self) -> List[float]:
        """获取标签"""
        pass

class CModelGenerator(abc.ABC):
    """抽象模型生成器"""
    
    def __init__(self, model_type: str, model_tag: str, **kwargs):
        self.model_type = model_type
        self.model_tag = model_tag
        self.model_info = None
        self.config = kwargs
        
    @abc.abstractmethod
    def train(self, train_set: CDataSet, test_set: CDataSet) -> None:
        """训练模型"""
        pass

    @abc.abstractmethod
    def create_train_test_set(self, sample_iter) -> Tuple[CDataSet, CDataSet]:
        """创建训练测试集"""
        pass

    @abc.abstractmethod
    def save_model(self) -> None:
        """保存模型"""
        pass

    @abc.abstractmethod
    def load_model(self) -> int:
        """加载模型，返回特征维度"""
        pass

    @abc.abstractmethod
    def predict(self, dataset: CDataSet) -> List[float]:
        """预测"""
        pass

    @abc.abstractmethod
    def create_data_set(self, feature_arr: List[List[float]]) -> CDataSet:
        """创建数据集"""
        pass
        
    def get_model_path(self) -> str:
        """获取模型保存路径"""
        return f"models/{self.model_type}_{self.model_tag}.model" 