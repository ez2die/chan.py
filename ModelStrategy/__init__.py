"""
Chan.py机器学习框架
基于缠论的量化交易模型训练和预测系统
"""

__version__ = "1.0.0"
__author__ = "Chan.py ML Team"

from .ModelGenerator import CModelGenerator, CDataSet
from .DatasetManager import ChanDatasetManager
from .FeatureCalculator import ChanFeatureCalculator
from .ChanMLTrainer import ChanMLTrainer

__all__ = [
    'CModelGenerator',
    'CDataSet', 
    'ChanDatasetManager',
    'ChanFeatureCalculator',
    'ChanMLTrainer',
]
