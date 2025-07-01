#!/usr/bin/env python3
"""
Phase 1测试脚本
测试基础机器学习框架
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from ModelStrategy.ChanMLTrainer import ChanMLTrainer

def test_basic_training():
    """测试基础训练流程"""
    print("="*60)
    print("Phase 1 基础训练测试")
    print("="*60)
    
    # 创建训练器
    config = {
        'data_dir': 'data/ml_datasets',
        'model_dir': 'models/phase1',
        'log_dir': 'logs/phase1',
        'dataset_name': 'main_training',
        'target_column': 'binary_direction',
        'random_state': 42,
        'models': ['xgb']
    }
    
    trainer = ChanMLTrainer(config)
    
    # 运行训练
    results = trainer.run_full_training_pipeline()
    
    print("\n训练完成! 结果摘要:")
    for model_type, result in results.items():
        val_metrics = result['val_metrics']
        print(f"{model_type}: AUC={val_metrics['auc']:.4f}, "
              f"Acc={val_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    test_basic_training() 