# 项目结构变更说明

## 整理前的问题

1. **文件分散混乱**：所有文件都在根目录，没有清晰的组织结构
2. **代码重复**：fc_classification.py包含所有传统模型逻辑，难以维护
3. **命名不规范**：包含空格的中文文件名，不符合编程规范
4. **结果分散**：combat结果和非combat结果混合存放
5. **缺乏文档**：没有统一的项目说明和使用指南

## 整理后的结构

```
机器学习/
├── src/                          # 源代码（核心代码）
│   ├── models/                   # 所有模型实现
│   │   ├── traditional_ml.py    # SVM和Random Forest
│   │   ├── braingnn.py          # BrainGNN模型
│   │   └── brainnetcnn.py       # BrainNetCNN模型
│   ├── data/                     # 数据加载
│   │   ├── dataset_braingnn.py
│   │   └── dataset_brainnetcnn.py
│   ├── experiments/              # 实验运行脚本
│   │   ├── run_traditional_ml.py
│   │   ├── train_braingnn.py
│   │   └── train_brainnetcnn.py
│   └── utils/                    # 工具函数
│       ├── data_utils.py        # 数据加载工具
│       └── metrics.py           # 评估指标
├── data/                        # 数据（人口学信息）
│   └── demographics/
├── results/                      # 实验结果（结构化存放）
│   ├── combat/                  # ComBat后的结果
│   │   ├── svm_rf/
│   │   ├── braingnn/
│   │   └── brainnetcnn/
│   └── nocombat/                # 非ComBat结果（仅存档）
│       └── svm_rf/
├── docs/                        # 所有文档
│   ├── model_architecture.md   # 模型架构说明
│   ├── combat_guide.md         # ComBat方法说明
│   ├── hybrid_strategy_comparison.md
│   ├── total_metrics_comparison.md
│   └── *.csv                   # 对比数据文件
├── scripts/                     # 辅助脚本
│   └── analyze_results.py      # 结果分析脚本
├── legacy/                      # 未使用的旧文件/备份
│   ├── fc_classification_original.py
│   ├── svm_rf_result_未使用/
│   └── 人口学_old/
├── BrainGNN/                    # 原始BrainGNN代码（保留）
├── BrainNetCNN/                 # 原始BrainNetCNN代码（保留）
└── README.md                    # 项目主文档
```

## 关键改进

### 1. 代码模块化
- 将`fc_classification.py`拆分为多个模块：
  - `traditional_ml.py`: 包含SVM和RF类
  - `data_utils.py`: 数据加载和处理函数
  - `metrics.py`: 评估指标计算
  - `run_traditional_ml.py`: 主运行脚本

### 2. 命名规范化
- 移除包含空格的文件名
- 统一使用英文命名
- 使用下划线分隔单词

### 3. 结果结构化
- `results/combat/`: 存放ComBat校正后的结果
- `results/nocombat/`: 存放非ComBat结果（标注为"仅存档"）
- `legacy/`: 存放未使用的旧结果

### 4. 文档集中化
- 所有文档集中在`docs/`目录
- 创建`README.md`说明项目结构
- 保留原有的对比结果文档

### 5. Python模块可导入
- 所有目录添加`__init__.py`
- 可以通过`from src.models.traditional_ml import SVMClassifier`导入

## 如何使用

### 运行传统机器学习实验
```bash
cd src/experiments
python run_traditional_ml.py
```

### 运行深度学习实验
```bash
# BrainGNN
cd src/experiments
python train_braingnn.py

# 或从原始目录运行
cd BrainGNN
python run_combat_braingnn.py
```

### 分析结果
```bash
cd scripts
python analyze_results.py
```

## 文件移动对照表

| 原位置 | 新位置 | 说明 |
|--------|--------|------|
| fc_classification.py | src/experiments/run_traditional_ml.py | 重构后的版本 |
| fc_classification.py | legacy/fc_classification_original.py | 备份原文件 |
| 人口学/ | data/demographics/ | 人口学数据 |
| svm_rf_result/ | legacy/svm_rf_result_未使用/ | 标注为未使用 |
| combat结果/svm_rf/ | results/combat/svm_rf/ | 结果文件 |
| BrainGNN/ | src/models/braingnn.py | 代码复制 |
| BrainNetCNN/ | src/models/brainnetcnn.py | 代码复制 |
| hybrid_strategy_comparison.md | docs/ | 移动 |
| total_metrics_comparison.md | docs/ | 移动 |
| BrainNetCNN 与 BrainGNN 模型实现说明md | docs/model_architecture.md | 重命名 |
| combat结果/comBat.md | docs/combat_guide.md | 重命名 |

## 注意事项

1. **路径更新**：新代码中的路径需要根据实际情况调整
2. **保留原始代码**：BrainGNN/和BrainNetCNN/目录保留，包含完整实验脚本
3. **模块化版本**：src/目录下是新结构的模块化代码
4. **依赖管理**：确保安装所有依赖包（见README.md）

## 后续建议

1. **配置管理**：建议使用`config.yaml`统一管理实验参数
2. **日志系统**：添加统一的日志记录功能
3. **单元测试**：为关键函数添加单元测试
4. **CI/CD**：设置自动化测试和代码检查