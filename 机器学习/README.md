# ADHD分类机器学习项目

基于FC（功能连接）矩阵的ADHD二分类实验框架，支持单中心和多中心数据，对比四个框架处理后的数据在两个深度学习模型（BrainGNN、BrainNetCNN）和两个传统机器学习模型（SVM、Random Forest）上的表现。

## 项目结构

```
机器学习/
├── src/                          # 源代码
│   ├── models/                   # 模型实现
│   │   ├── traditional_ml.py    # SVM和Random Forest
│   │   ├── braingnn.py          # BrainGNN模型
│   │   └── brainnetcnn.py       # BrainNetCNN模型
│   ├── data/                     # 数据加载
│   ├── experiments/              # 实验脚本
│   │   └── run_traditional_ml.py
│   └── utils/                    # 工具函数
│       ├── data_utils.py        # 数据加载工具
│       └── metrics.py           # 评估指标
├── results/                      # 实验结果
│   ├── combat/                  # ComBat后的结果
│   │   ├── svm_rf/
│   │   ├── braingnn/
│   │   └── brainnetcnn/
│   └── nocombat/                # 非ComBat结果（仅存档）
│       └── svm_rf/
├── data/                        # 数据
│   └── demographics/            # 人口学信息
├── docs/                        # 文档
│   ├── model_architecture.md   # 模型架构说明
│   └── combat_guide.md         # ComBat方法说明
├── legacy/                      # 未使用的旧结果
│   └── svm_rf_result_未使用/
├── scripts/                     # 快速运行脚本
├── BrainGNN/                    # BrainGNN模型代码
├── BrainNetCNN/                 # BrainNetCNN模型代码
└── README.md                    # 本文件
```

## 四个预处理框架

- **cpac-default**: CPAC默认配置
- **cpac-llm**: CPAC LLM增强配置
- **fmriprep**: fMRIPrep预处理
- **deepprep**: DeepPrep预处理

## 四个机器学习模型

### 深度学习模型
1. **BrainGNN**: 基于图神经网络的脑网络分类模型
2. **BrainNetCNN**: 专门用于脑网络矩阵的卷积神经网络

### 传统机器学习模型
3. **SVM (支持向量机)**: 线性核，C=0.01
4. **Random Forest (随机森林)**: 150棵树，最大深度10

## 快速开始

### 1. 运行传统机器学习实验

```bash
cd src/experiments
python run_traditional_ml.py
```

### 2. 运行深度学习实验

```bash
cd BrainGNN
python run_combat_braingnn.py

cd ../BrainNetCNN
python run_combat_brainnetcnn.py
```

## 数据流程

1. **数据加载**: 从CSV加载人口学信息，从txt加载FC矩阵
2. **预处理**: 
   - Fisher Z变换
   - ComBat多中心校正（可选）
3. **特征选择**: SelectKBest (k=200)
4. **标准化**: StandardScaler
5. **分类**: 五折分层交叉验证

## 评价指标

- Accuracy
- Precision
- Recall
- F1-score
- AUC
- Specificity
- Balanced Accuracy
- Confusion Matrix

## 实验设置

- **交叉验证**: 5折分层交叉验证
- **随机种子**: 42
- **类别不平衡处理**: class_weight='balanced'
- **特征选择**: k=200

## 结果查看

结果保存在 `results/` 目录下：
- `results/combat/`: 使用ComBat校正后的结果
- `results/nocombat/`: 不使用ComBat的结果（仅存档）

每个实验生成 `classification_results_{combat|nocombat}.txt` 文件，包含五折均值和标准差。

## 依赖

- Python 3.8+
- numpy
- pandas
- scikit-learn
- pytorch (用于深度学习模型)
- torch_geometric (用于BrainGNN)
- neuroHarmonize (用于ComBat，可选)

## 注意事项

1. BrainGNN和BrainNetCNN使用PyTorch实现，需要GPU支持以获得最佳性能
2. ComBat功能需要安装neuroHarmonize库
3. 所有路径需要根据实际情况修改

## 相关文档

- [模型架构说明](docs/model_architecture.md)
- [ComBat方法说明](docs/combat_guide.md)