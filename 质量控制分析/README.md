# 质量控制分析 (Quality Control Analysis)

本模块用于提取、分析和可视化fMRI预处理后的质量控制(QC)指标，对比不同预处理工具(CPAC-Default, CPAC-LLM, DeepPrep, fMRIPrep)的性能。

## 📁 目录结构

```
质量控制分析/
├── README.md                          # 本文档
│
├── 核心提取脚本（根目录）              # QC指标提取
│   ├── extract_cpac_qc.py             # 从CPAC输出提取QC指标
│   ├── extract_deepprep_qc.py         # 从DeepPrep输出提取QC指标
│   ├── extract_fmriprep_qc.py         # 从fMRIPrep输出提取QC指标
│   ├── extract_func_qc_metrics.py     # 功能像QC指标提取
│   └── extract_struct_qc_metrics.py   # 结构像QC指标提取
│
├── 画图/                              # 可视化脚本
│   ├── draw.py                        # 双框架对比（Default vs LLM）
│   ├── draw_all_frameworks.py         # 四框架对比箱线图
│   ├── draw_by_dataset.py             # 按数据集绘制对比图
│   └── qc_summary_statistics.csv      # QC统计汇总
│
├── cpac-default/                      # CPAC默认配置QC结果
│   ├── ds002748-default-qc-all.csv
│   ├── kki-default-qc-all.csv
│   ├── neuroimage-default-qc-all.csv
│   └── ohsu-default-qc-all.csv
│
├── cpac-llm/                          # CPAC-LLM配置QC结果
│   ├── ds002748-llm-qc-all.csv
│   ├── kki-llm-qc-all.csv
│   ├── neuroimage-llm-qc-all.csv
│   └── ohsu-llm-qc-all.csv
│
├── deepprep/                          # DeepPrep QC结果
│   ├── ds002748-deepprep-qc-all.csv
│   ├── KKI-deepprep-qc-all.csv
│   ├── NeuroIMAGE-deepprep-qc-all.csv
│   └── OHSU-deepprep-qc-all.csv
│
├── fmriprep/                          # fMRIPrep QC结果
│   ├── ds002748-fmriprep-qc-all.csv
│   ├── KKI-fmriprep-qc-all.csv
│   ├── NeuroIMAGE-fmriprep-qc-all.csv
│   └── OHSU-fmriprep-qc-all.csv
│
├── 六个指标均值标准差/                 # QC指标统计分析
│   ├── qc_metrics_summary.csv         # 指标汇总表
│   └── qc_summary_report.md           # 分析报告
│
└── qc_metrics_summary.csv             # 总体QC指标汇总
```

## 📊 QC指标体系

### 功能像指标 (Functional Metrics)

| 指标名 | 全称 | 含义 | 理想值 |
|--------|------|------|--------|
| **MeanFD_Power** | Mean Framewise Displacement (Power) | 平均帧间位移，反映头动程度 | 越小越好，<0.5mm |
| **MeanDVARS** | Mean DVARS | 信号强度变化率，反映时间序列稳定性 | 越小越好 |
| **boldSnr** | BOLD Signal-to-Noise Ratio | BOLD信噪比，反映数据质量 | 越大越好 |

### 结构像指标 (Structural Metrics)

| 指标名 | 全称 | 含义 | 理想值 |
|--------|------|------|--------|
| **CJV** | Coefficient of Joint Variation | 联合变异系数，反映GM/WM对比度 | 越小越好 |
| **EFC** | Entropy Focus Criterion | 熵聚焦准则，反映图像聚焦程度 | 越小越好 |
| **WM2MAX** | White Matter to Maximum | 白质强度与最大强度比值 | 接近1.0 |
| **GM_mean** | Gray Matter Mean | 灰质平均信号强度 | - |
| **WM_mean** | White Matter Mean | 白质平均信号强度 | - |

## 🚀 使用方法

### 1. 提取QC指标

**CPAC结果提取：**
```bash
# 编辑 extract_cpac_qc.py 中的路径参数
# BASE_OUTPUT_DIR: CPAC输出目录
# OUTPUT_CSV_PATH: 输出CSV路径
# PIPELINE_NAME: pipeline名称

python extract_cpac_qc.py
```

**DeepPrep结果提取：**
```bash
python extract_deepprep_qc.py
```

**fMRIPrep结果提取：**
```bash
python extract_fmriprep_qc.py
```

### 2. 生成对比图表

**四框架对比（论文主要图表）：**
```bash
cd 画图
python draw_all_frameworks.py
```
输出：四个预处理框架的QC指标对比箱线图

**按数据集对比：**
```bash
python draw_by_dataset.py
```
输出：不同数据集（ds002748, KKI, NeuroIMAGE, OHSU）的QC对比

**双框架对比（Default vs LLM）：**
```bash
python draw.py
```
输出：CPAC-Default与CPAC-LLM的详细对比

## 📈 数据集说明

| 数据集 | 描述 | 被试数 | 用途 |
|--------|------|--------|------|
| **ds002748** | 抑郁症rs-fMRI | ~50 | 主要分析数据集 |
| **KKI** | ADHD-200子集 | ~80 | 多站点验证 |
| **NeuroIMAGE** | ADHD-200子集 | ~60 | 多站点验证 |
| **OHSU** | ADHD-200子集 | ~70 | 多站点验证 |

## 🔧 脚本配置说明

所有提取脚本都采用**手动配置参数**方式，需在脚本头部修改：

```python
# ===== 手动在这里设置参数 =====
BASE_OUTPUT_DIR = "/path/to/output"      # 预处理结果根目录
OUTPUT_CSV_PATH = "/path/to/output.csv"  # 输出CSV路径
PIPELINE_NAME = "pipeline_name"          # Pipeline名称
SESSIONS = ["ses-1", "ses-2"]           # Session列表
TASK = None                              # Task名称（可选）
```

## 📊 结果文件格式

QC结果CSV包含以下列：
```csv
SubjectID,Session,MeanFD_Power,MeanDVARS,boldSnr,CJV,EFC,WM2MAX,GM_mean,WM_mean
sub-01,ses-1,0.123,0.456,150.2,0.345,0.567,0.890,850.2,950.5
...
```

## 📝 注意事项

1. **路径配置**：所有提取脚本都使用绝对路径，使用前请根据实际情况修改
2. **依赖包**：需要安装 `nibabel`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `mriqc`
3. **数据完整性**：确保预处理结果完整，脚本会自动跳过缺失数据的被试
4. **Session选择**：对于多session数据，脚本会按SESSIONS列表顺序选择第一个存在的session

## 📧 指标解读参考

- **MeanFD_Power** < 0.5mm：头动控制良好
- **MeanFD_Power** 0.5-1.0mm：中等头动，建议scrubbing
- **MeanFD_Power** > 1.0mm：头动较大，需谨慎分析
- **CJV** 越小：GM/WM对比度越好，分割质量越高
- **EFC** 越小：图像聚焦越好，伪影越少

## 🤝 与主流程集成

QC分析是预处理流程的最后一步：
```
预处理(CPAC/DeepPrep/fMRIPrep) → QC指标提取 → 对比分析 → 可视化
```

QC结果可用于：
- 评估预处理质量
- 对比不同工具性能
- 筛选合格被试
- 指导预处理参数优化
