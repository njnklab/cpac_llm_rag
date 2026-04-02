# MRI Preprocessing Tools

MRI预处理工具的统一调用接口，支持CPAC、DeepPrep和fMRIPrep三种主流预处理管道。

## 目录结构

```
preprocess/
├── README.md                          # 本文档
├── configs/                           # 配置文件目录
│   ├── config.yaml                   # 主配置文件（统一参数）
│   └── pipeline_config_default.yml   # CPAC管道配置
├── tools/                            # 工具脚本目录
│   ├── cpac_runner.py               # CPAC预处理
│   ├── deepprep_runner.py           # DeepPrep预处理（GPU）
│   └── fmriprep_runner.py           # fMRIPrep预处理
└── docs/                            # 文档目录
    └── USAGE.md                     # 详细使用说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install pyyaml
```

### 2. 配置参数

编辑 `configs/config.yaml`，修改以下关键路径：

```yaml
dataset:
  id: "你的数据集ID"
  bids_root: "/path/to/your/bids/data"
```

### 3. 运行预处理

**CPAC：**
```bash
cd tools
python cpac_runner.py
```

**DeepPrep（需要NVIDIA GPU）：**
```bash
cd tools
python deepprep_runner.py
```

**fMRIPrep：**
```bash
cd tools
python fmriprep_runner.py
```

## 配置说明

所有参数集中在 `configs/config.yaml` 中管理，分为以下几个部分：

### 通用配置

- `dataset`: 数据集ID和BIDS根目录
- `batch`: 批处理大小
- `paths`: 处理列表文件和日志文件路径

### CPAC配置 (`cpac`)

- `output_dir`: CPAC输出目录
- `working_dir`: CPAC工作目录

### DeepPrep配置 (`deepprep`)

- `bids_dir`: BIDS数据目录
- `output_dir`: 输出目录
- `work_root`: 工作目录根路径
- `fs_license`: FreeSurfer license文件路径
- `docker_image`: DeepPrep Docker镜像
- `gpu_ids`: GPU ID列表（如 `[0]` 或 `[0,1]`）
- `cpus_per_subject`: 每被试CPU数
- `memory_per_subject_gb`: 每被试内存（GB）

### fMRIPrep配置 (`fmriprep`)

- `bids_dir`: BIDS数据目录
- `output_dir`: 输出目录
- `work_root`: 工作目录根路径
- `bids_db_dir`: PyBIDS数据库目录
- `fs_license`: FreeSurfer license文件路径
- `image`: Docker镜像（`AUTO`自动选择或指定如 `nipreps/fmriprep:25.2.3`）
- `max_jobs`: 并发被试数
- `nprocs`: 每被试CPU线程数
- `mem_mb`: 每被试内存（MB）
- `omp_nthreads`: OpenMP线程数
- `skip_bids_validation`: 是否跳过BIDS验证

## 外部依赖

以下模块需要在PYTHONPATH中：

- **Tools**: `/home/user/zhangyan/cpac/workflow/intelliparam/`
- **qc_utils**: 位于脚本所在目录或PYTHONPATH中

## 预处理列表文件格式

### 待处理列表 (to_process_file)

每行一个被试ID，格式如下：
```
sub-01
sub-02
sub-03
```

或使用纯数字ID（脚本会自动添加 `sub-` 前缀）：
```
01
02
03
```

### 已处理列表 (processed_file)

由脚本自动维护，记录已完成的被试ID。

## 输出说明

### 日志文件

- **CPAC**: `/home/a001/zhangyan/cpac/log/cpac_pipeline.log`
- **DeepPrep**: `configs/config.yaml` 中 `deepprep.log_dir` 指定的目录
- **fMRIPrep**: `<output_dir>/logs/fmriprep_sub-<subject_id>.log`

### 状态报告

- **DeepPrep**: `KKI_processing_status.csv` 和 `KKI_qc_summary.csv`
- **fMRIPrep**: `<output_dir>/logs/summary.json` 和 `summary.csv`

## 注意事项

1. **Docker权限**: 确保当前用户有权限运行Docker命令
2. **GPU驱动**: DeepPrep需要NVIDIA GPU和nvidia-docker支持
3. **磁盘空间**: 预处理产生大量中间文件，确保有足够磁盘空间
4. **BIDS格式**: 输入数据需符合BIDS格式规范
5. **FreeSurfer License**: 需要有效的FreeSurfer license文件

## 常见问题

### Q: 如何切换数据集？

修改 `configs/config.yaml` 中的 `dataset.bids_root` 和相关路径配置即可。

### Q: 如何只处理部分被试？

编辑待处理列表文件（`to_process_file`），只保留需要处理的被试ID。

### Q: 如何重新处理已完成的被试？

从已处理列表文件（`processed_file`）中删除对应被试ID，或直接清空该文件。

### Q: fMRIPrep提示BIDS验证失败？

在配置中设置 `skip_bids_validation: true` 可跳过BIDS验证。

## 代码修改说明

**与原代码相比，改动极小**：

1. **只添加了配置加载函数**：每个脚本开头添加了 `load_config()` 函数和 `CONFIG_PATH` 定义
2. **只修改了参数来源**：将原来的硬编码参数改为从配置文件中读取
3. **完全保留原始代码风格**：包括所有注释、变量名、代码结构等

例如，原来 DeepPrep 中的：
```python
BIDS_DIR = "/media/a001/6F5E-9C72/zhangyan/ADHD/ADHD200_subset/bids/KKI"
```

现在改为：
```python
cfg = load_config()
BIDS_DIR = cfg['deepprep']['bids_dir']
```

仅此而已，其他代码完全不变。

## 版本历史

- 2026-04-02: 重构代码结构，统一配置文件
