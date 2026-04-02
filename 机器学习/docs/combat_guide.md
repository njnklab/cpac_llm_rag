可以，下面这版更短，适合直接贴给另一个代码模型：

````md
请帮我基于现有 Python 分类脚本，改造成一个 **ADHD-200 三站点合并 + 折内 ComBat harmonization + 5折分类** 的完整可运行版本。

## 已知数据
我有一个合并后的 csv，大致字段如下：

- `ScanDir ID`：被试ID
- `Site`：站点编号
- `Gender`
- `Age`
- `Handedness`
- `DX`

标签定义：
- `DX == 0` → 健康对照 `label=0`
- `DX != 0` → ADHD `label=1`

每个被试有一个 AAL116 的 `116×116` Pearson FC 矩阵，文件格式是 `txt`，存放在 `fc_dir` 中。  
文件名和 `ScanDir ID` 对应，请实现稳健匹配：优先精确匹配，若文件名包含该 ID 也可匹配；找不到就跳过并记录日志。

我现在代码入口风格是：

```python
participants_path = '...'
fc_dir = '...'
output_dir = '...'
````

请尽量保留这种接口形式。

---

## 任务目标

实现一个统一流程，适用于：

* SVM
* RF
* BrainNetCNN
* BrainGNN

总体流程必须是：

1. 读取合并后的人口学表
2. 读取每个被试的 FC txt 矩阵
3. 提取上三角（不含对角线），得到 6670 维特征
4. 对特征做 Fisher z 变换
5. 使用 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
6. 分层不要只按 label，而要按 `Site + "_" + label` 的联合分层键
7. **在每个 fold 内部**：

   * 只用训练集拟合 ComBat
   * batch = `Site`
   * covariates = `Age`, `Gender`
   * 再把训练好的 ComBat model 应用于该 fold 的训练集和测试集
8. 后续模型训练都基于 harmonized 特征进行
9. 输出每折结果和 5 折均值±标准差

---

## 非常重要：严禁数据泄漏

必须严格遵守：

1. **不能先对全体样本做 ComBat，再做 5折CV**
2. StandardScaler / PCA / 特征选择 等都只能在训练集 fit，再对测试集 transform
3. 测试集绝不能参与任何预处理的 fit

---

## ComBat 具体要求

优先用支持 `fit on train / apply on test` 的库，比如：

* `neuroHarmonize`
* 或其他可替代实现

ComBat 输入是在上三角 Fisher z 特征空间中进行，不是直接对 `116×116` 矩阵做。

---

## 各模型接入方式

### 1. SVM

流程：
`FC -> 上三角 -> Fisher z -> ComBat -> StandardScaler -> SVM`

建议默认：

* `kernel='rbf'`
* `class_weight='balanced'`

### 2. RF

流程：
`FC -> 上三角 -> Fisher z -> ComBat -> RF`
可选 StandardScaler，但请代码里结构清楚。

### 3. BrainNetCNN

流程：
`FC -> 上三角 -> Fisher z -> ComBat -> 重构回116×116对称矩阵 -> BrainNetCNN`

### 4. BrainGNN

流程：
`FC -> 上三角 -> Fisher z -> ComBat -> 重构回116×116对称矩阵 -> BrainGNN`
尽量不要大改原有 BrainGNN 训练逻辑，只替换输入矩阵。

---

## 需要你实现的函数建议

请尽量模块化，至少封装：

* `load_participants`
* `find_fc_file`
* `load_fc_matrix`
* `fc_to_upper_triangle`
* `fisher_z_transform`
* `rebuild_symmetric_matrix`
* `combat_fit_transform`
* `combat_apply`
* `evaluate_metrics`
* `save_fold_results`

---

## 数据清洗要求

若以下任一异常，则跳过样本并记录日志：

* 人口学字段缺失
* 找不到对应 FC 文件
* FC 读取失败
* FC shape 不是 `(116,116)`
* FC 有 NaN / inf

---

## 输出要求

每个模型都输出到自己的 `output_dir`，至少保存：

* 每折 train/test indices
* 每折 harmonized train/test 特征
* 每折 `y_test`, `y_pred`
* 每折 metrics
* confusion matrix
* 若支持，保存 `combat_model.pkl`
* 最终 `metrics_summary.csv`
* 最终 `predictions_all.csv`
* 日志文件

分类指标至少输出：

* Accuracy
* Precision
* Recall
* F1-score

另外请额外输出：

* Balanced Accuracy
* Macro-F1
* Weighted-F1

最终汇总为 5 折均值 ± 标准差。

---

## 额外要求

请尽量支持一个开关参数：

```python
use_combat = True / False
```

这样我可以直接比较：

* No ComBat
* With ComBat

---

## 你最终要给我的内容
先跟我确认思路
后给出或修改 **可运行 Python 代码**，。


代码里请写清楚注释，尤其说明：

* ComBat 如何在 fold 内做
* 如何避免数据泄漏
* 深度模型如何从 harmonized 上三角重构为矩阵


