````markdown
# BrainNetCNN 与 BrainGNN 模型实现说明（面向代码生成）

## 1. 文档目标

本说明文档不负责描述整个 FC 分类实验框架，只负责告诉代码生成 AI：

1. `BrainNetCNN` 应该如何理解、如何实现、输入输出是什么；
2. `BrainGNN` 应该如何理解、如何实现、输入输出是什么；
3. 二分类任务中类别不平衡如何处理；
4. 如何在 **五折交叉验证** 下训练与评估这两个深度学习模型；
5. 如何输出符合论文需要的结果指标。

当前任务默认场景：

- 输入数据：每个样本一个 `116×116` 的 FC 矩阵 `.txt`
- 任务：二分类（如 ADHD vs HC）
- 标签：`0/1`
- 框架：优先使用 `PyTorch`；`BrainGNN` 推荐 `PyTorch Geometric`

---

# 2. BrainNetCNN 实现说明

## 2.1 BrainNetCNN 的核心思想

`BrainNetCNN` 不是普通的 2D CNN。  
它不是把 FC 矩阵当成普通图像来处理，而是把它当成 **脑网络连接矩阵** 来处理。

对于一个 `N×N` 的 FC 矩阵：

- 行 / 列分别对应 ROI
- 元素 `A[i, j]` 表示 ROI i 和 ROI j 的连接强度

普通 CNN 的局部卷积假设“空间相邻像素具有局部连续意义”，  
但 FC 矩阵并不满足这个假设，因此不建议直接使用普通 `Conv2d(3x3)` 堆模型。  
BrainNetCNN 的重点是使用更适合连接矩阵的操作，尤其是：

1. `Edge-to-Edge (E2E)`
2. `Edge-to-Node (E2N)`
3. `Node-to-Graph (N2G)`

因此，代码实现时应尽量保留这种结构思想。

---

## 2.2 BrainNetCNN 的输入格式

每个样本输入保持完整 FC 矩阵，不展平。

推荐输入张量形状：

```python
[B, 1, N, N]
````

对于当前任务：

```python
[B, 1, 116, 116]
```

其中：

* `B` = batch size
* `1` = 单通道
* `N = 116`

### 输入预处理建议

1. 使用 `numpy.loadtxt()` 读取矩阵；
2. 转为 `float32`；
3. 建议将对角线置为 0；
4. 保留矩阵对称性；
5. 默认保留负值，不要一开始就删除负相关；
6. 若后续训练不稳定，可尝试：

   * clipping
   * Fisher z
   * 全局标准化
     但这些都应作为可选项，不要先强行固定。

---

## 2.3 BrainNetCNN 的最小可实现结构

如果不严格逐项复现原论文，也必须实现一个**结构上接近 BrainNetCNN 思想**的版本。

推荐最小结构：

```text
Input FC Matrix
   ↓
E2E Block
   ↓
E2E Block
   ↓
E2N Block
   ↓
N2G Block
   ↓
Fully Connected
   ↓
Classifier (2 logits)
```

---

## 2.4 如何理解 E2E / E2N / N2G

### 2.4.1 E2E（Edge-to-Edge）

作用：

* 学习“连接与连接之间”的模式；
* 强调某条边在其所在行和列中的上下文关系；
* 目的不是像普通图像卷积一样提局部纹理，而是提取网络连接模式。

实现时，代码 AI 可以采用“BrainNetCNN 风格的十字形卷积”或近似实现。
如果严格写自定义层较麻烦，可以写成一种“分别沿行和列聚合，再融合”的模块。

### 2.4.2 E2N（Edge-to-Node）

作用：

* 将边层面的表示压缩到节点层面；
* 输出可理解为：每个 ROI 的综合连接表示。

直观理解：

* 输入仍是连接矩阵级特征；
* 输出逐步变成“每个节点的特征”。

### 2.4.3 N2G（Node-to-Graph）

作用：

* 将节点层面的信息整合成图级表示；
* 为最终二分类提供一个 graph-level representation。

输出一般接：

* flatten / squeeze
* MLP
* binary classifier

---

## 2.5 BrainNetCNN 的代码实现建议

## 2.5.1 推荐文件结构

```text
model_brainnetcnn.py
dataset_brainnetcnn.py
train_brainnetcnn.py
```

---

## 2.5.2 dataset_brainnetcnn.py 应实现的内容

定义一个 `torch.utils.data.Dataset`，至少返回：

```python
fc_tensor, label, subject_id
```

其中：

* `fc_tensor.shape == [1, 116, 116]`
* `label` 为 `0/1`
* `subject_id` 用于后续保存预测结果

建议逻辑：

```python
1. 读取 txt
2. 转成 np.float32
3. 对角线置 0
4. 转成 torch.float32
5. unsqueeze(0) -> [1, 116, 116]
```

---

## 2.5.3 model_brainnetcnn.py 应实现的内容

至少实现两个类：

### （1）E2EBlock

负责边到边特征提取。

### （2）BrainNetCNN

完整主模型，结构建议：

```python
class BrainNetCNN(nn.Module):
    def __init__(self, n_nodes=116, n_classes=2):
        ...
    def forward(self, x):
        # x: [B, 1, N, N]
        ...
        return logits
```

建议结构示例（仅为参考，不要求完全一致）：

```text
Input [B,1,116,116]
→ E2EBlock(1 -> 32)
→ BN / ReLU / Dropout
→ E2EBlock(32 -> 64)
→ BN / ReLU / Dropout
→ E2NBlock(64 -> 128)
→ BN / ReLU / Dropout
→ N2GBlock(128 -> 256)
→ FC(256 -> 64)
→ Dropout
→ FC(64 -> 2)
```

---

## 2.5.4 BrainNetCNN 的训练建议

损失函数：

```python
nn.CrossEntropyLoss(...)
```

优化器：

```python
torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

可参考超参数：

* `batch_size = 8 / 16`
* `epochs = 80 ~ 150`
* `dropout = 0.3 ~ 0.5`
* `lr = 1e-3`
* `weight_decay = 1e-4`

若结果不好，可尝试：

1. 降低学习率到 `5e-4` 或 `1e-4`
2. 增大 dropout
3. 增加 early stopping
4. 对输入做 z-score 或 Fisher z
5. 先减小模型宽度，防止小样本过拟合

---

## 2.6 BrainNetCNN 的实现底线

必须满足：

1. 输入是完整 `116×116` FC 矩阵；
2. 不能简单写成“普通 2D CNN 分类图像”；
3. 模块命名应体现 `E2E / E2N / N2G` 思想；
4. 最终输出是图级二分类 logits。

---

# 3. BrainGNN 实现说明

## 3.1 BrainGNN 的核心思想

`BrainGNN` 是图神经网络思路。
它把 FC 矩阵转成一张图：

* ROI = 节点
* FC 连接 = 边
* 节点自身还可以有节点特征

相比 BrainNetCNN，BrainGNN 更强调：

1. 直接利用图结构；
2. 在节点之间做消息传递；
3. 通过 pooling/selection 聚焦重要 ROI；
4. 输出可解释的 graph-level 表征。

---

## 3.2 BrainGNN 的输入图构建

对于每个 `116×116` FC 矩阵，要构建成图数据对象。

推荐使用 `torch_geometric.data.Data`，每个样本至少包括：

```python
x           # 节点特征，shape [116, F]
edge_index  # 边索引，shape [2, E]
edge_weight # 边权重，shape [E]
y           # 标签
subject_id  # 样本ID（可额外保存）
```

---

## 3.3 节点特征 x 怎么定义

这是最重要的实现点之一。

### 推荐方案（默认首选）

对每个节点，使用其与所有节点的连接向量作为节点特征。

也就是：

* FC 矩阵第 i 行 = 第 i 个节点的特征

因此：

```python
x.shape = [116, 116]
```

优点：

1. 实现最简单；
2. 信息量足够；
3. 很适合当前只有 FC 矩阵而没有额外 ROI 属性的场景。

---

## 3.4 edge_index 和 edge_weight 怎么构建

### 推荐默认方案：全连接图（去自环）

对于每个 `i != j`：

* 建立一条边 `(i, j)`
* `edge_weight = FC[i, j]`

即：

* 不做阈值化
* 不先删弱边
* 不先只保留正边

原因：

* 先把实验跑通更重要；
* 阈值化本身会引入额外超参数；
* 若后续显存/噪声问题明显，再考虑稀疏化。

### 可选优化方案

如果后续效果不好或图太密：

1. 仅保留 `|FC|` 最大的 top-k% 边；
2. 只保留每个节点 top-k 邻居；
3. 对负边单独处理；
4. 尝试 `edge_weight = abs(FC)`；
5. 或拆成正边图 / 负边图（进阶版，不作为首选）。

---

## 3.5 BrainGNN 的最小可实现结构

如果完整复现原版 BrainGNN 太复杂，可以先写一个 **BrainGNN 风格简化版**。
但必须包含三个关键思想：

1. 图卷积层（graph convolution）
2. 节点筛选 / pooling
3. 图级分类输出

推荐最小结构：

```text
Input Graph
   ↓
Graph Conv Layer
   ↓
Graph Conv Layer
   ↓
TopK Pooling / ROI-aware Pooling
   ↓
Graph Conv Layer
   ↓
Global Pooling
   ↓
MLP
   ↓
Classifier
```

---

## 3.6 BrainGNN 代码实现建议

## 3.6.1 推荐文件结构

```text
model_braingnn.py
dataset_braingnn.py
train_braingnn.py
```

---

## 3.6.2 dataset_braingnn.py 应实现的内容

定义图数据集，读取 txt 后转成 `torch_geometric.data.Data`。

伪流程：

```python
1. 读取 116×116 FC
2. 对角线置 0
3. x = FC 本身（每行作为一个节点特征）
4. 遍历 i,j 构建 edge_index
5. edge_weight = FC[i,j]
6. y = label
7. 返回 Data(x, edge_index, edge_weight, y)
```

若使用无向图，可只构造上三角后再双向补边。
如果直接构造全矩阵双向边，也可以。

---

## 3.6.3 model_braingnn.py 应实现的内容

推荐最少实现以下类：

### （1）GraphConvBlock

封装一层图卷积 + BN + ReLU + Dropout

可选卷积层：

* `GCNConv`
* `GraphConv`
* `GATConv`

**推荐第一版优先用 `GCNConv` 或 `GraphConv`**，更稳。

### （2）BrainGNN

完整模型，建议接口：

```python
class BrainGNN(nn.Module):
    def __init__(self, in_channels=116, hidden_channels=64, num_classes=2):
        ...
    def forward(self, data):
        ...
        return logits
```

推荐第一版结构：

```text
x, edge_index, edge_weight
→ GraphConv(116 -> 64)
→ ReLU + Dropout
→ GraphConv(64 -> 64)
→ ReLU + Dropout
→ TopKPooling(64, ratio=0.5)
→ GraphConv(64 -> 128)
→ ReLU
→ global_mean_pool + global_max_pool
→ concat
→ FC(256 -> 64)
→ Dropout
→ FC(64 -> 2)
```

注意：

* `global_mean_pool` 和 `global_max_pool` 可拼接使用；
* 这样图级表征更稳定；
* pooling 之后一定要有 graph-level readout。

---

## 3.6.4 关于“原版 BrainGNN”与“简化版 BrainGNN”

如果代码 AI 很强，可以写更贴近 BrainGNN 论文思想的版本：

* ROI-aware graph convolution
* ROI-selection pooling
* 可能包含解释性约束项

但如果时间有限，**优先保证一个可跑通、结构合理的简化版**：

* 图卷积
* TopKPooling
* 图级读出
* 二分类输出

这已经足以作为 BrainGNN-style baseline。

---

## 3.6.5 BrainGNN 训练建议

损失函数：

```python
nn.CrossEntropyLoss(...)
```

优化器：

```python
torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

可参考超参数：

* `batch_size = 8 / 16`（图批）
* `epochs = 80 ~ 150`
* `hidden_channels = 64`
* `dropout = 0.3 ~ 0.5`
* `pooling ratio = 0.5`
* `lr = 1e-3`

若结果不好，可尝试：

1. `hidden_channels: 32 / 64 / 128`
2. `pooling ratio: 0.5 / 0.7 / 0.8`
3. `lr: 1e-3 -> 5e-4 -> 1e-4`
4. 图稀疏化（top-k edges）
5. `GraphConv` 替代 `GCNConv`
6. 改成 `GATConv`
7. 在 readout 里拼接 mean/max pooling
8. 输入边权做归一化

---

# 4. 类别不平衡处理（两种模型都要加）

如果两类样本数不一致，必须显式处理类别不平衡。
否则模型可能偏向多数类，表面 accuracy 不低，但 recall / specificity 会很难看。

---

## 4.1 推荐优先方案：loss 加 class weights

对于训练集每个 fold，统计类别数量：

```python
n0 = number of class 0 in training set
n1 = number of class 1 in training set
```

计算类别权重，例如：

```python
w0 = (n0 + n1) / (2 * n0)
w1 = (n0 + n1) / (2 * n1)
```

然后：

```python
criterion = nn.CrossEntropyLoss(weight=torch.tensor([w0, w1], dtype=torch.float32).to(device))
```

注意：

* 权重只能根据**当前训练折**计算；
* 不能根据全数据集计算，否则会引入轻微泄漏。

---

## 4.2 可选方案：WeightedRandomSampler

如果类别非常不平衡，也可以对训练集使用：

```python
torch.utils.data.WeightedRandomSampler
```

但第一版建议：

* **先用 class weights**
* sampler 作为可选增强手段

因为 sampler 有时会让训练波动更大。

---

## 4.3 不建议的做法

第一版不建议：

1. 先做 SMOTE 再进深度学习；
2. 在全数据上计算类别权重；
3. 为了追求 accuracy 完全忽略 recall / specificity。

---

# 5. 五折交叉验证要求

这两个深度学习模型都必须使用 **5-fold cross validation**。

---

## 5.1 基本原则

使用：

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=固定值)
```

要求：

1. 每一折类别分布尽量保持一致；
2. 每一折都重新训练模型；
3. 每一折都单独保存最优模型结果；
4. 最终报告 mean 和 std。

---

## 5.2 每个 fold 的推荐流程

对于第 k 折：

1. 划分 train / test
2. 再从 train 中划一个 validation（推荐 10%~20%）
3. 用 train 真正训练
4. 用 validation 做 early stopping / best model selection
5. 用 test 计算最终指标

注意：

* test fold 不能参与调参
* validation 只能来自当前训练折

---

## 5.3 early stopping

建议监控：

* `val_auc`
* 或 `val_f1`
* 或 `val_balanced_accuracy`

推荐：

* `patience = 15 ~ 20`

保存每个 fold 的 best checkpoint。

---

# 6. 评价指标要求

每个 fold 都要输出以下指标：

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. AUC
6. Confusion Matrix

此外建议额外计算：

7. Specificity
8. Balanced Accuracy

因为医学二分类里只看 accuracy 不够。

---

## 6.1 指标定义说明

设正类为 `1`（如 ADHD），负类为 `0`（如 HC）：

* `Accuracy = (TP + TN) / All`
* `Precision = TP / (TP + FP)`
* `Recall = TP / (TP + FN)`
* `F1 = 2PR / (P + R)`
* `AUC = roc_auc_score(y_true, y_prob)`
* `Specificity = TN / (TN + FP)`
* `Balanced Accuracy = (Recall + Specificity) / 2`

注意：

* `AUC` 应基于概率值而不是 hard label
* 二分类 softmax 后取正类概率即可

---

## 6.2 每个 fold 建议保存的内容

至少保存：

```python
{
    "fold": fold_id,
    "subject_ids": [...],
    "y_true": [...],
    "y_pred": [...],
    "y_prob": [...],
    "acc": ...,
    "precision": ...,
    "recall": ...,
    "f1": ...,
    "auc": ...,
    "confusion_matrix": [[tn, fp], [fn, tp]]
}
```

---

## 6.3 最终汇总输出格式

最终结果请按五折结果求均值和标准差，并输出成如下风格：

```text
Accuracy (mean): 0.7311
Accuracy (std): 0.1019
Precision (mean): 0.7984
Precision (std): 0.1028
Recall (mean): 0.8524
Recall (std): 0.0909
F1-score (mean): 0.8197
F1-score (std): 0.0677
AUC (mean): 0.5693
AUC (std): xxxxx
Specificity (mean): xxxxx
Specificity (std): xxxxx
Balanced Accuracy (mean): xxxxx
Balanced Accuracy (std): xxxxx
Confusion Matrix (sum over 5 folds):
[[TN FP]
 [FN TP]]
```

### 说明

1. `mean/std` 由 5 个 fold 的指标计算得到；
2. `Confusion Matrix` 推荐输出：

   * 每折 confusion matrix
   * 以及 5 折求和后的总 confusion matrix
3. 若你只想展示一个 confusion matrix，推荐展示 **5 折测试集结果累计后的总混淆矩阵**

---

# 7. 推荐的训练函数接口

建议代码 AI 写成类似下面这种接口：

## BrainNetCNN

```python
def run_brainnetcnn_cv(
    fc_paths,
    labels,
    subject_ids,
    n_splits=5,
    random_state=42,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-4,
    num_epochs=100,
):
    """
    return:
        fold_metrics: list[dict]
        summary_metrics: dict
        all_predictions: DataFrame or list[dict]
    """
```

## BrainGNN

```python
def run_braingnn_cv(
    fc_paths,
    labels,
    subject_ids,
    n_splits=5,
    random_state=42,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-4,
    num_epochs=100,
):
    """
    return:
        fold_metrics: list[dict]
        summary_metrics: dict
        all_predictions: DataFrame or list[dict]
    """
```

---

# 8. 若实验结果不好，可优先尝试的优化顺序

## 8.1 BrainNetCNN 优先优化顺序

1. 降低模型复杂度，防止过拟合
2. 增加 dropout
3. 调小学习率
4. 用 early stopping
5. 对输入 FC 做标准化 / Fisher z
6. 尝试 class weights
7. 检查对角线是否置零
8. 检查 train/val/test 是否泄漏

---

## 8.2 BrainGNN 优先优化顺序

1. 检查图构建是否正确
2. 先不要阈值化，直接跑全连接去自环图
3. 再尝试 top-k 稀疏化
4. 调整 hidden_channels
5. 调整 pooling ratio
6. 尝试 `GraphConv` 替代 `GCNConv`
7. 在 readout 中拼接 mean/max pooling
8. 检查 edge_weight 是否数值过大或过小
9. 加 class weights
10. 降低学习率并延长训练

---

# 9. 给代码 AI 的最终一句话指令

请只实现两个深度学习模型：

1. `BrainNetCNN`
2. `BrainGNN`（允许是 BrainGNN 风格的简化版）

要求：

* 输入为 `116×116` FC txt 文件
* 支持二分类
* 支持类别不平衡处理（优先 class weights）
* 强制使用 **5-fold stratified cross-validation**
* 输出每折指标与最终 mean/std
* 输出 confusion matrix、AUC、precision、recall、F1
* 代码结构清晰，可复现，可后续继续调参优化

```

这份说明里关于 **BrainNetCNN 的 E2E/E2N/N2G 设计思想**，依据是其原始论文提出的针对脑网络矩阵的专门卷积结构；关于 **BrainGNN 的 ROI-aware 图卷积与 ROI-selection pooling 思路**，依据是其原始论文对 fMRI 脑图解释性建模的定义；而关于 `GCNConv` 支持 `edge_weight`、以及可配合图池化与图级 readout 来实现简化版 BrainGNN，则可以直接参考 PyTorch Geometric 文档。:contentReference[oaicite:0]{index=0}

另外，我在文档里把 BrainGNN 写成了“**可先实现简化版**”而不是强求一比一复现原论文，这是因为原版 BrainGNN 结构和约束项更复杂；对你当前这个任务，先做一个 **图卷积 + pooling + graph-level classifier** 的 BrainGNN-style baseline，通常更务实。:contentReference[oaicite:1]{index=1}

你这个阶段最重要的不是把两个模型写得“最花哨”，而是让另一个 AI **先写出能稳定跑五折、能正确输出 fold 指标、能处理类别不平衡的版本**。等第一轮结果出来，再决定是继续加强 BrainNetCNN 的专用卷积细节，还是加强 BrainGNN 的 pooling / 稀疏图策略。
::contentReference[oaicite:2]{index=2}
```
