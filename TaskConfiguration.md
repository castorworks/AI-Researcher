# Task Configuration 配置指南

## 概述

Task Configuration 是 AI-Researcher 系统的核心配置系统，用于定义和控制 AI 研究智能体执行的具体任务类型、难度级别和执行参数。通过这套配置系统，AI 智能体能够理解具体的研究任务要求，选择合适的实现策略，并在预定义的框架内进行创新性研究。

## 配置结构

### 1. 环境变量配置

Task Configuration 通过环境变量进行配置，主要包含以下配置项：

```bash
# ================ 任务配置 ================
# 研究领域类别（对应 ./benchmark/final）
CATEGORY=vq
# 具体实例ID（对应 ./benchmark/final/{category}/{instance_id}.json）
INSTANCE_ID=one_layer_vq
# 任务级别（task1 或 task2）
TASK_LEVEL=task1
# 最大迭代次数（0表示无限制）
MAX_ITER_TIMES=0
```

### 2. 命令行参数配置

除了环境变量，也可以通过命令行参数进行配置：

```bash
python run_infer_plan.py \
    --instance_path ../benchmark/final/vq/one_layer_vq.json \
    --task_level task1 \
    --category vq \
    --max_iter_times 0
```

## 配置项详解

### CATEGORY（研究领域类别）

**作用**: 定义 AI 智能体要处理的研究领域

**可选值**:
- `diffu_flow` - 扩散流模型
- `gnn` - 图神经网络
- `reasoning` - 推理模型
- `recommendation` - 推荐系统
- `vq` - 向量量化

**对应目录**: `./benchmark/final/{category}/`

**示例**:
```bash
CATEGORY=gnn  # 对应 ./benchmark/final/gnn/
```

#### 1. **diffu_flow（扩散流模型）**
**研究领域**: 扩散模型、概率流、生成建模

**核心概念**: 
- 基于概率路径的生成模型框架
- 通过常微分方程（ODE）定义概率路径
- 从噪声到数据样本的变换过程
- 流匹配（Flow Matching）和一致性模型

**主要应用**:
- 图像生成
- 音频合成
- 分子设计
- 数据分布建模

**技术特点**:
- 使用连续归一化流（CNF）
- 速度场参数化
- 轨迹整流和优化
- 多段训练策略

**实例列表**:
- `con_flowmatching` - 一致性流匹配
- `i_rect_flow` - 迭代整流流
- `immiscible_diffusion` - 不混溶扩散
- `mmdit` - 多模态扩散变换器

#### 2. **gnn（图神经网络）**
**研究领域**: 图结构数据学习、节点分类、图表示学习

**核心概念**:
- 图卷积网络（GCN）
- 消息传递机制
- 图注意力网络
- 图对比学习

**主要应用**:
- 社交网络分析
- 分子性质预测
- 推荐系统
- 知识图谱推理

**技术特点**:
- 处理非欧几里得数据结构
- 捕获节点间的高阶关系
- 支持半监督学习
- 可扩展性优化

**实例列表**:
- `gnn_difformer` - 扩散变换器
- `gnn_nodeformer` - 节点变换器
- `gnn_universal` - 通用图神经网络
- `graphgpt` - 图GPT
- `allinone` - 一体化GNN
- `exphormer` - 指数变换器
- `gnn_fractional` - 分数GNN
- `gnn_generalization` - 泛化GNN
- `gnn_poly_gcl` - 多项式图对比学习

#### 3. **reasoning（推理模型）**
**研究领域**: 大语言模型推理、逻辑推理、类比推理

**核心概念**:
- 思维链（Chain-of-Thought）
- 类比推理
- 自生成示例
- 上下文学习

**主要应用**:
- 数学问题求解
- 代码生成
- 逻辑推理任务
- 常识推理

**技术特点**:
- 无需人工标注示例
- 自动生成相关知识
- 上下文适应性强
- 零样本和少样本学习

**实例列表**:
- `analog_reasoner` - 类比推理器
- `self_discover` - 自我发现推理

#### 4. **recommendation（推荐系统）**
**研究领域**: 协同过滤、图推荐、对比学习

**核心概念**:
- 用户-物品交互建模
- 图神经网络推荐
- 对比学习
- 意图解耦

**主要应用**:
- 电商推荐
- 内容推荐
- 社交推荐
- 知识图谱推荐

**技术特点**:
- 处理稀疏交互数据
- 捕获高阶关系
- 自监督学习
- 多模态融合

**实例列表**:
- `categorical_rec` - 分类推荐
- `dccf` - 解耦对比协同过滤
- `hgcl` - 异构图对比学习
- `kgrec` - 知识图谱推荐
- `maerec` - 多模态推荐
- `mhcl` - 多头对比学习

#### 5. **vq（向量量化）**
**研究领域**: 离散表示学习、编码器-解码器架构

**核心概念**:
- 向量量化（Vector Quantization）
- 码本学习
- 表示崩塌问题
- 离散潜在空间

**主要应用**:
- 图像压缩
- 音频编码
- 生成模型
- 表示学习

**技术特点**:
- 连续到离散的映射
- 码本利用率优化
- 梯度估计技术
- 多模态支持

**实例列表**:
- `one_layer_vq` - 单层向量量化
- `auto_wovq` - 自动无监督向量量化
- `distangle_vq` - 解耦向量量化
- `fsq` - 有限标量量化
- `res_vq_implict` - 隐式残差向量量化
- `rotation_vq` - 旋转向量量化

### INSTANCE_ID（具体实例ID）

**作用**: 指定要处理的具体研究论文实例

**格式**: 字符串，对应具体的 JSON 文件名（不含扩展名）

**示例**:
```bash
# 对应 ./benchmark/final/vq/one_layer_vq.json
INSTANCE_ID=one_layer_vq

# 对应 ./benchmark/final/gnn/gnn_difformer.json
INSTANCE_ID=gnn_difformer
```

**可用的实例列表**:

#### VQ 领域
- `one_layer_vq` - 单层向量量化
- `auto_wovq` - 自动无监督向量量化
- `distangle_vq` - 解耦向量量化
- `fsq` - 有限标量量化
- `res_vq_implict` - 隐式残差向量量化
- `rotation_vq` - 旋转向量量化

#### GNN 领域
- `gnn_difformer` - 扩散变换器
- `gnn_nodeformer` - 节点变换器
- `gnn_universal` - 通用图神经网络
- `graphgpt` - 图GPT
- `allinone` - 一体化GNN
- `exphormer` - 指数变换器
- `gnn_fractional` - 分数GNN
- `gnn_generalization` - 泛化GNN
- `gnn_poly_gcl` - 多项式图对比学习

#### 其他领域
- `diffu_flow` - 扩散流相关实例
- `reasoning` - 推理相关实例
- `recommendation` - 推荐系统相关实例

### TASK_LEVEL（任务级别）

**作用**: 定义任务的难度和复杂度

**可选值**:
- `task1` - 基础实现任务
- `task2` - 高级分析任务

**任务级别差异**:

#### Task1（基础实现任务）
- **目标**: 实现论文的核心方法
- **内容**:
  - 模型的核心技术/算法描述
  - 主要技术组件的功能和目的
  - 详细的实现细节和参数配置
  - 组件交互的步骤说明
  - 影响性能的关键实现细节
- **适用场景**: 代码生成、模型实现

#### Task2（高级分析任务）
- **目标**: 深入分析研究问题和贡献
- **内容**:
  - 研究要解决的核心挑战
  - 现有方法的局限性
  - 研究目标和预期贡献
  - 理论分析和创新点
- **适用场景**: 研究分析、论文写作

### MAX_ITER_TIMES（最大迭代次数）

**作用**: 控制 AI 智能体的最大尝试次数

**配置值**:
- `0` - 无限制，持续尝试直到成功
- `1` - 只尝试一次
- `N` - 最多尝试 N 次

**使用场景**:
- 开发阶段：设置为 0，允许充分调试
- 生产环境：设置合理限制，避免无限循环
- 测试环境：设置较小值，快速验证

## 配置文件结构

### Benchmark 实例文件格式

每个实例文件（如 `one_layer_vq.json`）包含以下结构：

```json
{
    "target": "论文标题",
    "instance_id": "实例ID",
    "authors": ["作者列表"],
    "year": 2024,
    "url": "论文链接",
    "abstract": "论文摘要",
    "venue": "发表场所",
    "source_papers": [
        {
            "reference": "参考文献标题",
            "rank": 1,
            "type": ["类型"],
            "justification": "选择理由",
            "usage": "使用方法"
        }
    ],
    "task1": "Task1 的详细指令",
    "task2": "Task2 的详细指令"
}
```

### 任务指令内容示例

#### Task1 指令结构
```
1. **Task**: 任务描述
2. **Core Techniques/Algorithms**: 核心技术/算法
3. **Purpose and Function of Major Technical Components**: 主要技术组件的目的和功能
4. **Implementation Details**: 实现细节
5. **Step-by-Step Description of Component Interaction**: 组件交互的步骤说明
6. **Critical Implementation Details**: 关键实现细节
```

#### Task2 指令结构
```
1. 主要任务描述
2. 现有方法的局限性
3. 核心挑战
4. 研究目标和预期贡献
```

## 使用方法

### 1. 环境变量配置

创建 `.env` 文件：

```bash
# 任务配置
CATEGORY=vq
INSTANCE_ID=one_layer_vq
TASK_LEVEL=task1
MAX_ITER_TIMES=0

# 其他必要配置
COMPLETION_MODEL=openrouter/google/gemini-2.5-pro-preview-05-20
CONTAINER_NAME=paper_eval
WORKPLACE_NAME=workplace
PORT=7020
```

### 2. 命令行执行

#### 想法推理任务
```bash
python run_infer_idea.py \
    --instance_path ../benchmark/final/vq/one_layer_vq.json \
    --task_level task1 \
    --category vq \
    --model gpt-4o-2024-08-06
```

#### 代码生成任务
```bash
python run_infer_plan.py \
    --instance_path ../benchmark/final/gnn/gnn_difformer.json \
    --task_level task1 \
    --category gnn \
    --model gpt-4o-2024-08-06
```

### 3. Web GUI 配置

通过 Web 界面配置：

```bash
python web_ai_researcher.py
```

在 Web 界面中：
1. 选择研究领域（CATEGORY）
2. 选择具体实例（INSTANCE_ID）
3. 选择任务级别（TASK_LEVEL）
4. 设置最大迭代次数（MAX_ITER_TIMES）

## 配置优先级

配置项的优先级从高到低：

1. **命令行参数** - 最高优先级
2. **前端配置** - 中等优先级
3. **环境变量** - 较低优先级
4. **默认值** - 最低优先级

## 工作流程

### 1. 配置加载
```python
# 从环境变量加载配置
category = os.getenv("CATEGORY")
instance_id = os.getenv("INSTANCE_ID")
task_level = os.getenv("TASK_LEVEL")
max_iter_times = int(os.getenv("MAX_ITER_TIMES"))
```

### 2. 实例文件解析
```python
# 加载对应的 benchmark 实例文件
instance_path = f"benchmark/final/{category}/{instance_id}.json"
with open(instance_path, "r", encoding="utf-8") as f:
    eval_instance = json.load(f)

# 提取任务指令
task_instructions = eval_instance[task_level]
```

### 3. 任务执行
根据 `task_level` 和 `category` 选择合适的执行流程：
- **Task1**: 代码生成、模型实现
- **Task2**: 研究分析、论文写作

## 最佳实践

### 1. 配置管理
- 使用 `.env` 文件管理环境变量
- 为不同项目创建不同的配置文件
- 定期备份和版本控制配置

### 2. 任务选择
- **开发阶段**: 使用 `task1` 进行代码实现
- **研究阶段**: 使用 `task2` 进行深入分析
- **测试阶段**: 使用较小的 `MAX_ITER_TIMES` 值

### 3. Category 选择指南

#### 根据研究目标选择 Category

**选择 diffu_flow 的场景**:
- 需要生成高质量图像、音频或视频
- 研究概率流和扩散过程
- 开发生成模型或数据合成系统
- 需要处理连续到连续的变换任务

**选择 gnn 的场景**:
- 处理图结构数据（社交网络、分子图等）
- 需要捕获实体间的关系和依赖
- 进行节点分类、图分类或链接预测
- 研究图上的表示学习和推理

**选择 reasoning 的场景**:
- 需要增强大语言模型的推理能力
- 处理数学问题、逻辑推理或代码生成
- 研究思维链和类比推理
- 开发智能问答或决策支持系统

**选择 recommendation 的场景**:
- 构建用户-物品推荐系统
- 处理稀疏交互数据
- 需要捕获用户意图和偏好
- 研究协同过滤和图推荐算法

**选择 vq 的场景**:
- 需要学习离散表示
- 研究编码器-解码器架构
- 处理图像压缩或音频编码
- 开发生成模型中的潜在表示

#### 根据技术栈选择 Category

**深度学习新手**:
- 推荐从 `vq` 开始，概念相对简单
- 然后是 `gnn`，有丰富的教程和工具

**有图像处理经验**:
- 推荐 `diffu_flow`，与计算机视觉相关
- 或者 `vq`，在图像生成中应用广泛

**有NLP经验**:
- 推荐 `reasoning`，与语言模型推理相关
- 或者 `gnn`，在知识图谱中应用广泛

**有推荐系统经验**:
- 推荐 `recommendation`，直接相关
- 或者 `gnn`，图神经网络在推荐中应用广泛

#### 根据应用场景选择 Category

**学术研究**:
- 所有 category 都适合，根据具体研究方向选择
- 建议选择有开源代码和数据的实例

**工业应用**:
- `recommendation` 和 `gnn` 有较多实际应用
- `diffu_flow` 在生成AI中有广泛应用

**教学演示**:
- `vq` 和 `gnn` 有直观的可视化效果
- `reasoning` 可以展示AI推理过程

#### Category 对比分析

| Category | 复杂度 | 应用成熟度 | 数据要求 | 计算资源 | 主要挑战 |
|----------|--------|------------|----------|----------|----------|
| **diffu_flow** | 高 | 中等 | 大量训练数据 | GPU密集 | 训练稳定性、采样质量 |
| **gnn** | 中等 | 高 | 图结构数据 | 中等 | 可扩展性、过平滑 |
| **reasoning** | 中等 | 中等 | 推理任务数据 | CPU/GPU | 提示工程、推理准确性 |
| **recommendation** | 中等 | 高 | 用户交互数据 | 中等 | 冷启动、数据稀疏性 |
| **vq** | 低-中等 | 高 | 编码数据 | 中等 | 码本崩塌、表示质量 |

#### Category 技术栈对比

| Category | 主要框架 | 核心算法 | 评估指标 | 典型数据集 |
|----------|----------|----------|----------|------------|
| **diffu_flow** | PyTorch, JAX | ODE求解器, 流匹配 | FID, IS | CIFAR-10, ImageNet |
| **gnn** | PyTorch Geometric, DGL | 图卷积, 注意力 | 准确率, AUC | Cora, PubMed, OGB |
| **reasoning** | Transformers, LangChain | 思维链, 类比推理 | 准确率, 推理步骤 | GSM8K, MATH, Codeforces |
| **recommendation** | PyTorch, TensorFlow | 协同过滤, 图学习 | NDCG, Recall | MovieLens, Amazon, Yelp |
| **vq** | PyTorch, TensorFlow | 编码器-解码器, 量化 | 重建质量, 码本利用率 | CIFAR-10, ImageNet |

### 3. 实例选择
- 根据研究领域选择合适的 `CATEGORY`
- 根据具体需求选择合适的 `INSTANCE_ID`
- 参考实例文件中的 `source_papers` 了解相关研究

### 4. 性能优化
- 合理设置 `MAX_ITER_TIMES` 避免无限循环
- 选择合适的 LLM 模型
- 配置适当的 GPU 资源

## 故障排除

### 常见问题

1. **配置文件不存在**
   - 检查 `benchmark/final/{category}/{instance_id}.json` 是否存在
   - 确认 `CATEGORY` 和 `INSTANCE_ID` 配置正确

2. **任务指令为空**
   - 检查 `TASK_LEVEL` 是否为 `task1` 或 `task2`
   - 确认实例文件包含对应的任务指令

3. **环境变量未生效**
   - 检查 `.env` 文件格式是否正确
   - 确认环境变量已重新加载
   - 使用 `printenv` 命令验证环境变量

4. **权限问题**
   - 检查文件和目录权限
   - 确认 Docker 容器权限配置

### 调试技巧

1. **启用详细日志**
   ```bash
   export DEBUG=true
   export DEFAULT_LOG=true
   ```

2. **检查配置加载**
   ```python
   print(f"Category: {category}")
   print(f"Instance ID: {instance_id}")
   print(f"Task Level: {task_level}")
   ```

3. **验证实例文件**
   ```python
   import json
   with open(instance_path, "r") as f:
       data = json.load(f)
   print(f"Available keys: {list(data.keys())}")
   print(f"Task1 content: {data.get('task1', 'Not found')}")
   ```

## 扩展配置

### 自定义任务级别

可以通过修改实例文件添加自定义任务级别：

```json
{
    "task1": "基础任务指令",
    "task2": "高级任务指令",
    "task3": "自定义任务指令"
}
```

### 动态配置

支持运行时动态修改配置：

```python
# 动态设置环境变量
os.environ["CATEGORY"] = "gnn"
os.environ["INSTANCE_ID"] = "gnn_difformer"
os.environ["TASK_LEVEL"] = "task2"
```

## 总结

Task Configuration 是 AI-Researcher 系统的核心，它通过标准化的配置项和灵活的任务定义，使 AI 智能体能够：

1. **标准化**不同研究领域的任务定义
2. **分层化**任务的复杂度（task1 vs task2）
3. **实例化**具体的实现目标
4. **自动化**AI 智能体的工作流程

通过合理配置这些参数，用户可以精确控制 AI 智能体的行为，实现高效、准确的研究任务执行。

## 快速参考

### 常用配置组合

#### 图像生成研究
```bash
CATEGORY=diffu_flow
INSTANCE_ID=con_flowmatching
TASK_LEVEL=task1
```

#### 图神经网络应用
```bash
CATEGORY=gnn
INSTANCE_ID=gnn_difformer
TASK_LEVEL=task1
```

#### 大模型推理增强
```bash
CATEGORY=reasoning
INSTANCE_ID=analog_reasoner
TASK_LEVEL=task2
```

#### 推荐系统开发
```bash
CATEGORY=recommendation
INSTANCE_ID=dccf
TASK_LEVEL=task1
```

#### 向量量化学习
```bash
CATEGORY=vq
INSTANCE_ID=one_layer_vq
TASK_LEVEL=task1
```

### 一键运行命令

#### 想法推理
```bash
python run_infer_idea.py \
    --instance_path ../benchmark/final/vq/one_layer_vq.json \
    --task_level task1 \
    --category vq \
    --model gpt-4o-2024-08-06
```

#### 代码生成
```bash
python run_infer_plan.py \
    --instance_path ../benchmark/final/gnn/gnn_difformer.json \
    --task_level task1 \
    --category gnn \
    --model gpt-4o-2024-08-06
```

### 配置检查清单

- [ ] 确认 `CATEGORY` 选择正确
- [ ] 验证 `INSTANCE_ID` 对应的文件存在
- [ ] 选择合适的 `TASK_LEVEL`
- [ ] 设置合理的 `MAX_ITER_TIMES`
- [ ] 检查环境变量是否正确加载
- [ ] 验证 Docker 环境是否就绪
- [ ] 确认 LLM API 配置正确
