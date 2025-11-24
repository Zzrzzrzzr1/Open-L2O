# HierarchicalRNN 架构详细分析

## 目录
1. [整体架构概览](#1-整体架构概览)
2. [数学公式推导](#2-数学公式推导)
3. [数据流图](#3-数据流图)
4. [关键组件分析](#4-关键组件分析)
5. [完整计算流程](#5-完整计算流程)

---

## 1. 整体架构概览

HierarchicalRNN 是一个 **三层分层递归神经网络优化器**，通过不同粒度的信息聚合来学习最优的参数更新策略。

### 1.1 架构设计理念

```
┌──────────────────────────────────────────────────────────────────┐
│              HierarchicalRNN 三层分层信息聚合                     │
└──────────────────────────────────────────────────────────────────┘

  粒度级别          RNN 层次              信息范围           输出作用
  ========          ========              ========           ========
    细粒度    →   Level 0: 逐参数层   →  单个参数元素   →  参数更新方向
                    (Per-Parameter)        (独立处理)       学习率调整
                                                            衰减系数
      ↑
      │ 聚合 (mean)
      │
    中粒度    →   Level 1: 逐张量层   →  整个张量       →  为 Level 0
                    (Per-Tensor)          (W1, b1等)       提供偏置
      ↑
      │ 聚合 (mean)
      │
    粗粒度    →   Level 2: 全局层     →  所有参数       →  为 Level 1
                    (Global)              (整个模型)       提供偏置
```

### 1.2 核心超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `level_sizes` | [10, 20, 20] | 三层 RNN 的隐藏层大小 |
| `num_gradient_scales` | 4 | 多尺度梯度累积的数量 |
| `learnable_decay` | True | 是否学习衰减系数 |
| `dynamic_output_scale` | True | 是否学习学习率缩放 |
| `use_gradient_shortcut` | True | 是否使用梯度捷径连接 |

---

## 2. 数学公式推导

### 2.1 输入预处理：多尺度梯度累积

对于参数 $\theta$ 的梯度 $g_t$，计算 **4 个不同时间尺度** 的累积梯度：

#### 2.1.1 输入衰减系数 (Input Decay)

$$
\begin{aligned}
\delta_{\text{inp}}^{(0)} &= \text{inp\_decay} \\
\delta_{\text{inp}}^{(i)} &= \sqrt{\delta_{\text{inp}}^{(i-1)}}, \quad i = 1, 2, 3
\end{aligned}
$$

**说明**：衰减系数逐渐减小，形成 4 个不同时间尺度：
- $\delta_{\text{inp}}^{(0)}$: 快速变化（短期）
- $\delta_{\text{inp}}^{(3)}$: 缓慢变化（长期）

#### 2.1.2 梯度累积 (Gradient Accumulation)

$$
\tilde{g}_t^{(i)} = g_t \cdot (1 - \delta_{\text{inp}}^{(i)}) + \tilde{g}_{t-1}^{(i)} \cdot \delta_{\text{inp}}^{(i)}, \quad i = 0, 1, 2, 3
$$

**物理意义**：这是一个指数移动平均 (EMA)，不同的 $i$ 对应不同的平滑程度：
- $i=0$: 接近当前梯度（短期趋势）
- $i=3$: 长期平均梯度（长期趋势）

#### 2.1.3 RMS 缩放 (Root Mean Square Scaling)

对每个累积梯度进行 RMS 归一化：

$$
\begin{aligned}
m_t^{(i)} &= \delta_{\text{scl}}^{(i)} \cdot m_{t-1}^{(i)} + (1 - \delta_{\text{scl}}^{(i)}) \cdot (\tilde{g}_t^{(i)})^2 \\
\bar{g}_t^{(i)} &= \frac{\tilde{g}_t^{(i)}}{\sqrt{m_t^{(i)} + \epsilon}}
\end{aligned}
$$

其中：
- $m_t^{(i)}$: 梯度平方的移动平均（均方值）
- $\bar{g}_t^{(i)}$: RMS 归一化后的梯度
- $\epsilon = 10^{-16}$: 数值稳定项

**作用**：
- 消除梯度尺度差异
- 提供梯度信噪比信息（大的 $m$ 表示信号强）

### 2.2 RNN 输入特征构建

将多尺度梯度和额外特征拼接成 RNN 输入 $\mathbf{x}_t$：

$$
\mathbf{x}_t = \left[ \bar{g}_t^{(0)}, \bar{g}_t^{(1)}, \bar{g}_t^{(2)}, \bar{g}_t^{(3)}, \mathbf{f}_{\text{extra}} \right]
$$

#### 2.2.1 额外特征 $\mathbf{f}_{\text{extra}}$

**1. 梯度交叉项 (Gradient Products)** - 曲率信息

$$
\mathbf{p}_t = \left[ \bar{g}_t^{(0)} \odot \bar{g}_t^{(1)}, \bar{g}_t^{(1)} \odot \bar{g}_t^{(2)}, \bar{g}_t^{(2)} \odot \bar{g}_t^{(3)} \right]
$$

**物理意义**：相邻尺度梯度的乘积反映了梯度的曲率变化

**2. 对数均方值 (Log Mean Squared)** - 信噪比

$$
\begin{aligned}
\mathbf{l}_t^{(i)} &= \log(m_t^{(i)} + \epsilon) \\
\bar{\mathbf{l}}_t &= \frac{1}{4} \sum_{i=0}^{3} \mathbf{l}_t^{(i)} \\
\mathbf{l}_t^{\text{rel}} &= \left[ \mathbf{l}_t^{(0)} - \bar{\mathbf{l}}_t, \ldots, \mathbf{l}_t^{(3)} - \bar{\mathbf{l}}_t \right]
\end{aligned}
$$

**物理意义**：
- 大的 $\mathbf{l}_t^{(i)}$ 表示该尺度梯度信号强
- 相对值反映不同尺度的信噪比差异

**3. 相对学习率 (Relative Learning Rate)**

$$
\mathbf{r}_t = \log(\alpha_t) - \frac{1}{N} \sum_{j=1}^{N} \log(\alpha_t^{(j)})
$$

**物理意义**：当前参数的学习率相对于张量平均学习率的偏差

**最终 RNN 输入维度**：
$$
\dim(\mathbf{x}_t) = 4 + 3 + 4 + 1 = 12
$$

### 2.3 Level 0: Per-Parameter RNN

对于参数张量 $\theta \in \mathbb{R}^{d_1 \times d_2 \times \ldots}$，展平为 $\mathbf{\theta} \in \mathbb{R}^n$ ($n = d_1 \cdot d_2 \cdots$)，每个元素独立处理。

#### 2.3.1 BiasGRU 单元

BiasGRU 是一个改进的 GRU，**外部偏置**由上层 RNN 提供：

$$
\begin{aligned}
\mathbf{b}^{(0)} &= \begin{cases}
    \mathbf{W}_{1 \to 0} \mathbf{h}_t^{(1)} + \mathbf{W}_{2 \to 0} \mathbf{h}_t^{(2)} & \text{if } L = 3 \\
    \mathbf{W}_{1 \to 0} \mathbf{h}_t^{(1)} & \text{if } L = 2 \\
    \mathbf{0} & \text{if } L = 1
\end{cases} \\
\end{aligned}
$$

其中：
- $\mathbf{h}_t^{(1)} \in \mathbb{R}^{20}$: Level 1 的隐藏状态
- $\mathbf{h}_t^{(2)} \in \mathbb{R}^{20}$: Level 2 的隐藏状态
- $\mathbf{W}_{1 \to 0} \in \mathbb{R}^{30 \times 20}$: Level 1 到 Level 0 的投影矩阵（30 = 3 × 10，因为 GRU 需要 reset、update、new 三个门的偏置）

#### BiasGRU 更新公式

$$
\begin{aligned}
\mathbf{r}_t &= \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1}^{(0)} + \mathbf{b}_r + \mathbf{b}^{(0)}_{[1:10]}) \\
\mathbf{z}_t &= \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1}^{(0)} + \mathbf{b}_z + \mathbf{b}^{(0)}_{[11:20]}) \\
\tilde{\mathbf{h}}_t &= \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}^{(0)}) + \mathbf{b}_h + \mathbf{b}^{(0)}_{[21:30]}) \\
\mathbf{h}_t^{(0)} &= (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1}^{(0)} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
\end{aligned}
$$

**输出**：$\mathbf{h}_t^{(0)} \in \mathbb{R}^{n \times 10}$

### 2.4 Level 1: Per-Tensor RNN

#### 2.4.1 输入聚合

将 Level 0 的所有隐藏状态取平均：

$$
\mathbf{x}_t^{(1)} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_{t,i}^{(0)} \in \mathbb{R}^{10}
$$

#### 2.4.2 BiasGRU 计算

$$
\begin{aligned}
\mathbf{b}^{(1)} &= \begin{cases}
    \mathbf{W}_{2 \to 1} \mathbf{h}_t^{(2)} & \text{if } L = 3 \\
    \mathbf{0} & \text{if } L = 2
\end{cases} \\
\mathbf{h}_t^{(1)} &= \text{BiasGRU}(\mathbf{x}_t^{(1)}, \mathbf{h}_{t-1}^{(1)}, \mathbf{b}^{(1)}) \in \mathbb{R}^{20}
\end{aligned}
$$

**作用**：
- 捕获整个张量的整体优化趋势
- 为 Level 0 提供张量级别的指导信息

### 2.5 Level 2: Global RNN

#### 2.5.1 输入聚合

对所有张量的 Level 1 状态取平均：

$$
\mathbf{x}_t^{(2)} = \frac{1}{M} \sum_{j=1}^{M} \mathbf{h}_{t,j}^{(1)} \in \mathbb{R}^{20}
$$

其中 $M$ 是模型中参数张量的总数（例如 MNIST MLP 有 4 个：W1, b1, W2, b2）

#### 2.5.2 GRU 计算

$$
\mathbf{h}_t^{(2)} = \text{GRU}(\mathbf{x}_t^{(2)}, \mathbf{h}_{t-1}^{(2)}) \in \mathbb{R}^{20}
$$

**作用**：
- 捕获全局优化策略
- 协调不同张量之间的优化行为

### 2.6 参数更新计算

#### 2.6.1 更新方向 (Update Delta)

从 Level 0 的隐藏状态投影得到更新方向：

$$
\Delta_t = \mathbf{W}_{\text{update}} \mathbf{h}_t^{(0)} \in \mathbb{R}^{n \times 1}
$$

**可选：梯度捷径 (Gradient Shortcut)**

如果启用 `use_gradient_shortcut`：

$$
\Delta_t = \mathbf{W}_{\text{update}} \mathbf{h}_t^{(0)} + \mathbf{W}_{\text{grad}} \left[ \bar{g}_t^{(0)}, \bar{g}_t^{(1)}, \bar{g}_t^{(2)}, \bar{g}_t^{(3)} \right]
$$

**物理意义**：允许 RNN 直接使用梯度信息，释放隐藏状态用于存储更复杂的优化策略

#### 2.6.2 动态输出缩放 (Dynamic Output Scale)

如果启用 `dynamic_output_scale`：

$$
\Delta_t \leftarrow \frac{\Delta_t}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} (\Delta_{t,i})^2 + \epsilon}}
$$

**作用**：归一化更新方向的模长，防止更新步长过大

#### 2.6.3 学习率计算

学习率通过 Level 0 隐藏状态动态计算：

$$
\begin{aligned}
\text{lr\_change}_t &= \mathbf{W}_{\text{lr}} \mathbf{h}_t^{(0)} + \mathbf{b}_{\text{lr}} \\
\log(\alpha_t^{\text{step}}) &= \log(\alpha_{t-1}) + \text{lr\_change}_t \\
\log(\alpha_t^{\text{step}}) &\leftarrow \text{clip}(\log(\alpha_t^{\text{step}}), -33, 33) \\
\log(\alpha_t) &= \beta \cdot \log(\alpha_{t-1}) + (1 - \beta) \cdot \log(\alpha_t^{\text{step}}) \\
\alpha_t &= \exp(\log(\alpha_t^{\text{step}}) + c)
\end{aligned}
$$

其中：
- $\beta = \sigma(\text{lr\_momentum\_logit})$: 学习率动量（默认约 0.96）
- $c = \text{param\_stepsize\_offset}$: 全局学习率偏移量（默认 -1）

#### 2.6.4 衰减系数计算

**Scale Decay**（用于 RMS 缩放）：

$$
\delta_{\text{scl}, t} = \sigma(\mathbf{W}_{\text{scl}} \mathbf{h}_t^{(0)} + b_{\text{scl}})
$$

**Input Decay**（用于梯度累积）：

$$
\delta_{\text{inp}, t} = \sigma(\mathbf{W}_{\text{inp}} \mathbf{h}_t^{(0)} + b_{\text{inp}})
$$

**物理意义**：
- RNN 学习到自适应的衰减系数
- 不同参数可以有不同的累积时间尺度

#### 2.6.5 最终参数更新

$$
\begin{aligned}
\mathbf{u}_t &= \alpha_t \odot \Delta_t \\
\theta_t &= \theta_{t-1} - \text{reshape}(\mathbf{u}_t, \text{shape}(\theta))
\end{aligned}
$$

---

## 3. 数据流图

### 3.1 完整数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    HierarchicalRNN 完整数据流图                           │
└──────────────────────────────────────────────────────────────────────────┘

输入: 参数 θ ∈ ℝ^(d₁×d₂), 梯度 g_t ∈ ℝ^(d₁×d₂)
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│            【步骤 1】梯度预处理：多尺度累积                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  g_t → Flatten → g̃ ∈ ℝ^n                                      │
│         │                                                      │
│         ├─────► [尺度 0] δ_inp^(0) = 0.89                     │
│         │       g̃_t^(0) = g̃_t(1-δ) + g̃_{t-1}^(0)·δ          │
│         │       m_t^(0) = δ_scl·m_{t-1}^(0) + (1-δ_scl)(g̃^(0))² │
│         │       ḡ_t^(0) = g̃_t^(0) / √(m_t^(0) + ε)           │
│         │                          [n×1]                       │
│         │                                                      │
│         ├─────► [尺度 1] δ_inp^(1) = √δ_inp^(0) ≈ 0.94       │
│         │       ḡ_t^(1) = ...      [n×1]                      │
│         │                                                      │
│         ├─────► [尺度 2] δ_inp^(2) ≈ 0.97                     │
│         │       ḡ_t^(2) = ...      [n×1]                      │
│         │                                                      │
│         └─────► [尺度 3] δ_inp^(3) ≈ 0.98                     │
│                 ḡ_t^(3) = ...      [n×1]                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│            【步骤 2】RNN 输入特征构建                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  [4 个缩放梯度]  ḡ_t^(0), ḡ_t^(1), ḡ_t^(2), ḡ_t^(3)         │
│                  [n×1]  [n×1]  [n×1]  [n×1]                   │
│         │                                                      │
│         ├─► [3 个梯度交叉项] (曲率信息)                       │
│         │   p₁ = ḡ_t^(0) ⊙ ḡ_t^(1)  [n×1]                    │
│         │   p₂ = ḡ_t^(1) ⊙ ḡ_t^(2)  [n×1]                    │
│         │   p₃ = ḡ_t^(2) ⊙ ḡ_t^(3)  [n×1]                    │
│         │                                                      │
│         ├─► [4 个对数均方值] (信噪比)                         │
│         │   l_t^(0) = log(m_t^(0)) - l̄_t  [n×1]              │
│         │   l_t^(1) = log(m_t^(1)) - l̄_t  [n×1]              │
│         │   l_t^(2) = log(m_t^(2)) - l̄_t  [n×1]              │
│         │   l_t^(3) = log(m_t^(3)) - l̄_t  [n×1]              │
│         │                                                      │
│         └─► [相对学习率]                                       │
│             r_t = log(α_t) - mean(log(α))  [n×1]              │
│                                                                │
│  拼接: x_t = [ḡ^(0), ḡ^(1), ḡ^(2), ḡ^(3),                    │
│               p₁, p₂, p₃, l^(0), l^(1), l^(2), l^(3), r_t]   │
│                         [n × 12]                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│   【步骤 3】Level 2 (Global RNN) - 全局优化策略               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  对所有张量的 h^(1) 聚合:                                      │
│                                                                │
│  ┌──────────────────────────────────────────┐                 │
│  │ 张量1: h_{t,1}^(1) ∈ ℝ^20                │                 │
│  │ 张量2: h_{t,2}^(1) ∈ ℝ^20                │                 │
│  │ 张量3: h_{t,3}^(1) ∈ ℝ^20      ┌────┐   │                 │
│  │ 张量4: h_{t,4}^(1) ∈ ℝ^20   ───►│Mean│   │                 │
│  │       ...                       └────┘   │                 │
│  └──────────────────────────────────────────┘                 │
│                    │                                           │
│                    ▼                                           │
│           x_t^(2) = mean(h^(1)) ∈ ℝ^20                        │
│                    │                                           │
│                    ▼                                           │
│        ┌───────────────────────────┐                          │
│        │  BiasGRUCell (无外部偏置) │                          │
│        │  state_size = 20          │                          │
│        └───────────────────────────┘                          │
│                    │                                           │
│                    ▼                                           │
│              h_t^(2) ∈ ℝ^20                                   │
│              (全局隐藏状态)                                    │
│                    │                                           │
│                    └───────────┐                               │
│                                │                               │
└────────────────────────────────┼───────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────┴───────────────────────────────┐
│   【步骤 4】Level 1 (Per-Tensor RNN) - 张量级优化             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  对当前张量的 h^(0) 聚合:                                      │
│                                                                │
│  h_{t,1}^(0), h_{t,2}^(0), ..., h_{t,n}^(0) ∈ ℝ^10           │
│         │          │              │                            │
│         └──────────┴──────────────┘                            │
│                    │                                           │
│                    ▼                                           │
│        x_t^(1) = mean(h^(0)) ∈ ℝ^10                           │
│                    │                                           │
│                    ▼                                           │
│        ┌──────────────────────────────┐                       │
│        │  BiasGRUCell                 │                       │
│        │  state_size = 20             │                       │
│        │  bias = W_{2→1}·h_t^(2) ────┼──┐ (来自 Level 2)     │
│        └──────────────────────────────┘  │                    │
│                    │                      │                    │
│                    ▼                      │                    │
│              h_t^(1) ∈ ℝ^20 ◄────────────┘                    │
│              (张量隐藏状态)                                    │
│                    │                                           │
│                    └───────────┐                               │
│                                │                               │
└────────────────────────────────┼───────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────┴───────────────────────────────┐
│   【步骤 5】Level 0 (Per-Parameter RNN) - 逐参数优化          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  对每个参数元素 i ∈ {1, ..., n}:                              │
│                                                                │
│         x_{t,i} ∈ ℝ^12  (第 i 个参数的输入特征)              │
│              │                                                 │
│              ▼                                                 │
│  ┌──────────────────────────────────────────┐                 │
│  │  BiasGRUCell                             │                 │
│  │  state_size = 10                         │                 │
│  │  bias = W_{1→0}·h_t^(1) + W_{2→0}·h_t^(2)│                 │
│  │         [30×20]·[20]   +  [30×20]·[20]   │                 │
│  │                                           │                 │
│  │  (来自 Level 1)    (来自 Level 2)        │                 │
│  └──────────────────────────────────────────┘                 │
│              │                                                 │
│              ▼                                                 │
│         h_{t,i}^(0) ∈ ℝ^10                                    │
│         (第 i 个参数的隐藏状态)                                │
│              │                                                 │
│              ▼                                                 │
│  收集所有: H_t^(0) = [h_{t,1}^(0), ..., h_{t,n}^(0)]^T       │
│                    ∈ ℝ^(n×10)                                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────────┐
│            【步骤 6】投影与参数更新计算                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. 更新方向:                                                  │
│     ┌────────────────────────────────────┐                    │
│     │ Δ_t = W_update · H_t^(0)           │                    │
│     │       [10×1]^T · [n×10]^T          │                    │
│     │     = [n×1]                         │                    │
│     │                                     │                    │
│     │ (可选) 梯度捷径:                    │                    │
│     │ Δ_t += W_grad · [ḡ^(0:3)]         │                    │
│     └────────────────────────────────────┘                    │
│              │                                                 │
│              ▼                                                 │
│     ┌────────────────────────────────────┐                    │
│     │ (可选) 动态缩放:                    │                    │
│     │ Δ_t = Δ_t / √(mean(Δ_t²) + ε)     │                    │
│     └────────────────────────────────────┘                    │
│              │                                                 │
│              ▼                                                 │
│  2. 学习率:                                                    │
│     ┌────────────────────────────────────┐                    │
│     │ lr_change = W_lr · H_t^(0) + b_lr  │                    │
│     │ log(α_t^step) = log(α_{t-1}) + lr_change               │
│     │ log(α_t^step) = clip(log(α_t^step), -33, 33)           │
│     │ log(α_t) = β·log(α_{t-1}) + (1-β)·log(α_t^step)        │
│     │ α_t = exp(log(α_t^step) + offset)                       │
│     │     ∈ ℝ^(n×1)                       │                    │
│     └────────────────────────────────────┘                    │
│              │                                                 │
│              ▼                                                 │
│  3. 衰减系数更新:                                              │
│     ┌────────────────────────────────────┐                    │
│     │ δ_scl,t = σ(W_scl·H_t^(0) + b_scl) │                    │
│     │ δ_inp,t = σ(W_inp·H_t^(0) + b_inp) │                    │
│     │          ∈ ℝ^(n×1)                  │                    │
│     └────────────────────────────────────┘                    │
│              │                                                 │
│              ▼                                                 │
│  4. 最终更新:                                                  │
│     ┌────────────────────────────────────┐                    │
│     │ u_t = α_t ⊙ Δ_t  ∈ ℝ^(n×1)        │                    │
│     │ θ_t = θ_{t-1} - reshape(u_t, shape(θ))                 │
│     └────────────────────────────────────┘                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                     │
                     ▼
           输出: 更新后的参数 θ_t
```

### 3.2 信息流向总结

```
┌────────────────────────────────────────────────────────────────┐
│                  三层 RNN 信息流向拓扑图                       │
└────────────────────────────────────────────────────────────────┘

  Level 2 (Global)         Level 1 (Per-Tensor)    Level 0 (Per-Parameter)
  ════════════════         ════════════════════    ═══════════════════════

  ┌──────────┐                                     ┌──────────┐
  │ h_t^(2)  │──┐                              ┌──→│h_{t,1}^(0)│→ θ₁更新
  │  [20]    │  │                              │   └──────────┘
  └──────────┘  │                              │   ┌──────────┐
       ↑        │     ┌──────────┐             ├──→│h_{t,2}^(0)│→ θ₂更新
       │        └────→│ h_{t,1}^(1)│────────────┤   └──────────┘
       │              │   [20]   │             │        ...
       │              └──────────┘             │   ┌──────────┐
       │                   ↑                   └──→│h_{t,n}^(0)│→ θₙ更新
       │                   │ mean                  │  [10]    │
       │                   │                       └──────────┘
       │              ┌────┴────┐                       ↑
       │              │         │                       │
  mean(h^(1))    h_{t,1}^(0)  h_{t,2}^(0)  ...         │
       │                                                │
       └────────────────────────────────────────────────┘
                          x_t [12]
                       (RNN 输入特征)

  信息流向:
  ────────
  → : 前向传播 (聚合)
  ──→: 偏置注入 (指导)
```

**关键洞察**：
1. **自下而上聚合**：Level 0 → Level 1 → Level 2（通过 mean 操作）
2. **自上而下指导**：Level 2 → Level 1 → Level 0（通过 bias 注入）
3. **并行处理**：Level 0 的 n 个参数元素独立并行处理
4. **信息共享**：高层 RNN 捕获全局模式，低层 RNN 处理局部细节

---

## 4. 关键组件分析

### 4.1 BiasGRU 单元

#### 4.1.1 为什么使用 BiasGRU？

标准 GRU 的偏置是**固定的可学习参数**，而 BiasGRU 的偏置**由外部动态提供**：

**标准 GRU**:
$$
\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \color{red}{\mathbf{b}_r})
$$
- $\mathbf{b}_r$ 是固定的可学习参数

**BiasGRU**:
$$
\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r + \color{green}{\mathbf{b}_{\text{external}}})
$$
- $\mathbf{b}_{\text{external}}$ 来自上层 RNN 的输出，**随时间动态变化**

**优势**：
- 实现层次化信息传递
- 高层 RNN 可以通过偏置**调制**低层 RNN 的行为
- 避免层之间直接的状态耦合

### 4.2 多尺度梯度累积

#### 4.2.1 时间尺度对比

| 尺度 $i$ | $\delta_{\text{inp}}^{(i)}$ | 半衰期 (步数) | 物理意义 |
|---------|----------------------------|--------------|---------|
| 0 | 0.89 | ~6 步 | 短期趋势（最近梯度） |
| 1 | 0.94 | ~12 步 | 中短期趋势 |
| 2 | 0.97 | ~23 步 | 中长期趋势 |
| 3 | 0.98 | ~35 步 | 长期趋势（历史梯度） |

**半衰期计算**：$T_{1/2} = \frac{\log(0.5)}{\log(\delta)}$

#### 4.2.2 为什么需要多尺度？

**1. 捕获不同频率的优化动态**
- 高频（短期）：快速响应梯度变化，适应局部地形
- 低频（长期）：平滑噪声，保持全局优化方向

**2. 提供曲率信息**
- 梯度交叉项 $\bar{g}^{(i)} \odot \bar{g}^{(j)}$ 反映不同尺度的梯度一致性
- 如果短期和长期梯度方向一致 → 低曲率区域 → 可以增大步长
- 如果不一致 → 高曲率区域 → 应该减小步长

**3. 信噪比估计**
- $m_t^{(i)}$ 大表示该尺度梯度信号强
- 不同尺度的 $m$ 差异反映梯度的噪声水平

### 4.3 动态学习率机制

#### 4.3.1 学习率更新策略

HierarchicalRNN 的学习率是 **per-parameter** 的，每个参数元素都有独立的学习率：

$$
\alpha_t^{(i)} = \exp\left( \log(\alpha_{t-1}^{(i)}) + \text{lr\_change}_t^{(i)} + c \right)
$$

**关键特性**：

1. **对数空间更新**
   - 在 log 空间更新避免学习率变为负数
   - $\text{lr\_change} \in (-\infty, +\infty)$ → $\alpha \in (0, +\infty)$

2. **动量平滑**
   $$
   \log(\alpha_t) = 0.96 \cdot \log(\alpha_{t-1}) + 0.04 \cdot \log(\alpha_t^{\text{step}})
   $$
   - 防止学习率剧烈波动
   - 类似 Adam 的一阶矩估计

3. **剪切保护**
   $$
   \log(\alpha_t^{\text{step}}) \in [-33, 33]
   $$
   - 防止数值溢出
   - $\alpha \in [e^{-33}, e^{33}] \approx [10^{-15}, 10^{14}]$

4. **全局偏移**
   - $c = \text{param\_stepsize\_offset}$ (默认 -1)
   - 整体调节学习率水平

#### 4.3.2 学习率的自适应性

不同参数的学习率可以根据其优化特性自动调整：

**示例**：
- **梯度稳定的参数**（如 bias）：RNN 学习增大 $\alpha$
- **梯度波动的参数**（如权重）：RNN 学习减小 $\alpha$
- **接近最优的参数**：RNN 学习逐渐减小 $\alpha$

---

## 5. 完整计算流程

### 5.1 单步更新伪代码

```python
def hierarchical_rnn_step(theta, grad, states, global_state):
    """
    单步 HierarchicalRNN 更新

    参数:
        theta: 当前参数 [d1, d2, ...]
        grad: 当前梯度 [d1, d2, ...]
        states: 优化器状态字典
        global_state: 全局 RNN 状态 [1, 20]

    返回:
        theta_new: 更新后的参数
        states_new: 更新后的状态
        global_state_new: 更新后的全局状态
    """
    # ========== 步骤 1: 梯度预处理 ==========
    grad_flat = flatten(grad)  # [n, 1]

    # 多尺度梯度累积
    grad_scaled = []
    ms_list = []
    for i in range(4):
        # 输入衰减
        delta_inp_i = sqrt^i(states['inp_decay'])
        grad_accum_i = grad_flat * (1 - delta_inp_i) + states[f'grad_accum{i}'] * delta_inp_i

        # RMS 缩放
        delta_scl_i = states['scl_decay']
        ms_i = delta_scl_i * states[f'ms{i}'] + (1 - delta_scl_i) * grad_accum_i**2
        grad_scaled_i = grad_accum_i / sqrt(ms_i + 1e-16)

        grad_scaled.append(grad_scaled_i)
        ms_list.append(ms_i)

    # ========== 步骤 2: RNN 输入特征 ==========
    # 梯度交叉项
    grad_products = [grad_scaled[i] * grad_scaled[i+1] for i in range(3)]

    # 对数均方值
    log_ms = [log(ms + 1e-16) for ms in ms_list]
    log_ms_mean = mean(log_ms)
    log_ms_relative = [lm - log_ms_mean for lm in log_ms]

    # 相对学习率
    log_lr = states['log_learning_rate']
    relative_lr = log_lr - mean(log_lr)

    # 拼接
    rnn_input = concat([grad_scaled, grad_products, log_ms_relative, relative_lr], axis=1)
    # shape: [n, 12]

    # ========== 步骤 3: Level 0 RNN ==========
    # 计算外部偏置
    bias_from_level1 = affine(states['layer'], 30)  # [1, 20] -> [1, 30]
    bias_from_level2 = affine(global_state, 30)     # [1, 20] -> [1, 30]
    bias_total = bias_from_level1 + bias_from_level2  # [1, 30]

    # BiasGRU
    h_param_new = BiasGRUCell(
        input=rnn_input,         # [n, 12]
        state=states['parameter'], # [n, 10]
        bias=bias_total           # [1, 30], 广播到 [n, 30]
    )  # -> [n, 10]

    # ========== 步骤 4: Level 1 RNN ==========
    # 聚合 Level 0 状态
    layer_input = mean(h_param_new, axis=0, keepdims=True)  # [1, 10]

    # 计算外部偏置
    bias_from_level2_to_1 = affine(global_state, 60)  # [1, 20] -> [1, 60]

    # BiasGRU
    h_layer_new = BiasGRUCell(
        input=layer_input,       # [1, 10]
        state=states['layer'],   # [1, 20]
        bias=bias_from_level2_to_1  # [1, 60]
    )  # -> [1, 20]

    # ========== 步骤 5: 参数更新计算 ==========
    # 5.1 更新方向
    delta = W_update @ h_param_new  # [10, 1]^T @ [n, 10]^T -> [n, 1]

    # 可选：梯度捷径
    delta += W_grad @ concat(grad_scaled, axis=1)

    # 动态缩放
    delta = delta / sqrt(mean(delta**2) + 1e-16)

    # 5.2 学习率
    lr_change = W_lr @ h_param_new + b_lr  # [n, 1]
    log_lr_step = log_lr + lr_change
    log_lr_step = clip(log_lr_step, -33, 33)
    log_lr_new = 0.96 * log_lr + 0.04 * log_lr_step
    lr_param = exp(log_lr_step + offset)

    # 5.3 衰减系数
    scl_decay_new = sigmoid(W_scl @ h_param_new + b_scl)
    inp_decay_new = sigmoid(W_inp @ h_param_new + b_inp)

    # 5.4 最终更新
    update_step = lr_param * delta  # [n, 1]
    theta_flat_new = flatten(theta) - update_step
    theta_new = reshape(theta_flat_new, shape(theta))

    # ========== 步骤 6: 状态更新 ==========
    states_new = {
        'parameter': h_param_new,
        'layer': h_layer_new,
        'scl_decay': scl_decay_new,
        'inp_decay': inp_decay_new,
        'true_param': theta_new,
        'log_learning_rate': log_lr_new,
        'grad_accum0': grad_accum_0, ...
        'ms0': ms_0, ...
    }

    return theta_new, states_new
```

### 5.2 完整训练循环

```python
# 初始化
theta = random_init()
states = initialize_states(theta)
global_state = initialize_global_state()  # [1, 20]

# 元训练循环
for meta_iter in range(num_meta_iterations):
    # 随机初始化 optimizee
    theta_optimizee = random_init()

    # Unroll 循环
    for unroll_step in range(num_unrolls):
        for t in range(unroll_length):  # 20 步
            # 前向传播计算梯度
            loss = problem.objective(theta_optimizee, data, labels)
            grad = tf.gradients(loss, theta_optimizee)

            # HierarchicalRNN 更新
            theta_optimizee, states = hierarchical_rnn_step(
                theta_optimizee, grad, states, global_state
            )

        # 更新 Level 2 Global RNN
        global_input = mean(all_layer_states)  # [1, 20]
        global_state = GRUCell(global_input, global_state)

    # 计算元训练损失
    meta_loss = scaled_objective + regularization

    # 反向传播更新 HierarchicalRNN 的参数
    meta_grads = tf.gradients(meta_loss, hrnn_params)
    hrnn_params = RMSProp.step(hrnn_params, meta_grads)
```

---

## 6. 总结

### 6.1 HierarchicalRNN 的核心创新

| 特性 | 创新点 | 优势 |
|------|--------|------|
| **三层分层结构** | Level 0 → Level 1 → Level 2 信息聚合 | 捕获不同粒度的优化模式 |
| **BiasGRU 机制** | 外部动态偏置注入 | 实现层次化信息传递 |
| **多尺度梯度** | 4 个时间尺度的梯度累积 | 捕获短期和长期优化动态 |
| **Per-Parameter 学习率** | 每个参数独立的自适应学习率 | 精细化参数更新控制 |
| **丰富的输入特征** | 梯度交叉项、信噪比、相对学习率 | 提供优化曲率和信噪比信息 |

### 6.2 与其他优化器的对比

| 优化器 | 学习率 | 历史信息 | 参数间交互 | 可学习性 |
|--------|--------|---------|-----------|---------|
| **SGD** | 固定/手动调度 | 无 | 无 | 否 |
| **Adam** | 自适应 (per-param) | 一阶矩、二阶矩 | 无 | 否 |
| **HierarchicalRNN** | 自适应 (per-param) | RNN 隐藏状态 | 三层信息共享 | 是（元学习） |

### 6.3 计算复杂度

对于参数量 $n$ 的单个张量：

| 组件 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 梯度预处理 | $O(n)$ | $O(n)$ |
| Level 0 RNN | $O(n \cdot 10 \cdot 12)$ | $O(n \cdot 10)$ |
| Level 1 RNN | $O(20 \cdot 10)$ | $O(20)$ |
| Level 2 RNN | $O(20 \cdot 20)$ | $O(20)$ |
| 投影计算 | $O(n \cdot 10)$ | $O(n)$ |
| **总计** | $O(n)$ | $O(n)$ |

**结论**：HierarchicalRNN 的时间和空间复杂度都是 **线性** 的，与参数量成正比，适合大规模优化问题。

---

## 附录：代码位置索引

| 功能 | 代码位置 |
|------|---------|
| 类定义 | `hierarchical_rnn.py:61-806` |
| 初始化 RNN cells | `hierarchical_rnn.py:220-242` |
| 多尺度梯度计算 | `hierarchical_rnn.py:444-495` |
| RNN 输入特征扩展 | `hierarchical_rnn.py:497-541` |
| Level 0/1 RNN 更新 | `hierarchical_rnn.py:542-604` |
| 投影与学习率计算 | `hierarchical_rnn.py:606-706` |
| Level 2 Global RNN | `hierarchical_rnn.py:708-728` |
| 完整更新流程 | `hierarchical_rnn.py:353-430` |
