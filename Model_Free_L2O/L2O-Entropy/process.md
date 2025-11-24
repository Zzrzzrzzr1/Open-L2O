# L2O-Entropy 项目流程文档

## 1. 项目概述

### 1.1 核心思想

**L2O-Entropy** 是论文 "Learning to Generalize Provably in Learning to Optimize" 的实现，专注于 **Model-Free Learning to Optimize (L2O)** 范式。该项目通过元学习训练一个**可学习优化器（Learnable Optimizer）**，使其能够高效优化各种机器学习问题。

**三大创新点**：
- **Hessian 正则化**：通过 Hessian 信息约束优化器的泛化能力，防止过拟合
- **课程学习 (Curriculum Learning)**：渐进式增加训练步数，提高训练稳定性
- **模仿学习 (Imitation Learning / MT)**：让可学习优化器模仿 Adam 等成熟优化器，加速收敛

### 1.2 项目结构

```
L2O-Entropy/
├── README.md                           # 项目说明
├── process.md                          # 本流程文档
└── L2O-ScalewHessian/
    ├── README.md                       # L2O-Scale 详细说明
    ├── L2O-Scale-Training/             # 训练代码目录
    │   ├── metarun.py                  # 主训练脚本
    │   ├── metaopt.py                  # 核心训练逻辑
    │   ├── metatest.py                 # 测试脚本
    │   ├── mt_utils.py                 # 模仿学习工具
    │   ├── dtata_process.py            # 数据处理
    │   ├── optimizer/                  # 可学习优化器模块
    │   │   ├── trainable_optimizer.py  # 基类
    │   │   ├── hierarchical_rnn.py     # 分层 RNN 优化器 (核心)
    │   │   ├── coordinatewise_rnn.py   # 逐坐标 RNN 优化器
    │   │   ├── rnn_cells.py            # 自定义 RNN 单元
    │   │   ├── trainable_adam.py       # 可训练 Adam
    │   │   └── utils.py                # 工具函数
    │   ├── problems/                   # 优化问题模块
    │   │   ├── problem_generator.py    # 问题生成器
    │   │   ├── problem_sets.py         # 预定义问题集
    │   │   ├── problem_spec.py         # 问题规格定义
    │   │   └── datasets.py             # 数据集
    │   └── mnist/                      # MNIST 相关实现
    └── L2O-Scale-Evaluation/           # 评估代码目录
        ├── metatest.py                 # 评估主脚本
        ├── metaopt.py                  # 评估核心逻辑
        ├── optimizer/                  # 复制的优化器代码
        └── problems/                   # 复制的问题代码
```

---

## 2. 核心架构设计

### 2.1 系统架构

该系统由三个核心模块组成：

```
┌─────────────────────────────────────────────────────────────────────┐
│                  L2O-Scale 可学习优化器系统架构                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │ Problem Set │     │Meta-Optimizer│     │ Evaluation  │          │
│  │ （问题集）  │     │（元优化器） │     │（评估模块） │          │
│  ├─────────────┤     ├─────────────┤     ├─────────────┤          │
│  │- Quadratic  │     │HierarchicalRNN│    │- Loss Curve │          │
│  │- MLP (MNIST)│────>│             │────>│- Generalization│        │
│  │- ConvNet    │     │ Level 0: Per│     │- Convergence│          │
│  │- Rosenbrock │     │ Parameter   │     └─────────────┘          │
│  │- ...        │     │ Level 1: Per│                              │
│  └─────────────┘     │ Tensor      │                              │
│                      │ Level 2:    │                              │
│                      │ Global      │                              │
│                      └─────────────┘                              │
│                             │                                      │
│                             v                                      │
│  ┌──────────────────────────────────────────────────────┐         │
│  │             正则化模块 (Regularization)              │         │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │         │
│  │  │Hessian   │  │Hessian   │  │Jacobian  │           │         │
│  │  │Trace     │  │ESD       │  │Trace     │           │         │
│  │  └──────────┘  └──────────┘  └──────────┘           │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐         │
│  │             增强学习模块 (Enhancement)                │         │
│  │  ┌──────────┐  ┌──────────┐                          │         │
│  │  │Curriculum│  │Imitation │                          │         │
│  │  │Learning  │  │Learning  │                          │         │
│  │  └──────────┘  └──────────┘                          │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 HierarchicalRNN 优化器架构

HierarchicalRNN 采用三层分层结构，每层处理不同粒度的信息：
[HierarchicalRNN架构详细分析](/Model_Free_L2O/L2O-Entropy/HierarchicalRNN_详细分析.md)

```
┌───────────────────────────────────────────────────────────────────┐
│            HierarchicalRNN 三层分层架构                           │
│  代码位置: optimizer/hierarchical_rnn.py:61-806                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  输入: gradient g, 优化器状态 state                               │
│       │                                                           │
│       v                                                           │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ Level 0: Per-Parameter RNN (逐参数层)                    │     │
│  │   输入: [g_scaled_1~4, log_ms, rel_lr]                   │     │
│  │   BiasGRUCell (hidden_size=10)                           │     │
│  │   输出: h_param [num_params, 10]                         │     │
│  │         -> update_delta, scl_decay, inp_decay, lr_change │     │
│  └─────────────────────────────────────────────────────────┘     │
│       │ mean(h_param)                                             │
│       v                                                           │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ Level 1: Per-Tensor RNN (逐张量层)                       │     │
│  │   BiasGRUCell (hidden_size=20)                           │     │
│  │   输出: h_layer [1, 20] -> bias for Level 0              │     │
│  └─────────────────────────────────────────────────────────┘     │
│       │ mean(h_layer)                                             │
│       v                                                           │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ Level 2: Global RNN (全局层)                             │     │
│  │   BiasGRUCell (hidden_size=20)                           │     │
│  │   输出: h_global [1, 20] -> bias for Level 1             │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│  最终输出: update_step = lr * update_delta                        │
│           new_param = param - update_step                         │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心算法流程

### 3.1 完整训练流程图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         L2O-Entropy 元训练完整流程                        │
└──────────────────────────────────────────────────────────────────────────┘

  ╔═══════════════════════════════════════════════════════════════════╗
  ║                      【阶段 1】初始化阶段                          ║
  ╚═══════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
          ┌──────────────────────────────────────────┐
          │  1. 加载超参数配置                       │
          │     - meta_learning_rate = 1e-6         │
          │     - num_meta_iterations = 100         │
          │     - fix_unroll_length = 20            │
          │     - alpha = 1e-4 (Hessian 正则权重)   │
          │     - reg_option = 'hessian-esd'        │
          └──────────────────────────────────────────┘
                                  │
                                  ▼
          ┌──────────────────────────────────────────┐
          │  2. 构建优化问题并加载数据               │
          │     problem: MNIST MLP (784→20→10)       │
          │     mini_data: [55000, 784] float32      │
          │     mini_labels: [55000] int32           │
          │     batch_size: 128                      │
          └──────────────────────────────────────────┘
                                  │
                                  ▼
          ┌──────────────────────────────────────────┐
          │  3. 创建 HierarchicalRNN 优化器          │
          │     (3层分层结构: 10, 20, 20)            │
          │     - Level 0: Per-Parameter (10)       │
          │     - Level 1: Per-Tensor (20)          │
          │     - Level 2: Global (20)              │
          └──────────────────────────────────────────┘
                                  │
                                  ▼
          ┌──────────────────────────────────────────┐
          │  4. 初始化 RMSProp 元优化器              │
          │     学习率: 1e-6, 梯度裁剪: 1e4          │
          └──────────────────────────────────────────┘
                                  │
                                  ▼
                       ╔═════════════════════╗
                       ║  是否启用 MT?       ║
                       ║  (if_mt)            ║
                       ╚═════════════════════╝
                          /              \
                    Yes  /                \  No
                        /                  \
                       ▼                    ▼
          ┌─────────────────────┐     ┌──────────┐
          │  创建 MT Utils      │     │  跳过    │
          │  (Adam, mt_k=1)     │     │          │
          └─────────────────────┘     └──────────┘
                       │                    │
                       └────────┬───────────┘
                                │
                                ▼

  ╔═══════════════════════════════════════════════════════════════════╗
  ║                  【阶段 2】元训练主循环                            ║
  ║            for k in range(num_meta_iterations)                   ║
  ╚═══════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
          ┌──────────────────────────────────────────┐
          │  1. 初始化 optimizee 参数                │
          │     init_tensors = random_normal(...)    │
          │     (W1, b1, W2, b2) 随机初始化          │
          └──────────────────────────────────────────┘
                                  │
                                  ▼
                       ╔═════════════════════╗
                       ║ 是否启用课程学习?   ║
                       ║     (if_cl)         ║
                       ╚═════════════════════╝
                          /              \
                    Yes  /                \  No
                        /                  \
                       ▼                    ▼
   ┌────────────────────────────┐  ┌──────────────────────┐
   │ 根据 curriculum_idx        │  │ 使用固定 num_steps   │
   │ 确定 num_steps             │  │ (e.g., 100)          │
   │ [100, 200, 500, ...]       │  │                      │
   └────────────────────────────┘  └──────────────────────┘
                       │                    │
                       └────────┬───────────┘
                                │
                                ▼
          ┌──────────────────────────────────────────┐
          │  2. 计算 unroll 迭代次数                 │
          │     num_unrolls = num_steps / unroll_len │
          │     (例: 100/20 = 5 次 unroll)           │
          └──────────────────────────────────────────┘
                                  │
                                  ▼
          ┌──────────────────────────────────────────┐
          │  3. 预生成所有 batch 索引                │
          │     batches = dataset.batch_indices(     │
          │       num_batches=num_unrolls×20,        │
          │       batch_size=128)                    │
          │     输出: [[idx1~128], [idx129~256],...] │
          └──────────────────────────────────────────┘
                                  │
                                  ▼
                    ╔════════════════════════════╗
                    ║ 是否 MT 概率触发?          ║
                    ║ (random() < mt_ratio)      ║
                    ╚════════════════════════════╝
                          /              \
                    Yes  /                \  No
                        /                  \
                       ▼                    ▼
          ┌─────────────────────┐     ┌──────────┐
          │ 生成 MT 标签        │     │  跳过    │
          │ (使用 Adam 运行)    │     │          │
          └─────────────────────┘     └──────────┘
                       │                    │
                       └────────┬───────────┘
                                │
                                ▼

  ╔═══════════════════════════════════════════════════════════════════╗
  ║                   【阶段 3】Unroll 循环                            ║
  ║           for unroll_itr in range(num_unrolls)                   ║
  ╚═══════════════════════════════════════════════════════════════════╝
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                  │
         ▼                                                  │
  ┌─────────────────────────────────────────────────┐     │
  │  for itr in range(unroll_length):  # 20 步      │     │
  │      │                                           │     │
  │      ▼                                           │     │
  │  ┌────────────────────────────────────────┐     │     │
  │  │ 1. 获取当前 batch 数据                 │     │     │
  │  │    batch_inds = batches[               │     │     │
  │  │      unroll_itr*20 + itr]  # [128]索引 │     │     │
  │  │    batch_data = mini_data[batch_inds]  │     │     │
  │  │                           [128, 784]    │     │     │
  │  │    batch_labels = mini_labels[...]     │     │     │
  │  │                           [128]         │     │     │
  │  └────────────────────────────────────────┘     │     │
  │      │                                           │     │
  │      ▼                                           │     │
  │  ╔════════════════════════════════════════╗     │     │
  │  ║  【关键】双目标函数计算                ║     │     │
  │  ║  (不同数据范围 - 核心设计)             ║     │     │
  │  ╚════════════════════════════════════════╝     │     │
  │      │                                           │     │
  │      ├─────────────────┬─────────────────┐      │     │
  │      ▼                 ▼                 ▼      │     │
  │  ┌─────────────┐ ┌─────────────┐ ┌──────────┐ │     │
  │  │ 2a. 评估目标│ │ 2b. 训练目标│ │ 2c. 计算 │ │     │
  │  │             │ │             │ │  梯度和  │ │     │
  │  │ obj =       │ │ current_obj │ │  正则化  │ │     │
  │  │ problem.    │ │   = problem.│ │          │ │     │
  │  │ objective(  │ │   objective(│ │ grads,   │ │     │
  │  │   params,   │ │     params, │ │ reg =    │ │     │
  │  │   mini_data,│ │     batch_  │ │ problem. │ │     │
  │  │   mini_     │ │     data,   │ │ gradients│ │     │
  │  │   labels)   │ │     batch_  │ │ (current_│ │     │
  │  │             │ │     labels) │ │  obj,...)│ │     │
  │  │ [55000样本] │ │ [128 样本]  │ │ [128样本]│ │     │
  │  │             │ │             │ │          │ │     │
  │  │ 用于元损失  │ │ 用于梯度    │ │ 梯度&Hess│ │     │
  │  └─────────────┘ └─────────────┘ └──────────┘ │     │
  │      │                 │              │        │     │
  │      │                 └──────┬───────┘        │     │
  │      │                        │                │     │
  │      │                        ▼                │     │
  │      │      ╔══════════════════════════╗       │     │
  │      │      ║ 是否启用正则化?          ║       │     │
  │      │      ║ (reg_optimizer=True)     ║       │     │
  │      │      ╚══════════════════════════╝       │     │
  │      │          /              \               │     │
  │      │    Yes  /                \  No          │     │
  │      │        /                  \             │     │
  │      │       ▼                    ▼            │     │
  │      │  ┌────────────────┐  ┌──────────┐      │     │
  │      │  │ 累积 Hessian   │  │  跳过    │      │     │
  │      │  │ 正则化项:      │  │          │      │     │
  │      │  │ regular +=     │  │          │      │     │
  │      │  │   alpha * reg  │  │          │      │     │
  │      │  │ ⚠️ reg 在      │  │          │      │     │
  │      │  │ batch 上计算!  │  │          │      │     │
  │      │  │ [128 样本]     │  │          │      │     │
  │      │  └────────────────┘  └──────────┘      │     │
  │      │       │                    │            │     │
  │      │       └────────┬───────────┘            │     │
  │      │                │                        │     │
  │      │                ▼                        │     │
  │      │  ┌────────────────────────────────────┐│     │
  │      │  │ 3. 调用 HierarchicalRNN 计算更新   ││     │
  │      │  │    updates = optimizer._compute_   ││     │
  │      │  │              updates(params, grads, ││     │
  │      │  │              states, global_state)  ││     │
  │      │  │    返回: new_params, new_states,    ││     │
  │      │  │          update_steps               ││     │
  │      │  └────────────────────────────────────┘│     │
  │      │                │                        │     │
  │      ▼                │                        │     │
  │  ┌─────────────────┐ │                        │     │
  │  │ 4. 累加元目标   │ │                        │     │
  │  │    (完整数据)   │ │                        │     │
  │  │                 │ │                        │     │
  │  │ obj_accum +=    │ │                        │     │
  │  │   obj_weights   │ │                        │     │
  │  │   [itr] * obj   │ │                        │     │
  │  │                 │ │                        │     │
  │  │ ⚠️ obj 在       │ │                        │     │
  │  │ 55000 样本上    │ │                        │     │
  │  │ 计算!           │ │                        │     │
  │  └─────────────────┘ │                        │     │
  │      │                │                        │     │
  │      ▼                ▼                        │     │
  │      ╔══════════════════════════╗              │     │
  │      ║ MT 模式?                 ║              │     │
  │      ║ (if mode_mt)             ║              │     │
  │      ╚══════════════════════════╝              │     │
  │          /              \                      │     │
  │    Yes  /                \  No                 │     │
  │        /                  \                    │     │
  │       ▼                    ▼                   │     │
  │  ┌────────────┐      ┌──────────┐             │     │
  │  │ 累加 MT    │      │  跳过    │             │     │
  │  │ loss:      │      │          │             │     │
  │  │ MSE(update,│      │          │             │     │
  │  │     mt_upd)│      │          │             │     │
  │  └────────────┘      └──────────┘             │     │
  │       │                    │                   │     │
  │       └────────┬───────────┘                   │     │
  │                │                               │     │
  │                ▼                               │     │
  │  ┌────────────────────────────────────────┐   │     │
  │  │ 5. 更新参数和状态                      │   │     │
  │  │    params = new_params                 │   │     │
  │  │    states = new_states                 │   │     │
  │  │    (继续下一次 BPTT 步骤)              │   │     │
  │  └────────────────────────────────────────┘   │     │
  │                │                               │     │
  │                └─────────────► 循环继续        │     │
  └─────────────────────────────────────────────────┘     │
                                  │                        │
                                  ▼                        │
          ┌──────────────────────────────────────────┐    │
          │  【关键】计算元训练损失 (混合数据策略)   │    │
          │                                          │    │
          │  scaled_meta_obj =                       │    │
          │    scale_objective(obj_accum, ...)       │    │
          │    ↑                                     │    │
          │    └─ 基于完整数据 (55000 样本)          │    │
          │                                          │    │
          │  final_meta_loss =                       │    │
          │    scaled_meta_obj + regular             │    │
          │                      ↑                   │    │
          │                      └─ 基于 batch (128) │    │
          │                                          │    │
          │  ⚠️ 核心设计: 元损失由两部分组成         │    │
          │  - 主目标 obj_accum: 55000 样本          │    │
          │  - 正则项 regular: 128 样本/batch        │    │
          └──────────────────────────────────────────┘    │
                                  │                        │
                                  └────────────────────────┘
                                  │
                                  ▼

  ╔═══════════════════════════════════════════════════════════════════╗
  ║                   【阶段 4】元参数更新                             ║
  ╚═══════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
          ┌──────────────────────────────────────────┐
          │  1. 计算总体元训练目标                   │
          │     final_loss = final_meta_loss + jacob │
          │     (jacob 为可选的 Jacobian 正则项)     │
          └──────────────────────────────────────────┘
                                  │
                                  ▼
          ┌──────────────────────────────────────────┐
          │  2. 梯度裁剪 (clip to 1e4)               │
          │     防止梯度爆炸                         │
          └──────────────────────────────────────────┘
                                  │
                                  ▼
          ┌──────────────────────────────────────────┐
          │  3. RMSProp 更新优化器参数               │
          │     meta_lr = 1e-6                       │
          │     更新 HierarchicalRNN 的所有参数:     │
          │     - BiasGRU 权重                       │
          │     - 投影矩阵 (W_update, W_lr, ...)    │
          │     - 层间偏置矩阵 (W_{1→0}, W_{2→1})   │
          └──────────────────────────────────────────┘
                                  │
                                  ▼

  ╔═══════════════════════════════════════════════════════════════════╗
  ║                    【阶段 5】评估阶段                              ║
  ╚═══════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
                    ╔════════════════════════════╗
                    ║ 是否周期性评估?            ║
                    ║ (k+1) % eval_period == 0   ║
                    ╚════════════════════════════╝
                          /              \
                    Yes  /                \  No
                        /                  \
                       ▼                    ▼
          ┌─────────────────────┐     ┌──────────────────┐
          │ 运行 validation     │     │ 继续下一次       │
          │ cost = validate()   │     │ meta iteration   │
          │ (在 validation_set  │     │                  │
          │  上评估性能)        │     │                  │
          └─────────────────────┘     └──────────────────┘
                       │                    │
                       ▼                    │
            ╔═══════════════════╗           │
            ║ 性能改进?         ║           │
            ║ cost < best?      ║           │
            ╚═══════════════════╝           │
                  /        \                │
            Yes  /          \  No           │
                /            \              │
               ▼              ▼             │
   ┌─────────────────┐  ╔═════════════════════════╗
   │ 保存 checkpoint │  ║ 是否 CL 满足切换条件?   ║
   │ model.ckpt-{idx}│  ║ (num_eval >= min &&     ║
   └─────────────────┘  ║  improved)              ║
               │         ╚═════════════════════════╝
               │              /            \
               │        Yes  /              \  No
               │            /                \
               │           ▼                  │
               │  ┌──────────────────┐       │
               │  │ curriculum_idx++ │       │
               │  │ 恢复最佳模型     │       │
               │  │ 切换到下一阶段   │       │
               │  │ (更长的训练步数) │       │
               │  └──────────────────┘       │
               │           │                  │
               └───────────┼──────────────────┼───────────┐
                           │                  │           │
                           └──────────────────┘           │
                                  │                       │
                                  └───────────────────────┘
                                  │
                                  ▼
                    ╔════════════════════════════╗
                    ║ 是否完成所有元训练迭代?    ║
                    ║ k >= num_meta_iterations   ║
                    ╚════════════════════════════╝
                          /              \
                    No   /                \  Yes
                        /                  \
                       ▼                    ▼
              【回到阶段 2】         ┌──────────────┐
              继续元训练循环         │  训练完成    │
                                    │  保存最终模型│
                                    └──────────────┘
```

### 3.2 元训练主循环 (metaopt.py:117-738)

**配置参数**:
- `meta_learning_rate = 1e-6` - 元学习率
- `gradient_clip = 1e4` - 梯度裁剪
- `num_meta_iterations = 100` - 元训练迭代次数
- `fix_unroll_length = 20` - Unroll 长度
- `alpha = 1e-4` - 正则化权重
- `reg_option = 'hessian-esd'` - 正则化选项

**主循环步骤**:
1. **初始化参数**: `init_tensors = [random_normal(shape) for shape in param_shapes]`
2. **计算 unroll 次数**: 根据课程学习阶段或固定值
3. **生成 MT 标签** (如果启用): `mt_labels = mt_utils.get_mt_labels(...)`
4. **Unroll 循环**: 执行多次参数更新
5. **周期性评估**: 保存最佳模型
6. **课程学习切换**: 逐步增加训练难度

### 3.3 单次 Unroll (trainable_optimizer.py:200-470)

```
for itr in range(unroll_length):
    # Step 1: 获取 batch 数据
    batch_data = mini_data[batch_indices[itr]]
    batch_labels = mini_labels[batch_indices[itr]]

    # Step 2: 计算目标函数
    obj = problem.objective(params, mini_data, mini_labels)

    # Step 3: 计算梯度和正则化项
    current_obj = problem.objective(params, batch_data, batch_labels)
    grads, reg = problem.gradients(current_obj, params)

    # Step 4: 累加目标
    obj_accum += obj_weights[itr] * obj

    # Step 5: 调用 HierarchicalRNN 计算更新
    updates = optimizer._compute_updates(params, grads, states, ...)

    # Step 6: MT Loss 计算 (如果启用)
    if mode_mt:
        mse_loss = MSE(update_steps, update_steps_mt)
        obj_accum_mt += obj_weights[itr] * mse_loss

    # Step 7: 更新参数和状态
    params = new_params
    states = new_states
```

---

## 4. 正则化模块详解

### 4.1 正则化类型

| 正则化类型 | 配置代码 | 数学公式 | 计算方法 | 代码位置 |
|-----------|---------|---------|---------|---------|
| Hessian 迹 | `hessian` | Tr(H) = E[v^T H v] | Hutchinson 估计 | problem_generator.py:181-209 |
| Hessian ESD | `hessian-esd` | Σᵢ λᵢ | Lanczos 算法 | problem_generator.py:248-327 |
| Hessian 特征值 | `hessian-ev` | Σᵢ λᵢ (top_n) | Power Iteration | problem_generator.py:211-246 |
| Jacobian 迹 | `jacob` | Σᵢ ‖gᵢ‖² / n | 直接计算 | problem_generator.py:128-140 |

### 4.2 Hutchinson 估计 Hessian 迹

**数学原理**:
```
Tr(H) = E_v[v^T H v]  其中 v ~ Rademacher 分布 (+1/-1 均匀)
估计: Tr(H) ≈ (1/N) * Σᵢ (vᵢ^T H vᵢ)
```

**算法流程** (problem_generator.py:181-209):
```python
for i in range(max_iter):  # 默认 max_iter=10
    # Step 1: 生成 Rademacher 随机向量
    v = random(0, 2, shape) -> {-1, +1}

    # Step 2: 计算 Hessian-向量积 Hv
    Hv = tf.gradients(grads, params, grad_ys=v)

    # Step 3: 计算内积 v^T H v
    products = sum(Hv_i * v_i for i in params)
    trace_vhv.append(products)

return mean(trace_vhv)
```

### 4.3 Lanczos 算法估计特征值分布

**算法流程** (problem_generator.py:248-327):
```python
for k in range(n_v):  # SLQ 采样迭代
    # Step 1: 初始化
    v = normalize(Rademacher_vector())
    alpha_list, beta_list = [], []

    # Step 2: Lanczos 迭代
    for i in range(max_iter):
        w' = H @ v  # Hessian-向量积
        alpha = <w', v>  # 对角元素
        beta = ||w||  # 次对角元素
        v = orthonormalize(w, v_list)
        ...

    # Step 3: 构建三对角矩阵 T
    T = tridiagonal(alpha_list, beta_list)

    # Step 4: 计算 T 的特征值
    eigenvalues = tf.self_adjoint_eig(T)
    eigen_list_full.append(eigenvalues)

return sum(eigen_list_full)
```

---

## 5. 增强学习模块

### 5.1 课程学习 (Curriculum Learning)

**代码位置**: metaopt.py:170-178, 617-688

**核心思想**: 逐渐增加优化问题的训练步数

**阶段切换配置**:
```python
num_steps = [100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

# 对应 unroll 迭代次数 (fix_unroll_length=20)
num_unrolls_train = [5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
```

**切换条件**:
```python
curriculum_idx = 0      # 当前阶段索引
num_eval = 0            # 当前阶段评估次数
improved = False        # 是否取得改进
min_num_eval = 3        # 最小评估次数门槛

if num_eval >= min_num_eval and improved:
    restore_model(model.ckpt-{curriculum_idx})  # 恢复最佳模型
    curriculum_idx += 1  # 切换到下一阶段
elif num_eval >= min_num_eval and not improved:
    break  # 终止训练
```

### 5.2 模仿学习 (Imitation Learning / MT)

**代码位置**: mt_utils.py:16-86, metaopt.py:355-361, 409-449

**核心思想**: 让可学习优化器模仿成熟优化器 (如 Adam) 的更新方向

**MT 标签生成** (mt_utils.py:49-86):
```python
def get_mt_labels(init_tensors, data, labels, ...):
    opt = Adam(lr=0.01)
    mt_labels = []

    for unroll_itr in range(num_unrolls):
        for itr in range(unroll_len):
            x_prev = get_params()
            for ki in range(k):  # k 步 Adam 更新
                opt.step(batch_data, batch_labels)
            x_cur = get_params()
            x_update = x_cur - x_prev  # 参数更新量
            mt_labels_roll.append(x_update)
        mt_labels.append(mt_labels_roll)

    return mt_labels
```

**MT Loss 计算**:
```python
mse_loss = sum([
    reduce_sum(0.5 * (d1 - d2)^2)
    for d1, d2 in zip(update_steps, update_steps_mt)
]) / num_params
```

---

## 6. 数据流与 MNIST 详细调用

### 6.1 MNIST 数据加载与处理流程

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    MNIST 数据完整调用流程图                               │
│                代码位置: datasets.py, problem_sets.py                     │
└──────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════╗
║                     【阶段 1】数据加载阶段                              ║
║                    datasets.py:221-230                                ║
╚═══════════════════════════════════════════════════════════════════════╝
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  1. 调用 tensorflow.contrib.learn         │
         │     .datasets.mnist.load_mnist()         │
         │                                          │
         │  自动下载文件 (如果不存在):               │
         │  - train-images-idx3-ubyte.gz  (60000张) │
         │  - train-labels-idx1-ubyte.gz  (60000个) │
         │  - t10k-images-idx3-ubyte.gz   (10000张) │
         │  - t10k-labels-idx1-ubyte.gz   (10000个) │
         └──────────────────────────────────────────┘
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  2. 数据预处理                            │
         │     - 图像: [N, 784] -> [N, 28, 28, 1]   │
         │     - 归一化: [0, 255] -> [0.0, 1.0]     │
         │     - dtype: float32                      │
         │                                          │
         │  训练集: 55,000 样本                      │
         │  验证集: 5,000 样本 (从训练集分离)        │
         │  测试集: 10,000 样本                      │
         └──────────────────────────────────────────┘
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  3. 封装为 Dataset namedtuple             │
         │     Dataset(data, labels)                │
         │     data:   [55000, 28, 28, 1] float32   │
         │     labels: [55000] int32                │
         └──────────────────────────────────────────┘
                                 │
                                 ▼

╔═══════════════════════════════════════════════════════════════════════╗
║                    【阶段 2】问题定义阶段                               ║
║                   problem_sets.py:66-72                               ║
╚═══════════════════════════════════════════════════════════════════════╝
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  mnist_mlp_problems() 定义:              │
         │                                          │
         │  return [                                │
         │    (                                     │
         │      _Spec(pg.FullyConnected,            │
         │            (784, 10),                    │
         │            {"hidden_sizes": (20,),       │
         │             "activation": tf.nn.sigmoid}),│
         │      datasets.mnist(train=True),         │  ◄─ 数据集
         │      128                                 │  ◄─ batch_size
         │    )                                     │
         │  ]                                       │
         └──────────────────────────────────────────┘
                                 │
                                 ▼

╔═══════════════════════════════════════════════════════════════════════╗
║                   【阶段 3】Batch 索引生成                              ║
║                   datasets.py:53-92, 284-322                          ║
╚═══════════════════════════════════════════════════════════════════════╝
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  batch_indices(data_size, num_batches,   │
         │                batch_size)               │
         │                                          │
         │  参数:                                   │
         │  - data_size = 55000 (训练集大小)        │
         │  - num_batches = num_unrolls × unroll_len│
         │                = 5 × 20 = 100 (示例)     │
         │  - batch_size = 128                      │
         └──────────────────────────────────────────┘
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  生成流程:                                │
         │                                          │
         │  1. 初始化数据集索引:                     │
         │     indices = [0, 1, 2, ..., 54999]      │
         │     shuffle(indices)                     │
         │                                          │
         │  2. 循环生成 num_batches 个 batch:       │
         │     for b in range(num_batches):         │
         │       if 索引用完:                       │
         │         reshuffle(indices)  # 新 epoch   │
         │         重置 start = 0                   │
         │       batch[b] = indices[start:start+128]│
         │       start += 128                       │
         │                                          │
         │  输出: [[idx1~128], [idx129~256], ...]   │
         │        shape: [num_batches, batch_size]  │
         └──────────────────────────────────────────┘
                                 │
                                 ▼

╔═══════════════════════════════════════════════════════════════════════╗
║                【阶段 4】训练循环中的数据使用                            ║
║               trainable_optimizer.py:263-401                          ║
╚═══════════════════════════════════════════════════════════════════════╝
                                 │
    ┌────────────────────────────┴────────────────────────────┐
    │                                                          │
    ▼                                                          │
  ┌─────────────────────────────────────────────────────────┐ │
  │  for itr in range(unroll_length):  # 每次 unroll        │ │
  │                                                          │ │
  │  ┌───────────────────────────────────────────────────┐  │ │
  │  │ 步骤 1: 获取当前迭代的 batch 索引                  │  │ │
  │  │                                                    │  │ │
  │  │ batch_indices = tf.gather(batches, itr)           │  │ │
  │  │ # 从预生成的 batch 索引数组中取第 itr 个          │  │ │
  │  │ # shape: [128]                                    │  │ │
  │  └───────────────────────────────────────────────────┘  │ │
  │                       │                                  │ │
  │                       ▼                                  │ │
  │  ┌───────────────────────────────────────────────────┐  │ │
  │  │ 步骤 2: 提取当前 batch 数据                        │  │ │
  │  │                                                    │  │ │
  │  │ batch_data = tf.gather(mini_data, batch_indices)  │  │ │
  │  │ # shape: [128, 784]                               │  │ │
  │  │                                                    │  │ │
  │  │ batch_labels = tf.gather(mini_labels, batch_indices)│ │ │
  │  │ # shape: [128]                                    │  │ │
  │  └───────────────────────────────────────────────────┘  │ │
  │                       │                                  │ │
  │                       ▼                                  │ │
  │  ╔═══════════════════════════════════════════════════╗  │ │
  │  ║      【关键】两种目标函数计算 (不同数据范围)       ║  │ │
  │  ╚═══════════════════════════════════════════════════╝  │ │
  │                       │                                  │ │
  │          ┌────────────┴────────────┐                    │ │
  │          │                         │                    │ │
  │          ▼                         ▼                    │ │
  │  ┌─────────────────┐       ┌─────────────────┐         │ │
  │  │ 评估目标 (obj)  │       │ 训练目标        │         │ │
  │  │                 │       │ (current_obj)   │         │ │
  │  │ 在完整数据上    │       │ 在当前 batch    │         │ │
  │  │ 计算:           │       │ 上计算:         │         │ │
  │  │                 │       │                 │         │ │
  │  │ obj = problem.  │       │ current_obj =   │         │ │
  │  │   objective(    │       │   problem.      │         │ │
  │  │     params,     │       │   objective(    │         │ │
  │  │     mini_data,  │       │     params,     │         │ │
  │  │     mini_labels)│       │     batch_data, │         │ │
  │  │                 │       │     batch_labels)│        │ │
  │  │ [55000 样本]    │       │ [128 样本]      │         │ │
  │  └─────────────────┘       └─────────────────┘         │ │
  │          │                         │                    │ │
  │          │                         ▼                    │ │
  │          │                 ┌─────────────────┐         │ │
  │          │                 │ 计算梯度和正则化 │         │ │
  │          │                 │                 │         │ │
  │          │                 │ grads, reg =    │         │ │
  │          │                 │   problem.      │         │ │
  │          │                 │   gradients(    │         │ │
  │          │                 │     current_obj,│         │ │
  │          │                 │     params)     │         │ │
  │          │                 │                 │         │ │
  │          │                 │ 用于参数更新    │         │ │
  │          │                 │ 和 Hessian 正则 │         │ │
  │          │                 └─────────────────┘         │ │
  │          │                         │                    │ │
  │          ▼                         │                    │ │
  │  ┌─────────────────┐               │                    │ │
  │  │ 累加元目标      │               │                    │ │
  │  │                 │               │                    │ │
  │  │ obj_accum +=    │               │                    │ │
  │  │   obj_weights   │               │                    │ │
  │  │   [itr] * obj   │               │                    │ │
  │  │                 │               │                    │ │
  │  │ 用于元训练      │               │                    │ │
  │  │ 的损失函数      │               │                    │ │
  │  └─────────────────┘               │                    │ │
  │          │                         │                    │ │
  │          └────────────┬────────────┘                    │ │
  │                       │                                  │ │
  │                       ▼                                  │ │
  │  ┌───────────────────────────────────────────────────┐  │ │
  │  │ 步骤 3: 调用优化器计算参数更新                     │  │ │
  │  │                                                    │  │ │
  │  │ updates = self._compute_updates(                   │  │ │
  │  │     params, grads, states, global_state, ...)     │  │ │
  │  │                                                    │  │ │
  │  │ new_params, new_states, update_steps = updates    │  │ │
  │  └───────────────────────────────────────────────────┘  │ │
  │                                                          │ │
  └──────────────────────────────────────────────────────────┘ │
                                 │                              │
                                 └──────────────────────────────┘
```

### 6.2 数据使用的关键设计

#### 6.2.1 双目标函数机制 ⚠️ 重要设计

**核心问题**：为什么在同一次迭代中计算两个目标函数？

```python
# 代码位置: trainable_optimizer.py:308, 315-316, 349-352, 367-369

# 1. 评估目标 - 使用完整数据集
obj = problem.objective(params, mini_data, mini_labels)
# mini_data shape: [55000, 784]
# 计算在 55000 个样本上的平均损失

# 2. 训练目标 - 使用当前 batch
current_obj = problem.objective(params, batch_data, batch_labels)
# batch_data shape: [128, 784]
# 计算在 128 个样本上的平均损失

# 3. 计算梯度和 Hessian 正则化 - 都基于 batch
grads, reg = problem.gradients(current_obj, params)
# ⚠️ 关键: reg (Hessian 正则化) 是在 batch_data 上计算的！

# 4. 累积正则化项
if FLAGS.reg_optimizer:
    regular += alpha * reg  # alpha = 1e-4

# 5. 元损失累加 - 使用 obj (完整数据集的损失!)
obj_accum += obj_weights[itr] * obj  # ← obj 来自完整数据集

# 最终元损失 = scale_objective(obj_accum, ...) + regular
#            = (完整数据的损失累加)    + (batch上的Hessian正则化)
```

**设计原理表**：

| 目标函数/正则项 | 数据范围 | 计算频率 | 用途 | 设计原因 |
|---------------|---------|---------|------|----------|
| **`obj`** | **完整 mini_data**<br>(55000 样本) | **每次迭代** | **元训练损失**<br>`obj_accum` | ✓ 准确评估优化器性能<br>✓ 避免 mini-batch 噪声<br>✓ 提供稳定的元梯度信号 |
| `current_obj` | 当前 batch<br>(128 样本) | 每次迭代 | **梯度计算** | ✓ 减少计算开销<br>✓ 引入随机性<br>✓ 模拟实际 SGD 场景 |
| **`reg`**<br>(Hessian 正则化) | **当前 batch**<br>(128 样本) | 每次迭代 | **元训练正则项**<br>`regular` | ⚠️ Hessian 计算极其昂贵<br>✓ 在 batch 上计算可行<br>✓ 仍能约束优化器复杂度 |

**为什么元损失 `obj` 用完整数据集，而正则项 `reg` 用 batch？**

1. **元学习目标 (obj)**：训练优化器在**整个数据集**上表现良好
   - 如果用 batch 计算元损失 → 优化器可能过拟合到特定 batch 模式
   - 用完整数据集 → 优化器学习泛化的优化策略

2. **稳定的元梯度 (obj)**：
   ```
   元梯度 = ∂(obj_accum + regular)/∂optimizer_params
   obj_accum 来自完整数据 → 元梯度主信号稳定，训练收敛快
   ```

3. **Hessian 正则化的计算成本 (reg)**：
   - **为什么不能在完整数据集上计算？**
     ```
     Hessian-vector 积计算 = 二阶导数 = 两次反向传播
     batch (128) 上计算 Hessian: 可行
     完整数据 (55000) 上计算 Hessian: 计算成本 ≈ 430× → 不可行！
     ```
   - **在 batch 上计算够用吗？**
     - ✓ Hessian 正则化的目的：约束优化器不要过于复杂
     - ✓ batch 上的 Hessian 信息已足够提供曲率约束
     - ✓ 正则项只是辅助项，不是主要优化目标

4. **计算成本总结**：
   ```
   每次迭代:
   obj (完整数据前向传播):        成本 = 430× batch ≈ 可接受
   reg (batch Hessian计算):        成本 = 2× batch 反向传播 ≈ 可接受
   假设的 reg (完整数据 Hessian):  成本 = 2×430× batch ≈ 不可接受！

   元训练总成本 (100次×5 unrolls×20步):
   额外 obj 计算: 100,000 次前向传播
   Hessian 计算: 20,000 次二阶导数 (在 batch 上)
   ```

#### 6.2.2 数据采样策略

- **Mini-batch SGD**: 每次迭代使用不同的 128 个样本计算梯度
- **Epoch 循环**: 当数据用完后自动 reshuffle 开始新 epoch
- **随机性**: 打乱顺序确保梯度估计的无偏性

**完整数据流示意**：
```
第 itr 次迭代:

  ┌─────────────────────────────────────────────────────────────────┐
  │  batch_data [128 样本]                                          │
  │       │                                                         │
  │       ▼                                                         │
  │  current_obj ─────┬─→ grads ──→ HierarchicalRNN ──→ param_update│
  │                   │                                             │
  │                   └─→ reg (Hessian 正则化)                      │
  │                        ↓                                        │
  │                   regular += alpha * reg  ─────────┐           │
  └─────────────────────────────────────────────────────┼───────────┘
                                                        │
  ┌─────────────────────────────────────────────────────┼───────────┐
  │  mini_data [55000 样本]                              │           │
  │       │                                              │           │
  │       ▼                                              │           │
  │  obj ──→ obj_accum += obj_weights[itr] * obj ───────┼─────┐     │
  └─────────────────────────────────────────────────────┘     │     │
                                                              │     │
  ┌───────────────────────────────────────────────────────────┼─────┼───┐
  │  元训练最终损失                                            ▼     ▼   │
  │                                                                      │
  │  final_loss = scale_objective(obj_accum, ...) + regular             │
  │               └──────────┬──────────┘           └────┬────┘         │
  │                      完整数据 (55000)           batch (128)          │
  │                          ↓                                           │
  │              反向传播 ──→ 更新优化器参数                              │
  └──────────────────────────────────────────────────────────────────────┘
```

### 6.3 MNIST MLP 示例

- **输入**: 784 像素
- **网络**: 784 -> 20 -> 10 (MLP with sigmoid)
- **参数**: W1[784,20], b1[20], W2[20,10], b2[10]
- **总参数量**: 784×20 + 20 + 20×10 + 10 = 15,910

### 6.4 完整数据流

```
[输入层]
mini_data:    [55000, 784]    # 完整训练集 (已 flatten)
mini_labels:  [55000]         # 类别标签 (0-9)

[参数层]
params = [W1, b1, W2, b2]
       = [[784,20], [20], [20,10], [10]]

[梯度层]
grads = tf.gradients(loss, params)
      = [[784,20], [20], [20,10], [10]]  # 与 params 同形

[优化器状态] 对每个参数 p with shape [d1, d2, ...]:
state = {
    "parameter": [d1*d2*..., 10],    # Level 0 RNN 状态
    "layer": [1, 20],                 # Level 1 RNN 状态
    "scl_decay": [d1*d2*..., 1],     # scale decay
    "inp_decay": [d1*d2*..., 1],     # input decay
    "log_learning_rate": [d1*d2*..., 1],  # 学习率
    "grad_accum1~4": [d1*d2*..., 1],      # 梯度累积
    "ms1~4": [d1*d2*..., 1],              # mean squared
}

global_state = [1, 20]  # Level 2 RNN 状态

[RNN 输入] 对每个参数 (Flatten后元素个数为 n):
rnn_input = concat([
    grads_scaled_1~4,   # [n, 4]  - 多尺度衰减梯度
    grad_products,      # [n, 3]  - 梯度交叉项
    log_ms,             # [n, 4]  - log mean squared
    relative_lr,        # [n, 1]  - 相对学习率
], axis=1)              # 总计 [n, 12]

[输出层]
update_step = lr_param * update_delta  # 与参数同形
new_param = param - update_step
```

---

## 7. 元测试流程

### 7.1 测试流程图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    元优化器测试完整流程图                                 │
│                  代码位置: metaopt.py:783-932                             │
└──────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════╗
║                    【阶段 1】测试初始化                                 ║
╚═══════════════════════════════════════════════════════════════════════╝
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  1. 加载测试配置                          │
         │     - num_testing_itrs = 10000           │
         │     - test_problem (e.g., MNIST ReLU)    │
         │     - batch_size = 128                   │
         │     - random_seed (可选)                 │
         └──────────────────────────────────────────┘
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  2. 构建测试问题                          │
         │     problem = problem_spec.build()       │
         │     dataset = datasets.mnist(train=False)│
         │     (使用测试集: 10000 样本)              │
         └──────────────────────────────────────────┘
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  3. 初始化 optimizee 参数                 │
         │     if seed is list:                     │
         │       params = init_fixed_variables(seed)│
         │     else:                                │
         │       params = problem.init_variables()  │
         └──────────────────────────────────────────┘
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  4. 构建计算图                            │
         │     obj = problem.objective(params, ...) │
         │     grads = problem.gradients(obj, ...)  │
         └──────────────────────────────────────────┘
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  5. 创建优化器更新操作                     │
         │     train_op = optimizer.apply_gradients(│
         │                  zip(grads, params))     │
         └──────────────────────────────────────────┘
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  6. 加载预训练的优化器参数                 │
         │     restorer.restore(sess, ckpt_path)    │
         │     (从 train_dir/model.ckpt-X 加载)     │
         └──────────────────────────────────────────┘
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  7. 生成 batch 索引                       │
         │     batch_inds = dataset.batch_indices(  │
         │       num_testing_itrs, batch_size)      │
         │     # [[idx1~128], [idx129~256], ...]    │
         └──────────────────────────────────────────┘
                                 │
                                 ▼

╔═══════════════════════════════════════════════════════════════════════╗
║                    【阶段 2】测试主循环                                 ║
║              for itr in range(num_testing_itrs)                       ║
╚═══════════════════════════════════════════════════════════════════════╝
                                 │
    ┌────────────────────────────┴────────────────────────────┐
    │                                                          │
    ▼                                                          │
  ┌─────────────────────────────────────────────────────────┐ │
  │  单次测试迭代循环                                        │ │
  │                                                          │ │
  │  ┌───────────────────────────────────────────────────┐  │ │
  │  │ 步骤 1: 获取当前 batch 数据                        │  │ │
  │  │                                                    │  │ │
  │  │ batch = batch_inds[itr]  # [128] 索引            │  │ │
  │  │ batch_data = dataset.data[batch]                  │  │ │
  │  │ batch_labels = dataset.labels[batch]              │  │ │
  │  └───────────────────────────────────────────────────┘  │ │
  │                       │                                  │ │
  │                       ▼                                  │ │
  │  ┌───────────────────────────────────────────────────┐  │ │
  │  │ 步骤 2: 构建 feed_dict                             │  │ │
  │  │                                                    │  │ │
  │  │ feed = {                                          │  │ │
  │  │   data_placeholder: batch_data,    # [128, 784]  │  │ │
  │  │   labels_placeholder: batch_labels # [128]       │  │ │
  │  │ }                                                 │  │ │
  │  └───────────────────────────────────────────────────┘  │ │
  │                       │                                  │ │
  │                       ▼                                  │ │
  │  ╔═══════════════════════════════════════════════════╗  │ │
  │  ║      【关键】测试时的目标函数计算                  ║  │ │
  │  ╚═══════════════════════════════════════════════════╝  │ │
  │                       │                                  │ │
  │          ┌────────────┴────────────┐                    │ │
  │          │                         │                    │ │
  │          ▼                         ▼                    │ │
  │  ┌─────────────────┐       ┌─────────────────┐         │ │
  │  │ 计算梯度        │       │ 记录完整目标    │         │ │
  │  │ (用于更新参数)  │       │ (可选,每N步)    │         │ │
  │  │                 │       │                 │         │ │
  │  │ grads 在        │       │ full_obj 在     │         │ │
  │  │ batch_data 上   │       │ 完整测试集上    │         │ │
  │  │ [128 样本]      │       │ [10000 样本]    │         │ │
  │  └─────────────────┘       └─────────────────┘         │ │
  │          │                         │                    │ │
  │          │                         │                    │ │
  │          ▼                         ▼                    │ │
  │  ┌─────────────────┐       ┌─────────────────┐         │ │
  │  │ 应用优化器更新  │       │ records.append( │         │ │
  │  │                 │       │   objective,     │         │ │
  │  │ train_op:       │       │   grad_norm,     │         │ │
  │  │   新参数 =      │       │   param_norm,    │         │ │
  │  │   optimizer     │       │   ...)           │         │ │
  │  │   .step(...)    │       │                 │         │ │
  │  └─────────────────┘       └─────────────────┘         │ │
  │          │                         │                    │ │
  │          ▼                         │                    │ │
  │  ┌───────────────────────────────────────────────────┐  │ │
  │  │ 步骤 3: 记录当前目标值                             │  │ │
  │  │                                                    │  │ │
  │  │ obj_value = sess.run([train_op, obj],             │  │ │
  │  │                      feed_dict=feed)[1]           │  │ │
  │  │ # 注意: obj 在 batch 上计算                       │  │ │
  │  │                                                    │  │ │
  │  │ objective_values.append(obj_value)                │  │ │
  │  └───────────────────────────────────────────────────┘  │ │
  │                       │                                  │ │
  │                       └─────────────► 继续下一次迭代     │ │
  └──────────────────────────────────────────────────────────┘ │
                                 │                              │
                                 └──────────────────────────────┘
                                 │
                                 ▼

╔═══════════════════════════════════════════════════════════════════════╗
║                    【阶段 3】测试结果保存                               ║
╚═══════════════════════════════════════════════════════════════════════╝
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  1. 提取最终参数                          │
         │     final_params = [sess.run(p)          │
         │                     for p in params]     │
         └──────────────────────────────────────────┘
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  2. 保存测试结果                          │
         │     - objective_values: 损失曲线          │
         │     - parameters: 最终参数                │
         │     - records: 详细记录(可选)             │
         │       * grad_norm, param_norm            │
         │       * optimizer slot values             │
         └──────────────────────────────────────────┘
                                 │
                                 ▼
         ┌──────────────────────────────────────────┐
         │  3. 绘制和分析                            │
         │     - 绘制损失曲线                        │
         │     - 对比不同优化器性能                  │
         │     - 保存 pickle 文件                    │
         └──────────────────────────────────────────┘
                                 │
                                 ▼
                          ┌──────────┐
                          │ 测试完成 │
                          └──────────┘
```

### 7.2 元训练 vs 元测试的关键区别

| 维度 | 元训练 (Meta-Training) | 元测试 (Meta-Testing) |
|------|----------------------|---------------------|
| **目的** | 训练优化器的参数 | 评估优化器的泛化能力 |
| **数据集** | MNIST 训练集 (55000 样本) | **MNIST 测试集 (10000 样本)** |
| **优化器参数** | 通过元梯度更新 | **冻结**（从 checkpoint 加载） |
| **Optimizee 参数** | 每次元迭代重新初始化 | 初始化一次，持续优化 |
| **损失计算** | `obj` 在完整数据 (55000)<br>`current_obj` 在 batch (128) | 在 batch (128) 上计算<br>可选记录 full_obj (10000) |
| **迭代次数** | 短 (100-5000 步) | 长 (10000+ 步) |
| **Hessian 正则化** | 启用 (if `reg_optimizer=True`) | **不启用** |
| **MT 损失** | 启用 (if `if_mt=True`) | **不启用** |
| **反向传播** | 通过时间反向传播 (BPTT) | 仅前向传播 |

### 7.3 测试命令示例

```bash
cd L2O-Scale-Evaluation

python metatest.py \
    --train_dir=../L2O-Scale-Training/hess_cl_mt \
    --save_dir=hess_cl_mt_eval \
    --include_mnist_mlp_relu_problems \
    --model_name=mnist-relu \
    --restore_model_name=model.ckpt-0 \
    --num_testing_itrs=10000
```

**测试数据处理**：
- 加载 **MNIST 测试集** (`train=False`) - **10000 个样本**
- 使用相同的 batch_size (128)
- 但测试迭代数远超训练 (10000 vs 100-5000)
- 评估优化器在**未见过的测试数据**上的泛化能力和长时间优化的稳定性

---

## 8. 超参数配置

### 8.1 元训练超参数

| 参数名 | 默认值 | 说明 | 代码位置 |
|-------|--------|------|---------|
| `meta_learning_rate` | 1e-6 | 元优化器学习率 | metarun.py:73 |
| `gradient_clip_level` | 1e4 | 梯度裁剪门槛 | metarun.py:75 |
| `num_meta_iterations` | 100 | 元训练迭代次数 | metarun.py:45 |
| `fix_unroll_length` | 20 | 每次 unroll 步数 | metarun.py:230 |
| `alpha` | 1e-4 | Hessian 正则化权重 | trainable_optimizer.py:34 |
| `reg_option` | 'hessian-esd' | 正则化类型 | problem_generator.py:35 |
| `hessian_itrs` | 10 | Hessian 估计迭代次数 | problem_generator.py:33 |
| `if_cl` | True/False | 是否启用课程学习 | metarun.py:228 |
| `if_mt` | True/False | 是否启用模仿学习 | metaopt.py:69 |
| `mt_ratio` | 0.1 | MT 训练概率 | metaopt.py:70 |

### 8.2 HierarchicalRNN 超参数

| 参数名 | 默认值 | 说明 | 代码位置 |
|-------|--------|------|---------|
| `level_sizes` | [10, 20, 20] | 三层 RNN 隐藏层大小 | metarun.py:243 |
| `init_lr_range` | (1e-6, 1e-2) | 初始学习率范围 | metarun.py:154-156 |
| `num_gradient_scales` | 4 | 梯度多尺度数量 | metarun.py:194 |
| `learnable_decay` | True | 是否学习 decay 系数 | metarun.py:181 |
| `dynamic_output_scale` | True | 是否学习输出缩放 | metarun.py:184 |
| `use_gradient_shortcut` | True | 是否使用梯度捷径 | metarun.py:205 |

---

## 9. 使用指南

### 9.1 训练命令

**基础训练 (L2O-Scale)**:
```bash
cd L2O-Scale-Training

python metarun.py \
    --train_dir=hess_cl_mt \
    --regularize_time=none \
    --alpha=1e-4 \
    --reg_optimizer=True \
    --reg_option=hessian-esd \
    --include_mnist_mlp_problems \
    --num_problems=1 \
    --num_meta_iterations=100 \
    --fix_unroll=True \
    --fix_unroll_length=20 \
    --evaluation_period=1 \
    --evaluation_epochs=5 \
    --use_second_derivatives=False \
    --if_cl=False \
    --if_mt=False
```

**增强训练 (启用课程学习和模仿学习)**:
```bash
python metarun.py \
    --train_dir=hess_cl_mt \
    --regularize_time=none \
    --alpha=1e-4 \
    --reg_optimizer=True \
    --reg_option=hessian-esd \
    --include_mnist_mlp_problems \
    --num_problems=1 \
    --num_meta_iterations=100 \
    --fix_unroll=True \
    --fix_unroll_length=20 \
    --evaluation_period=1 \
    --evaluation_epochs=5 \
    --use_second_derivatives=False \
    --if_cl=True \
    --if_mt=True \
    --mt_ratio=0.1 \
    --mt_k=1
```

### 9.2 评估命令

```bash
cd L2O-Scale-Evaluation

python metatest.py \
    --train_dir=../L2O-Scale-Training/hess_cl_mt \
    --save_dir=hess_cl_mt_eval \
    --include_mnist_mlp_relu_problems \
    --model_name=mnist-relu \
    --restore_model_name=model.ckpt-0 \
    --num_testing_itrs=10000
```

---

## 10. 总结

本项目实现了带 Hessian 正则化的可学习优化器，主要贡献包括：

1. **分层 RNN 优化器架构**: 通过三层 RNN (逐参数、逐张量、全局) 实现不同粒度的优化信息聚合

2. **Hessian 正则化**: 使用 Hutchinson 估计或 Lanczos 算法计算 Hessian 信息，约束优化器泛化能力

3. **课程学习**: 从简单到复杂渐进式训练，提高收敛稳定性

4. **模仿学习**: 通过模仿 Adam 等成熟优化器加速前期收敛

该方法能够训练出泛化能力强、收敛速度快的可学习优化器，在多种优化问题上表现优异。
