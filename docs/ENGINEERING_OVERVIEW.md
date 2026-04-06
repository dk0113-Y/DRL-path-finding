# 当前工程介绍（按最新主线）

## 1. 工程定位

本工程是一个面向未知二维栅格地图自主探索任务的深度强化学习（DRL）训练框架。当前正式主线已经收口到：

- 动态累计认知地图（Cumulative Belief Map）
- 共享语义层（Shared Semantic Layer）
- 双状态输入（Advantage State + Value State）
- 语义解耦的 Dueling Q 网络（Semantic Dueling Head）

当前主线不再使用旧的 near/mid/token 三分支主架构；旧模块若仍在仓库中，属于历史参考或底层复用，不作为当前训练语义主路径。

## 2. 环境与观测主线

### 2.1 地图与观测

- 随机地图由 `RandomMapGenerator` 生成。
- 智能体使用局部雷达式占据观测，而不是直接读取全局真值地图。
- 每一步局部观测会写入 `CumulativeBeliefMap`，形成随探索逐步扩张的累计认知地图。

### 2.2 累计认知地图（Cumulative Belief Map）

`CumulativeBeliefMap` 当前承担以下职责：

- 维护 agent-side belief map，而非直接暴露真值图
- 维护访问计数 `visit_count`
- 维护增量 frontier cache
- 维护 zero-margin `analysis_box`
- 维护 coverage 统计：
  - 分母 = 4 邻域 reachable free + 邻接障碍边界
  - 分子 = 新揭示已知格的增量累计

### 2.3 analysis_box 当前语义

当前 `analysis_box` 已经改为 **zero margin**：

- 只覆盖当前已知区域的最小外接矩形
- 不再向外额外扩未知带
- 不在表征层猜测 known boundary 之外的未知拓扑

这意味着共享语义层只分析“当前已知边界截出来的未知结构”，而不是人为外推更远未知海洋。

## 3. 当前共享语义层：frontier-first 三层语义树

当前 `SharedSemanticLayer` 的正式主线是 frontier-first，而不是旧的 unknown-first。

当前语义树为：

- `UnknownBlock`
  - `FrontierCluster`
    - `SupportGeometry`

### 3.1 FrontierCluster（纯前沿簇）

- 对 `analysis_box` 内的 pure frontier cells 做 **8 连通分簇**
- 这一步只作用于 pure frontier cells
- 不将 support 区混入第一层分簇对象

当前 `FrontierCluster` 表示：

- 一段 8 连通的纯前沿边界簇
- 具有自身的 `frontier_geometry`
- 具有稳定锚点 `frontier_anchor_rc`
- 位置与距离相对 agent 的定义围绕该锚点展开

### 3.2 SupportGeometry（前沿簇局部已知侧支撑描述）

当前 `SupportGeometry` **不再**由固定步数形态学膨胀生成。

现在的定义是：

- 以 frontier cluster 的 tight bbox 为中心
- 四周做对称 padding = 2
- 得到 frontier-cluster local analysis box
- 只在该局部框中对 **已知侧内容** 做统计

当前 `SupportGeometry` 主要提供：

- `support_free_geometry`
- `support_area`
- `support_clearance`
- `support_complexity`

其中：

- `support_area` 来自局部框中的已知 free cells 数量
- `support_clearance / support_complexity` 基于局部框中 known cells 的分布统计
- 不再把 support 解释为“从前沿向已知方向膨胀出来的一片栅格”

### 3.3 UnknownBlock（未知块）

当前 `UnknownBlock` 由 frontier-first unknown grouping 形成：

- 先由 frontier clusters 在 unknown 侧给相邻 unknown cells 打 owner
- 再通过统一多源传播在 unknown 区域中扩展
- 若不同 frontier clusters 的传播波在 unknown 中汇合，则合并为同一块

因此，当前 `UnknownBlock` 的语义不是“analysis box 内所有 unknown 的先验静态连通块”，而是：

- 一片由 frontier-first unknown-side grouping 形成的未知推进主干
- 其下可挂接一个或多个 frontier clusters

## 4. 语义树为网络提供的表征信息

当前由累计地图构造出的“未知块—前沿树”主要向网络提供两类表征：

### 4.1 Advantage 分支使用的局部表征

由 `AdvantageStateBuilder` 构造 `Advantage Canvas`，当前为 6 通道：

1. `unknown`
2. `free`
3. `obstacle`
4. `main_entry_mask`
5. `nonmain_entry_mask`
6. `main_block_fragment_mask`

其中与语义树直接相关的三项是：

- `main_entry_mask`：主未知块下的 pure frontier clusters 局部掩码
- `nonmain_entry_mask`：非主未知块下的 pure frontier clusters 局部掩码
- `main_block_fragment_mask`：主未知块在当前局部窗口中的 unknown fragment

注意：

- 当前 advantage 分支画的是 **pure frontier geometry**
- 不再画旧 support dilation 区

### 4.2 Value 分支使用的层级结构化表征

由 `ValueStateBuilder` 将语义树整理为 block-tree tensor state。

#### Block 特征（7 维）

1. `block_area_ratio`
2. `bbox_height_ratio`
3. `bbox_width_ratio`
4. `bbox_aspect_ratio`
5. `frontier_cluster_count_ratio`
6. `nearest_frontier_dist_ratio`
7. `opportunity_score`

这些特征表达的是 UnknownBlock 层的全局价值信息：

- 这块未知主干有多大
- 形状如何
- 挂了多少个前沿簇
- 最近前沿簇离 agent 多远
- 该未知块的综合推进价值

#### Entry / Frontier 特征（6 维）

1. `entry_dir_r`
2. `entry_dir_c`
3. `entry_dist_ratio`
4. `entry_width_ratio`
5. `support_clearance`
6. `support_area_ratio`

来源分工：

- 前 4 维来自 `FrontierCluster`
- 后 2 维来自 `SupportGeometry`

因此当前 value 分支中的“entry feature”其实已经不是旧意义上的复合 entry，而是：

- pure frontier cluster 几何属性
- 附属 support geometry 的局部已知侧统计

## 5. 网络两条支路及其结构

## 5.1 总体网络

主网络定义在 `agents/q_value_agent.py`，名称为 `ExplorationQNetwork`。

数据流为：

- `advantage_canvas` -> `AdvantageCanvasEncoder` -> per-action advantage states
- `value_block_features` + `value_entry_features` + masks -> `ValueTreeEncoder` -> global value state
- `SemanticDuelingHead` -> Q-values

## 5.2 Advantage 支路

输入：

- `advantage_canvas`，shape = `[B, C, H, W]`
- 当前 `C = 6`

编码器：`encoders/advantage_encoder.py` 中的 `AdvantageCanvasEncoder`

当前结构：

- 卷积 backbone：
  - Conv + BN + GELU
  - 多个 residual block
- 得到局部特征图后，针对每个动作构造 directional masks
- 对每个动作执行定向池化，得到 action-specific pooled feature
- 同时读取：
  - 当前中心特征
  - 对应动作的 landing-cell 特征
  - 动作 embedding
- 拼接后通过 MLP，得到每个动作一个 `action_state`

输出：

- shape = `[B, A, D_adv]`
- 当前 `A = 8`
- 当前 `D_adv = 160`

语义角色：

- 负责刻画“当前局部一步怎么走更有优势”
- 输入以局部几何 + 局部前沿簇位置 + 主未知块局部碎片为主

## 5.3 Value 支路

输入：

- `value_block_features`，shape = `[B, N, D_block]`
- `value_entry_features`，shape = `[B, N, M, D_entry]`
- `value_block_mask`
- `value_entry_mask`

当前维度：

- `D_block = 7`
- `D_entry = 6`

编码器：`encoders/value_encoder.py` 中的 `ValueTreeEncoder`

当前结构：

### 第一步：编码 FrontierCluster / SupportGeometry 子层

- 对 `entry_features` 逐 entry 做 MLP 编码，得到 `entry_repr`
- 对 block 内各 entry 做 attention pooling
- 同时计算：
  - weighted summary
  - masked mean
  - masked max

### 第二步：融合到 UnknownBlock 层

- 将 block 原始特征与 entry 聚合结果拼接
- 经 `block_fusion` 得到 block representation

### 第三步：全局 block 聚合

- 对 blocks 做 block attention
- 计算：
  - weighted summary
  - masked mean
  - masked max
- 再经 `state_head` 得到最终 `value_state`

输出：

- shape = `[B, D_value]`
- 当前 `D_value = 192`

语义角色：

- 负责刻画“当前整体未知推进局势值不值”
- 主对象是 UnknownBlock
- FrontierCluster 作为子对象附着在 block 下参与聚合

## 5.4 Semantic Dueling Head

定义在 `heads/semantic_dueling_head.py`

输入：

- `value_state`，shape = `[B, D_value]`
- `advantage_state`，shape = `[B, A, D_adv]`

当前结构：

- `value_head(value_state) -> V(s)`
- `advantage_head(advantage_state) -> A(s, a)`
- 通过 dueling aggregation 输出：
  - `Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)`

这意味着：

- Value 支路提供全局局势价值
- Advantage 支路提供动作相关局部优势
- 决策头只负责最后的语义解耦聚合，不重新做地图解释

## 6. 状态适配器（StateTensorAdapter）

`StateTensorAdapter` 在 `agents/q_value_agent.py` 中负责把环境侧对象变成网络张量。

它的顺序是：

1. `SharedSemanticLayer.analyze()` 先构造 `SharedSemanticSnapshot`
2. `AdvantageStateBuilder` 构造 `advantage_canvas`
3. `ValueStateBuilder` 构造 block-tree tensor state
4. 转成 batch tensor 后送入网络

输出的五个核心张量是：

- `advantage_canvas`
- `value_block_features`
- `value_entry_features`
- `value_block_mask`
- `value_entry_mask`

## 7. 当前方法语义总结

按当前正式主线，可以把整套方法概括为：

- belief map 不是直接做 end-to-end 像素决策
- 而是先在 zero-margin analysis box 中解析 frontier-first 语义树：
  - UnknownBlock
  - FrontierCluster
  - SupportGeometry
- 局部分支读取局部 canvas，强调 frontier 几何与主未知块局部碎片
- 全局分支读取 block-tree 状态，强调未知主干价值与子前沿簇支持信息
- 最后通过 semantic dueling head 融合为 Q 值

## 8. 当前与旧版本相比的关键变化

相对旧实现，当前正式主线已经明确发生以下变化：

1. `analysis_box` 改为 zero margin
2. 删除 revisit/recency 语义链
3. coverage 改为 reachable free + obstacle boundary 域
4. 共享语义层改为 frontier-first
5. FrontierCluster 改为 8 连通 pure frontier cluster
6. SupportGeometry 改为 frontier-cluster local box 内的已知侧统计
7. 表征层移除 per-entry A*
8. advantage 分支画纯 frontier，不再画旧 support dilation 区
9. value 分支 entry 特征显式拆分为 FrontierCluster 几何 + SupportGeometry 统计

## 9. 推荐阅读顺序

若要快速理解当前代码主线，建议按以下顺序阅读：

1. `env/core_cummap.py`
2. `env/shared_semantic_layer.py`
3. `env/advantage_state_builder.py`
4. `env/value_state_builder.py`
5. `encoders/advantage_encoder.py`
6. `encoders/value_encoder.py`
7. `heads/semantic_dueling_head.py`
8. `agents/q_value_agent.py`

## 10. 当前文档用途

本文件用于说明“当前正式工程主线”，优先服务于：

- 继续做方法讨论
- 生成 Codex 修改指令
- 审查新代码是否偏离主线
- 论文方法图与工程实现的一致性校对
