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

当前代码中的 `SupportGeometry` 主要提供：

- `local_box_bounds`
- `support_free_geometry`
- `support_obstacle_density`

其中：

- `support_free_geometry` 保留局部框中的已知 free-cell 几何，主要用于可视化/调试。
- `support_obstacle_density` 是当前学习侧使用的局部已知侧障碍密度统计。
- 不再把 support 解释为“从前沿向已知方向膨胀出来的一片栅格”，也不再把 `support_area / support_clearance / support_complexity` 作为当前 value entry 的学习输入。

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

由 `AdvantageStateBuilder` 构造 `Advantage Canvas`，当前代码定义为 5 通道：

1. `free`
2. `obstacle`
3. `frontier_block_area_map`
4. `visit_count_log_norm`
5. `recent_trajectory_decay`

其中：

- `free` / `obstacle` 来自当前累计 belief map 在雷达局部窗口中的采样。
- `frontier_block_area_map` 在局部可见 frontier cluster 位置写入其所属 UnknownBlock 的面积占比。
- `visit_count_log_norm` 是由累计访问计数生成的 log-normalized revisit pressure。
- `recent_trajectory_decay` 绘制最近轨迹中通向当前状态的历史位置，不重新标记当前 agent 中心格。

注意：

- 当前 advantage 分支画的是 **pure frontier geometry**
- 不再画旧 support dilation 区
- 当前代码不包含 `unknown`、`main_entry_mask`、`nonmain_entry_mask` 或 `main_block_fragment_mask` 通道。

### 4.2 Value 分支使用的层级结构化表征

由 `ValueStateBuilder` 将语义树整理为 block-tree tensor state。

#### Block 特征（2 维）

1. `block_area_ratio`
2. `frontier_cluster_count`

这些特征表达的是 UnknownBlock 层的直接摘要信息：

- 这块未知区域占全部可达未知面积的比例
- 该块当前挂了多少个 pure frontier clusters

block 节点不再携带单一代表入口 / 代表前沿信息。
block-entry 对应关系由层级张量结构表达：`entry_features[block_slot, ...]`
就是该 block 的子入口集合，学习侧 block 特征不再手工复制 child-entry identity。

#### Entry / Frontier 特征（4 维）

1. `delta_r_ratio`
2. `delta_c_ratio`
3. `entry_width_ratio`
4. `support_obstacle_density`

来源分工：

- 前 3 维来自 `FrontierCluster`
- 第 4 维来自 `SupportGeometry`

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
- 当前 `C = 5`

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
- 输入以局部 free/obstacle 几何、frontier block-area 投影、累计访问压力和短时轨迹几何为主

## 5.3 Value 支路

输入：

- `value_block_features`，shape = `[B, N, D_block]`
- `value_entry_features`，shape = `[B, N, M, D_entry]`
- `value_block_mask`
- `value_entry_mask`

当前维度：

- `D_block = 2`
- `D_entry = 4`

编码器：`encoders/value_encoder.py` 中的 `ValueTreeEncoder`

当前结构：

### 第一步：编码 UnknownBlock 与 FrontierCluster / SupportGeometry 子层

- 对 `block_features` 做 MLP 编码，得到 parent block token。
- 对 `entry_features` 逐 entry 做 MLP 编码，得到 child entry token。
- 对同一 block 下的 sibling entries 做 masked self-attention。

### 第二步：parent-conditioned child aggregation

- parent block token 作为 query 聚合其子 entry tokens。
- child summary 通过 parent-grounded gated update 融入 block representation。
- 该路径不选择单一代表 entry，也不使用 mean/max fallback pooling 作为学习主路径。

### 第三步：全局 block 聚合

- 对 block representations 做 masked global block self-attention。
- 使用 learned state query 对有效 block 做 weighted pooling。
- 经 `value_state_head` 得到最终 `value_state`。

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
- 局部分支读取局部 canvas，强调局部 free/obstacle、frontier block-area 投影、累计访问压力与短时轨迹
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
8. advantage 分支画 pure frontier 的 block-area 投影，并加入累计访问压力与短时轨迹通道；不再画旧 support dilation 区，也不包含旧 main/nonmain entry mask
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
