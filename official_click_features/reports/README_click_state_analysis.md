# Click Train Acoustic State Analysis

## 1. 分析目标

本分析基于官方 `clicks.txt` 给出的 click train 时间段，从完整原始录音中切分 click train 片段，并对这些片段提取声学特征后进行聚类。

本分析的目标不是直接证明海豚某一时刻正在进行某种确定行为，而是根据 click train 的声学结构推断若干**行为/声学状态候选**，尤其关注回声定位强度、短 ICI 密集点击活动，以及可能的觅食或 feeding-buzz-like 候选片段。

## 2. 输入数据与输出文件

输入数据包括：

- 官方 click train 时间标注：`data/labels/clicks.txt`
- 原始完整录音：`data/raw/full_recording.wav`
- 切分后的 click train 音频片段：`official_click_features/segments/`
- 提取后的 click train 特征表：`official_click_features/features/official_click_features.csv`

本阶段整理出的最终结果文件包括：

- `click_state_cluster_summary_final.csv`：最终 click cluster 解释表
- `click_cluster_profile_chart.png`：不同 cluster 的归一化声学特征画像图
- `click_cluster_counts.png`：不同 cluster 的数量分布图
- `README_click_state_analysis.md`：本说明文件

## 3. 使用的主要特征

对每个官方 click train 片段提取或整理了以下特征：

- `original_duration_sec`：官方标注片段的持续时间
- `official_ici`：官方给出的 ICI 指标；在最终表中以均值和中位数表示
- `rms_energy`、`peak_amplitude`：能量与峰值振幅
- `spectral_centroid_hz`、`spectral_bandwidth_hz`、`dominant_frequency_hz`：频谱结构特征
- `bandwidth_est_hz`：有效频率跨度估计
- `estimated_peak_count`：在切片音频中估计的脉冲峰数量
- `estimated_click_rate_per_sec`：估计每秒 click 峰数量
- `estimated_mean_ici_sec`、`estimated_min_ici_sec`：基于峰值估计的 ICI 统计

## 4. 聚类结果

KMeans 聚类中，silhouette score 在当前测试范围内选择的最佳聚类数为 **k = 3**，对应 silhouette score = **0.3581**。最终得到 3 个 click train cluster。

### Cluster 总览

| Cluster | 数量 | 中文标签 | 英文标签 | 核心解释 |
|---:|---:|---|---|---|
| 0 | 1793 | 高频宽带常规回声定位型 | Broadband regular echolocation candidate | 该类数量最多，片段时长中等，频谱重心和带宽较高，但 click rate 与 ICI 不如 Cluster 2 极端。可作为常规但较宽带的回声定位/脉冲活动候选。 |
| 1 | 1184 | 低能量低频短促 click train 型 | Low-energy low-frequency click-train candidate | 该类能量最低，频谱重心和有效高频范围最低，片段较短。虽然估计 click rate 不低，但频谱更偏低频，可能代表较弱的 click train、较远距离信号、背景影响较强的片段，或较低强度回声定位活动。 |
| 2 | 346 | 高密度短 ICI 觅食/feeding-buzz 候选型 | Short-ICI high-density foraging/buzz-like candidate | 该类数量最少，但片段时长最长、能量最高、估计 peak 数量最多、click rate 最高、mean ICI 最短，官方 ICI 中位数也最低。声学上最接近密集回声定位或 feeding-buzz-like 活动，可作为觅食/目标接近阶段的候选片段。 |

## 5. 各 cluster 的详细解释

### Cluster 0: 高频宽带常规回声定位型

**English label:** Broadband regular echolocation candidate

**样本数量：** 1793

**主要平均特征：**

- 平均片段时长 `original_duration_sec`: 0.4238 s
- 官方 ICI 均值 `official_ici_mean`: 24.5901
- 官方 ICI 中位数 `official_ici_median`: 15.0
- RMS 能量 `rms_energy`: 0.1217
- 频谱重心 `spectral_centroid_hz`: 19825.6424 Hz
- 有效频率跨度 `bandwidth_est_hz`: 27835.4364 Hz
- 估计峰数量 `estimated_peak_count`: 42.6286
- 估计 click rate `estimated_click_rate_per_sec`: 75.3088 /s
- 估计 mean ICI `estimated_mean_ici_sec`: 0.0166 s

**状态解释：**

该类数量最多，片段时长中等，频谱重心和带宽较高，但 click rate 与 ICI 不如 Cluster 2 极端。可作为常规但较宽带的回声定位/脉冲活动候选。

### Cluster 1: 低能量低频短促 click train 型

**English label:** Low-energy low-frequency click-train candidate

**样本数量：** 1184

**主要平均特征：**

- 平均片段时长 `original_duration_sec`: 0.3751 s
- 官方 ICI 均值 `official_ici_mean`: 35.8488
- 官方 ICI 中位数 `official_ici_median`: 30.0
- RMS 能量 `rms_energy`: 0.0665
- 频谱重心 `spectral_centroid_hz`: 3488.2692 Hz
- 有效频率跨度 `bandwidth_est_hz`: 2290.9364 Hz
- 估计峰数量 `estimated_peak_count`: 47.8285
- 估计 click rate `estimated_click_rate_per_sec`: 107.2121 /s
- 估计 mean ICI `estimated_mean_ici_sec`: 0.0095 s

**状态解释：**

该类能量最低，频谱重心和有效高频范围最低，片段较短。虽然估计 click rate 不低，但频谱更偏低频，可能代表较弱的 click train、较远距离信号、背景影响较强的片段，或较低强度回声定位活动。

### Cluster 2: 高密度短 ICI 觅食/feeding-buzz 候选型

**English label:** Short-ICI high-density foraging/buzz-like candidate

**样本数量：** 346

**主要平均特征：**

- 平均片段时长 `original_duration_sec`: 3.5873 s
- 官方 ICI 均值 `official_ici_mean`: 6.2717
- 官方 ICI 中位数 `official_ici_median`: 5.0
- RMS 能量 `rms_energy`: 0.2278
- 频谱重心 `spectral_centroid_hz`: 20759.5784 Hz
- 有效频率跨度 `bandwidth_est_hz`: 28885.8382 Hz
- 估计峰数量 `estimated_peak_count`: 1006.9509
- 估计 click rate `estimated_click_rate_per_sec`: 266.7626 /s
- 估计 mean ICI `estimated_mean_ici_sec`: 0.0041 s

**状态解释：**

该类数量最少，但片段时长最长、能量最高、估计 peak 数量最多、click rate 最高、mean ICI 最短，官方 ICI 中位数也最低。声学上最接近密集回声定位或 feeding-buzz-like 活动，可作为觅食/目标接近阶段的候选片段。

## 6. 主要结论

1. 官方 click train 片段可分为三个相对不同的声学状态候选类型。
2. Cluster 2 最突出：它具有最长片段时长、最高能量、最高 peak count、最高 click rate 和最短 estimated mean ICI，因此最适合作为短 ICI 高密度 click activity，或 feeding-buzz-like / foraging-like 候选。
3. Cluster 0 是数量最多的一类，具有较高频谱重心和较宽频带，但密度不如 Cluster 2，可解释为较常规的宽带回声定位/脉冲活动。
4. Cluster 1 能量和频率指标最低，可能对应较弱、较低频或受背景影响更明显的 click train 活动。
5. 由于缺乏同步视频或行为标签，这些类别不能直接被称为“确定觅食”“确定探索”或“确定情绪”。更稳妥的说法是：它们代表不同的 click-train acoustic states，其中 Cluster 2 是最强的觅食/feeding-buzz-like 候选。

## 7. 后续可如何使用这些结果

- 将 Cluster 2 的时间段作为优先回看对象，检查其频谱图和原始音频，判断是否符合 feeding buzz 或密集回声定位活动。
- 将 click cluster 与 whistle cluster 在时间轴上合并，分析高密度 click 活动是否伴随高唤醒 whistle 类别。
- 进一步按 1 秒或 5 秒窗口统计 click rate 与 whistle rate，构建整段录音的声学活动时间线。
