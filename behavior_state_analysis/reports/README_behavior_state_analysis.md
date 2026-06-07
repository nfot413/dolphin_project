# Behavior and Emotion Candidate Analysis from Official Whistle and Click Data

## 1. 目标

本分析使用官方提供的 whistle 与 click train 数据，以及前面得到的 whistle/click 聚类结果，构建一个基于声学代理指标的行为与情绪状态候选评分系统。

重要说明：这些标签不是行为真值，也不是可靠情绪识别。它们只是根据声学特征得到的候选解释，用于探索性分析。

## 2. 输入数据

- `official_whistle_features_with_clusters.csv`：官方 303 个 whistle 片段的特征和聚类标签。
- `official_click_features_with_clusters.csv`：官方 click train 片段的特征和聚类标签。
- `whistles.txt`：官方 whistle 起止时间与类型/质量等级。
- `clicks.txt`：官方 click train 起止时间与 ICI 指标。

## 3. 方法

整段录音被划分为 30 秒时间窗口。每个窗口中统计 whistle 和 click train 的数量、聚类比例、ICI、click rate、能量、频谱重心等指标，然后计算若干候选分数。

### 评分维度

- `foraging_score`：短 ICI、高 click rate、高密度 click train。
- `social_score`：whistle 数量较多，常规交流型 whistle 比例较高。
- `courtship_affiliative_score`：社交分数较高、click 活动中等、冲突和觅食分数较低。
- `conflict_like_score`：高能量 whistle、强 click 活动、whistle/click 同时增强。
- `avoidance_stress_score`：高频/高能量 whistle、声学活动突变和较高唤醒。
- `positive_like_score`：由社交或觅食候选增强，同时冲突和压力分数较低。
- `negative_like_score`：由压力/躲避候选、冲突候选和高唤醒增强。

## 4. 行为/状态候选类别

### Foraging-like acoustic activity

- 窗口数量：83
- whistle 总数：39
- click train 总数：1554
- 平均 foraging score：0.449
- 平均 social score：0.125
- 平均 arousal score：0.108
- 平均 conflict-like score：0.233
- 平均 avoidance/stress score：0.102

由短 ICI、高 click rate 和高密度 click train 支持，是当前数据中相对最有依据的行为候选。它不等于确定觅食，但可作为 feeding-buzz-like 或目标接近阶段的声学代理。

### Social communication candidate

- 窗口数量：65
- whistle 总数：219
- click train 总数：988
- 平均 foraging score：0.252
- 平均 social score：0.509
- 平均 arousal score：0.073
- 平均 conflict-like score：0.265
- 平均 avoidance/stress score：0.067

由较高 whistle 活动和常规/稳定型 whistle 支持，更接近社交交流或群体联系候选。

### Uncertain mixed acoustic state

- 窗口数量：20
- whistle 总数：0
- click train 总数：143
- 平均 foraging score：0.177
- 平均 social score：0.045
- 平均 arousal score：0.000
- 平均 conflict-like score：0.078
- 平均 avoidance/stress score：0.025

多个分数接近或特征不足，无法给出更明确候选。

### Conflict-like high-intensity activity

- 窗口数量：14
- whistle 总数：33
- click train 总数：415
- 平均 foraging score：0.389
- 平均 social score：0.196
- 平均 arousal score：0.386
- 平均 conflict-like score：0.537
- 平均 avoidance/stress score：0.327

由高能量 whistle、较强 click 活动以及 whistle/click 同时增强支持。它不能证明打架，只能表示高强度脉冲/冲突样声学活动候选。

### Low-activity / no annotated acoustic event

- 窗口数量：9
- whistle 总数：0
- click train 总数：0
- 平均 foraging score：0.000
- 平均 social score：0.000
- 平均 arousal score：0.000
- 平均 conflict-like score：0.000
- 平均 avoidance/stress score：0.050

该时间窗内没有官方 whistle/click 标注事件或声学活动很弱。

### Avoidance / stress-like arousal

- 窗口数量：6
- whistle 总数：12
- click train 总数：223
- 平均 foraging score：0.350
- 平均 social score：0.188
- 平均 arousal score：0.526
- 平均 conflict-like score：0.400
- 平均 avoidance/stress score：0.472

由高频宽带 whistle、高能量 whistle 和活动突变支持。它不能证明躲避天敌，只能表示压力/受干扰/躲避样高唤醒候选。

## 5. 主要结论

1. 觅食样候选主要由 click 数据支持，尤其是短 ICI、高 click rate 和高密度 click train。因此它是本分析中相对最有声学依据的行为候选。
2. 社交交流与亲和/求偶样候选主要由 whistle 活动支持，但由于缺少同步行为录像，求偶标签只能作为非常弱的推断。
3. 冲突/打斗样候选和躲避/压力样候选主要依赖高能量、高频、强脉冲和活动突变等高唤醒线索；这些标签不应被视为确定行为。
4. 正负状态只能作为倾向性解释：foraging-like 和 social-like 更偏 positive-active candidate；avoidance/stress-like 和 conflict-like 更偏 negative-aroused candidate。但 valence 的可信度低于 arousal。

## 6. 输出文件

- `behavior_state_candidates.csv`：每个 30 秒窗口的特征、分数和最终候选标签。
- `behavior_state_summary.csv`：各行为/状态候选类别的统计汇总。
- `behavior_state_timeline.png`：候选状态随时间变化图。
- `behavior_state_score_profile.png`：各类候选分数随时间变化图。
- `behavior_state_summary.png`：候选状态数量分布图。

## 7. 使用限制

本分析没有同步视频、个体身份、群体组成、捕食者信息或真实行为标签。因此所有输出均为声学代理推断，而非确定行为识别。尤其是躲避天敌、打架、求偶和正负情绪，只能作为探索性候选标签。