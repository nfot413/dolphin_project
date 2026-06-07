# 海豚声学状态合成音频说明

## 1. 生成目的

本文件夹包含两段根据前期 whistle 与 click 聚类分析结果生成的合成海豚声学场景：

1. `positive_active_dolphin_scene.wav`
2. `negative_aroused_dolphin_scene.wav`

这两段声音并不是对真实海豚个体声音的复制，也不是严格意义上的“海豚情绪生成模型”。它们是基于前期声学分析结果构建的参数化合成示例，用来直观展示两类不同的海豚声学状态候选。

每段音频都混合了以下成分：

- 类似海豚 whistle 的连续调频信号；
- 类似海豚 click 的短脉冲串；
- 弱海底背景噪声，用来模拟真实水下录音环境。

## 2. 积极活跃候选声场

对应文件：

- `audio/positive_active_dolphin_scene.wav`
- `figures/positive_active_spectrogram.png`

该音频用于模拟一种“积极活跃”或“社交/觅食活跃”候选状态。

它的设计依据是：

- whistle 数量适中；
- whistle 频率处于中等范围；
- whistle 轮廓较平滑、较有规律；
- click train 存在，但整体密度不过分极端；
- 偶尔出现较短 ICI 的 click burst，用来模拟探索或觅食样活动；
- 背景噪声较弱，整体声场较清晰。

因此，这段声音可以被理解为：

> 稳定交流 + 中等回声定位活动 + 可能的探索/觅食样活跃状态候选。

这里的“积极”并不等同于人类意义上的“快乐”。它只是表示该声场更接近社交活跃、觅食活跃或较有组织的高活动状态。

## 3. 消极高唤醒候选声场

对应文件：

- `audio/negative_aroused_dolphin_scene.wav`
- `figures/negative_aroused_spectrogram.png`

该音频用于模拟一种“消极高唤醒”或“压力/干扰/冲突样”候选状态。

它的设计依据是：

- whistle 出现更频繁；
- whistle 频率整体更高；
- whistle 调频范围更宽；
- whistle 轮廓更不规则；
- click / pulse train 更密集；
- 脉冲幅度更强；
- 时间分布更不规则；
- 背景噪声更强。

因此，这段声音可以被理解为：

> 高频 whistle + 密集脉冲活动 + 更强背景噪声共同构成的高唤醒声学场景。

这里的“消极”也不能直接理解为“害怕”“愤怒”或“痛苦”。它只是表示该声场在声学特征上更接近压力、干扰、冲突样活动或不稳定高唤醒状态的候选。

## 4. 与前期聚类分析的关系

前期分析中，我们分别对官方 whistle 和 click 数据进行了声学特征提取和聚类。合成音频的参数设计参考了这些结果。

### Whistle 部分

积极活跃候选声场更多参考：

- 常规低活跃型 whistle；
- 中等频率、较平滑的 whistle；
- 较稳定的社交交流候选信号。

消极高唤醒候选声场更多参考：

- 高频宽带 whistle；
- 高能量 whistle；
- 调频更复杂、更不规则的 whistle。

### Click 部分

积极活跃候选声场使用：

- 中等密度 click train；
- 偶尔出现较短 ICI 的 click burst；
- 较有规律的点击声节奏。

消极高唤醒候选声场使用：

- 更密集的 click / pulse train；
- 更短、更不规则的 ICI；
- 更强幅度的脉冲活动。

## 5. 重要限制

这两段声音是合成示例，不应该被当作真实海豚情绪声音样本。

本项目中使用的“积极”和“消极”只是声学状态候选标签：

- “积极活跃”表示社交/觅食/探索样的较有组织活跃状态；
- “消极高唤醒”表示压力/干扰/冲突样的高唤醒声学状态。

由于缺少同步视频、行为记录、个体身份、群体结构和环境事件标注，本项目不能直接证明海豚处于某种具体情绪中。

因此，更准确的表述应为：

> 这两段音频是基于 whistle 与 click 聚类结果构建的参数化海豚声学状态模拟，而不是严格的海豚情绪识别或情绪生成结果。

## 6. 输出文件

本阶段生成的文件包括：

- `audio/positive_active_dolphin_scene.wav`
- `audio/negative_aroused_dolphin_scene.wav`
- `figures/positive_active_spectrogram.png`
- `figures/negative_aroused_spectrogram.png`
- `reports/README_generated_audio.md`

## 7. 后续可改进方向

后续如果想让合成音频更接近真实海豚录音，可以继续改进：

1. 使用真实 whistle 轮廓拟合参数，而不是手工设定频率轨迹；
2. 从真实 click train 中估计 ICI 分布，再用于 click 合成；
3. 引入背景海洋噪声样本，而不是使用简单随机噪声；
4. 使用更多行为标注数据，建立更可靠的状态分类规则；
5. 在数据量足够时，尝试训练神经音频生成模型。

当前版本的优势是：参数清楚、逻辑可解释、与前期聚类结果直接对应，适合作为项目最终展示中的声学状态模拟示例。