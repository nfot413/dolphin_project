# 四种候选海豚声学状态环形声波可视化说明

## 1. 可视化目的

本目录中的环形声波图用于把前期合成的两类海豚声学场景进一步转化为适合展示的视觉材料。图像采用黑色背景、中心空洞、霓虹发光圆环和径向波动形式，突出不同候选声学状态在整体能量包络、波动强度和节奏密度上的差异。

这些图片面向项目汇报、展示页和阶段性成果说明，帮助读者直观看到不同 whistle/click 组合所对应的声学状态候选。

## 2. 四种状态含义

1. `social_positive`：社交积极型  
   英文标题为 `Social-positive acoustic state`。该状态使用 `positive_active_dolphin_scene.wav`，表示较平滑、较稳定的社交交流候选声学状态。

2. `foraging_active`：觅食活跃型  
   英文标题为 `Foraging-active acoustic state`。该状态同样使用 `positive_active_dolphin_scene.wav`，但在视觉上表现为更密集、更活跃的径向波动，用于表达探索、回声定位或觅食样活动候选。

3. `stress_avoidance`：压力/躲避型  
   英文标题为 `Stress / avoidance acoustic state`。该状态使用 `negative_aroused_dolphin_scene.wav`，表示更高唤醒、更不规则、可能与压力或躲避相关的声学状态候选。

4. `conflict_like`：冲突/打斗样高强度型  
   英文标题为 `Conflict-like high-intensity acoustic state`。该状态使用 `negative_aroused_dolphin_scene.wav`，在视觉上使用最高、最尖锐的波动，用于表达高强度冲突样声学活动候选。

## 3. 颜色与波动高度含义

- 青蓝色 `#00E5FF`：社交积极型，波动中等且更平滑，强调稳定交流。
- 绿色 `#39FF88`：觅食活跃型，波动中高且更密集，强调探索和回声定位活动。
- 紫色 `#B366FF`：压力/躲避型，波动较高且更不规则，强调高唤醒和不稳定。
- 红橙色 `#FF5A36`：冲突/打斗样高强度型，波动最高且更尖锐，强调强烈脉冲和高强度活动。

图像中的径向起伏来自音频 amplitude envelope。脚本会把 WAV 自动转为 mono，归一化为 float，使用 Hilbert transform 提取包络，重采样到 720 个点，并通过非线性增强让波动在视觉上更清晰。

## 4. 重要说明

这些环形声波图是声学状态的艺术化可视化，不是严格的科学频谱图，也不能替代 spectrogram、功率谱密度、whistle 轮廓分析或 click train 参数统计。

本项目中的四个状态标签是基于 whistle/click 特征提取、聚类结果和行为/状态候选分析得到的解释性标签，不是确定的海豚真实情绪。由于缺少同步视频、个体身份、群体结构、环境事件和人工行为标注，不能直接证明海豚处于某种具体情绪。

因此，这些图更准确的用途是：

> 展示基于合成海豚声学场景的四类候选声学状态视觉表达，而不是宣称完成了严格的海豚情绪识别。

## 5. 输出文件列表

脚本运行后会生成以下图片：

- `generated_dolphin_audio/visualization/figures/social_positive_radial_wave.png`
- `generated_dolphin_audio/visualization/figures/foraging_active_radial_wave.png`
- `generated_dolphin_audio/visualization/figures/stress_avoidance_radial_wave.png`
- `generated_dolphin_audio/visualization/figures/conflict_like_radial_wave.png`
- `generated_dolphin_audio/visualization/figures/four_state_radial_wave_overview.png`

对应脚本为：

- `generated_dolphin_audio/visualization/scripts/make_radial_wave_plots.py`
