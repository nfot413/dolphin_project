# 复杂版海豚候选声学状态环形声波可视化说明

## 1. 本阶段目的

本阶段是在基础环形声波图 `generated_dolphin_audio/visualization/` 的基础上新增的复杂化视觉版本。复杂版图像使用黑色背景、多层同心环、断续圆弧、外侧短径向线、发光粒子、点阵环和局部爆发点，形成更接近音乐可视化海报的圆形声波视觉效果。

这些图用于展示四种海豚声学状态候选的艺术化可视化，方便在汇报、海报、演示页或项目成果展示中使用。

## 2. 四种状态含义

1. `social_positive`：社交积极型  
   使用 `positive_active_dolphin_scene.wav`。图像结构更完整、平滑、有序，代表较稳定、明亮但不尖锐的社交交流候选声学状态。

2. `foraging_active`：觅食活跃型  
   使用 `positive_active_dolphin_scene.wav`。图像粒子更多、点阵更密集、局部爆发更明显，用于表现探索、回声定位或觅食样活跃状态候选。

3. `stress_avoidance`：压力/躲避型  
   使用 `negative_aroused_dolphin_scene.wav`。图像具有更强断裂感和不规则扰动，代表可能与压力、躲避或不稳定高唤醒相关的声学状态候选。

4. `conflict_like`：冲突/打斗样高强度型  
   使用 `negative_aroused_dolphin_scene.wav`。图像具有最长径向尖峰、最大爆发点和更强红橙/黄色脉冲，用于表现高强度冲突样声学活动候选。

## 3. 颜色、波动和粒子的含义

- 青蓝色、蓝色、浅绿色：用于 `social_positive`，强调平滑、有序、较稳定的声学状态。
- 绿色、青色、黄绿色：用于 `foraging_active`，强调密集、活跃和局部高能量爆发。
- 紫色、蓝紫色、粉紫色：用于 `stress_avoidance`，强调不规则、破碎和状态不稳定。
- 红橙色、黄色、深红色：用于 `conflict_like`，强调尖锐、高强度和明显脉冲。

波动高度主要由音频 amplitude envelope 控制。粒子密度表示视觉上的活动密集程度，断续弧数量和破碎程度表示状态结构的稳定或不稳定，外侧爆发点表示包络局部峰值附近的高能量区域。

## 4. 重要限制

这些复杂版图像不是严格频谱图，也不是行为真值图。它们没有替代 spectrogram、功率谱、whistle 轮廓分析或 click train 参数统计。

四种状态标签来自前期 whistle/click 特征提取、聚类分析和候选行为解释，是基于合成音频和聚类解释的视觉表达，不代表已经确定海豚真实情绪。

## 5. 输出文件

- `generated_dolphin_audio/visualization_complex/figures/social_positive_complex_radial_wave.png`
- `generated_dolphin_audio/visualization_complex/figures/foraging_active_complex_radial_wave.png`
- `generated_dolphin_audio/visualization_complex/figures/stress_avoidance_complex_radial_wave.png`
- `generated_dolphin_audio/visualization_complex/figures/conflict_like_complex_radial_wave.png`
- `generated_dolphin_audio/visualization_complex/figures/four_state_complex_radial_overview.png`

对应脚本：

- `generated_dolphin_audio/visualization_complex/scripts/make_complex_radial_wave_plots.py`

## 6. 与基础版 visualization 目录的区别

基础版 `generated_dolphin_audio/visualization/` 更强调清晰、简洁的环形声波表达，主要包含单层主声波、基础光晕、点阵和径向线。

复杂版 `generated_dolphin_audio/visualization_complex/` 在不覆盖基础版结果的前提下，增加了多层同心环、双层包络声波、断续圆弧、发光粒子、外侧短径向线、点阵环和局部爆发区域，视觉层次更丰富，更适合海报化展示。
