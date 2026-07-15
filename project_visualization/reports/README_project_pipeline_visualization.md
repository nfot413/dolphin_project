# 海豚声学状态项目流程总图说明

## 目的

`dolphin_acoustic_state_pipeline_overview.png` 用一张横向技术总图呈现项目从标注音频到声学状态视觉表达的完整过程。设计以现有项目数据和结果图为依据，减少说明文字，通过数据密度、特征画像、聚类分布、状态时间线、声谱和环形声波之间的视觉连接表达流程。

## 六个阶段

1. `ANNOTATED AUDIO`：303 个 whistle 与 3323 个 click train。
2. `FEATURE PROFILES`：频率、能量、带宽、持续时间和 click rate 等特征画像。
3. `PCA + KMEANS`：whistle 和 click train 的三类聚类结果。
4. `STATE WINDOWS`：197 个 30 秒窗口的候选状态评分与时间分布。
5. `SYNTHETIC SCENES`：积极活跃与消极高唤醒两类参数化合成声场。
6. `RADIAL STATE IMPRESSIONS`：平静、社交、觅食、压力和冲突五种环形视觉意象。

## 科学边界

图中的状态是 whistle/click 声学特征和聚类结果支持的候选解释，不是同步行为观测得到的真实情绪标签。环形图属于艺术化声学表达，不是频谱图或行为识别结果。

## 生成方式

从项目根目录运行：

```bash
python project_visualization/scripts/make_project_pipeline_visualization.py
```

输出文件：

- `project_visualization/figures/dolphin_acoustic_state_pipeline_overview.png`
