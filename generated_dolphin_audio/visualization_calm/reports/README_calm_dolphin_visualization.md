# 平静海豚叫声环形声波可视化说明

本图是在前面环形声波可视化风格基础上新增的一张“平静的海豚叫声”视觉意象图。

由于现有合成音频主要对应积极活跃和消极高唤醒两类状态，本图优先选择较适合作为平静基础的 `positive_active_dolphin_scene.wav`，并对其 amplitude envelope 做强平滑、动态范围压缩和低振幅约束，使环形波形呈现更柔和、均匀、舒展的状态。

图像不是严格科学频谱图，也不代表真实行为标注中的确定“平静情绪”。它是基于已有海豚合成音频进行平滑化处理后的艺术化视觉表达，重点表现平静、海洋感、科技感和柔和声波意象。

输出文件：

- `generated_dolphin_audio/visualization_calm/figures/calm_dolphin_call_radial_wave.png`

生成脚本：

- `generated_dolphin_audio/visualization_calm/scripts/make_calm_dolphin_radial_wave.py`
