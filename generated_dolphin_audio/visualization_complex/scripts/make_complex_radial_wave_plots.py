from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import zlib

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, resample


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "dolphin_project_matplotlib_cache"
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "dolphin_project_xdg_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib.pyplot as plt


AUDIO_DIR = PROJECT_ROOT / "generated_dolphin_audio" / "audio"
OUT_FIG_DIR = PROJECT_ROOT / "generated_dolphin_audio" / "visualization_complex" / "figures"
OUT_REPORT_DIR = PROJECT_ROOT / "generated_dolphin_audio" / "visualization_complex" / "reports"
README_PATH = OUT_REPORT_DIR / "README_complex_radial_visualization.md"

POINTS = 1080


@dataclass(frozen=True)
class StateConfig:
    key: str
    chinese_name: str
    title: str
    label: str
    wav_name: str
    main_color: str
    accent_color: str
    secondary_color: str
    smooth_sigma: float
    wave_height: float
    second_wave_height: float
    radial_scale: float
    particle_count: int
    arc_count: int
    burst_count: int
    breakiness: float
    irregularity: float
    spike_gain: float
    output_name: str


STATES: dict[str, StateConfig] = {
    "social_positive": StateConfig(
        key="social_positive",
        chinese_name="社交积极型",
        title="Social-positive acoustic state",
        label="SOCIAL-POSITIVE",
        wav_name="positive_active_dolphin_scene.wav",
        main_color="#00E5FF",
        accent_color="#2D7DFF",
        secondary_color="#78FFD6",
        smooth_sigma=8.0,
        wave_height=0.18,
        second_wave_height=0.11,
        radial_scale=0.13,
        particle_count=230,
        arc_count=18,
        burst_count=5,
        breakiness=0.12,
        irregularity=0.015,
        spike_gain=0.16,
        output_name="social_positive_complex_radial_wave.png",
    ),
    "foraging_active": StateConfig(
        key="foraging_active",
        chinese_name="觅食活跃型",
        title="Foraging-active acoustic state",
        label="FORAGING-ACTIVE",
        wav_name="positive_active_dolphin_scene.wav",
        main_color="#39FF88",
        accent_color="#00E5FF",
        secondary_color="#D8FF4D",
        smooth_sigma=4.0,
        wave_height=0.25,
        second_wave_height=0.16,
        radial_scale=0.22,
        particle_count=420,
        arc_count=30,
        burst_count=12,
        breakiness=0.22,
        irregularity=0.040,
        spike_gain=0.34,
        output_name="foraging_active_complex_radial_wave.png",
    ),
    "stress_avoidance": StateConfig(
        key="stress_avoidance",
        chinese_name="压力/躲避型",
        title="Stress / avoidance acoustic state",
        label="STRESS-AVOIDANCE",
        wav_name="negative_aroused_dolphin_scene.wav",
        main_color="#B366FF",
        accent_color="#5E66FF",
        secondary_color="#FF7AE6",
        smooth_sigma=2.8,
        wave_height=0.28,
        second_wave_height=0.18,
        radial_scale=0.24,
        particle_count=330,
        arc_count=38,
        burst_count=9,
        breakiness=0.55,
        irregularity=0.095,
        spike_gain=0.42,
        output_name="stress_avoidance_complex_radial_wave.png",
    ),
    "conflict_like": StateConfig(
        key="conflict_like",
        chinese_name="冲突/打斗样高强度型",
        title="Conflict-like high-intensity acoustic state",
        label="CONFLICT-LIKE",
        wav_name="negative_aroused_dolphin_scene.wav",
        main_color="#FF5A36",
        accent_color="#FFE84D",
        secondary_color="#B3132B",
        smooth_sigma=1.7,
        wave_height=0.36,
        second_wave_height=0.24,
        radial_scale=0.34,
        particle_count=390,
        arc_count=42,
        burst_count=16,
        breakiness=0.34,
        irregularity=0.130,
        spike_gain=0.72,
        output_name="conflict_like_complex_radial_wave.png",
    ),
}


def read_audio_mono(path: Path) -> np.ndarray:
    _, x = wavfile.read(path)

    if x.ndim > 1:
        x = x.mean(axis=1)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32) / max(abs(info.min), info.max)
    else:
        x = x.astype(np.float32)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 1.0:
            x = x / peak

    return x


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.min(x)
    peak = np.max(x)
    if peak < 1e-12:
        return np.zeros_like(x)
    return x / peak


def get_audio_envelope(x: np.ndarray, points: int = POINTS) -> np.ndarray:
    if x.size == 0:
        return np.zeros(points, dtype=np.float64)

    env = np.abs(hilbert(x))
    env = resample(env, points)
    env = np.maximum(env, 0.0)
    env = normalize01(env)
    env = gaussian_filter1d(env, sigma=2.0, mode="wrap")
    env = normalize01(env) ** 0.50
    return normalize01(env)


def transform_envelope_for_state(
    env: np.ndarray,
    state_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    cfg = STATES[state_name]
    theta = np.linspace(0, 2 * np.pi, env.size, endpoint=False)
    smooth = gaussian_filter1d(env, sigma=cfg.smooth_sigma, mode="wrap")
    fine = np.maximum(env - gaussian_filter1d(env, sigma=18.0, mode="wrap"), 0.0)
    fine = normalize01(fine)

    modulation = (
        0.035 * np.sin((7 + cfg.arc_count // 4) * theta + 0.5)
        + 0.022 * np.sin((19 + cfg.burst_count) * theta + 1.7)
    )
    irregular = cfg.irregularity * rng.normal(0.0, 1.0, env.size)
    spikes = cfg.spike_gain * fine**1.9

    if state_name == "social_positive":
        transformed = 0.90 * smooth + 0.10 * env + 0.45 * modulation
        transformed = gaussian_filter1d(transformed, sigma=7.0, mode="wrap")
    elif state_name == "foraging_active":
        local_bursts = fine * (0.55 + 0.45 * np.sin(31 * theta) ** 2)
        transformed = 0.68 * smooth + 0.20 * env + spikes + local_bursts + modulation
        transformed = gaussian_filter1d(transformed, sigma=2.4, mode="wrap")
    elif state_name == "stress_avoidance":
        fracture = rng.uniform(0.45, 1.25, env.size)
        fracture = gaussian_filter1d(fracture, sigma=10.0, mode="wrap")
        transformed = 0.55 * smooth * fracture + 0.20 * env + spikes + irregular + modulation
        transformed = gaussian_filter1d(transformed, sigma=1.8, mode="wrap")
    else:
        pulse = np.maximum(env - np.percentile(env, 64), 0.0)
        pulse = normalize01(pulse) ** 1.25
        transformed = 0.48 * smooth + 0.22 * env + 1.35 * spikes + 0.85 * pulse + irregular
        transformed = gaussian_filter1d(transformed, sigma=1.1, mode="wrap")

    return normalize01(transformed) ** 0.45


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    color = hex_color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def interpolate_color(color1: str, color2: str, ratio: float) -> tuple[float, float, float]:
    a = np.array(hex_to_rgb(color1))
    b = np.array(hex_to_rgb(color2))
    return tuple((1.0 - ratio) * a + ratio * b)


def closed(theta: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.r_[theta, theta[0]], np.r_[radius, radius[0]]


def setup_complex_axis(ax: plt.Axes, limit: float = 1.72) -> None:
    ax.set_facecolor("black")
    ax.figure.set_facecolor("black")
    ax.set_axis_off()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, limit)


def draw_glow_line(
    ax: plt.Axes,
    theta: np.ndarray,
    radius: np.ndarray,
    color: str | tuple[float, float, float],
    scale: float = 1.0,
    zorder: int = 5,
) -> None:
    t, r = closed(theta, radius)
    for lw, alpha in [(18, 0.035), (11, 0.065), (6, 0.120), (3, 0.260)]:
        ax.plot(t, r, color=color, linewidth=lw * scale, alpha=alpha, zorder=zorder)
    ax.plot(t, r, color=color, linewidth=1.35 * scale, alpha=0.98, zorder=zorder + 1)


def draw_arc_segment(
    ax: plt.Axes,
    start: float,
    length: float,
    radius: float,
    color: str | tuple[float, float, float],
    linewidth: float,
    alpha: float,
    zorder: int,
    points: int = 90,
) -> None:
    arc_theta = np.linspace(start, start + length, points)
    arc_radius = np.full_like(arc_theta, radius)
    for glow_lw, glow_alpha in [(linewidth * 4.2, alpha * 0.10), (linewidth * 2.2, alpha * 0.18)]:
        ax.plot(arc_theta, arc_radius, color=color, linewidth=glow_lw, alpha=glow_alpha, zorder=zorder)
    ax.plot(arc_theta, arc_radius, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder + 1)


def draw_complex_radial_state(
    ax: plt.Axes,
    env: np.ndarray,
    state_config: StateConfig,
    rng: np.random.Generator,
    title: bool = True,
    label: bool = True,
) -> None:
    setup_complex_axis(ax)
    theta = np.linspace(0, 2 * np.pi, env.size, endpoint=False)
    state_env = transform_envelope_for_state(env, state_config.key, rng)
    alt_env = normalize01(np.roll(state_env, env.size // 9) * 0.62 + env * 0.38)

    main = state_config.main_color
    accent = state_config.accent_color
    secondary = state_config.secondary_color

    # Low-alpha rings create a radar-like acoustic substrate.
    ring_radii = np.array([0.39, 0.52, 0.66, 0.81, 0.97, 1.14, 1.32, 1.49])
    for i, radius in enumerate(ring_radii):
        color = interpolate_color(main, secondary if i % 2 else accent, i / (len(ring_radii) - 1))
        lw = 0.45 + 0.28 * (i % 3)
        alpha = 0.08 + 0.035 * (i % 2)
        ax.plot(theta, np.full_like(theta, radius), color=color, linewidth=lw, alpha=alpha, zorder=1)

    # Fine dotted rings, with denser outer texture for active states.
    dot_step = 5 if state_config.key in {"foraging_active", "conflict_like"} else 8
    dot_theta = theta[::dot_step]
    for radius, size, alpha in [(1.24, 2.2, 0.20), (1.38, 1.8, 0.18), (1.53, 2.5, 0.16)]:
        jitter = rng.normal(0, 0.004 + 0.010 * state_config.breakiness, dot_theta.size)
        ax.scatter(
            dot_theta,
            radius + jitter,
            s=size,
            color=interpolate_color(main, accent, 0.35),
            alpha=alpha,
            linewidths=0,
            zorder=2,
        )

    # Envelope-controlled radial short lines form the outer sonic bristles.
    radial_step = 3 if state_config.key in {"foraging_active", "conflict_like"} else 5
    for i in range(0, env.size, radial_step):
        angle = theta[i]
        local = state_env[i]
        base = 1.18 + 0.030 * np.sin(i * 0.07)
        tip = base + 0.06 + state_config.radial_scale * local
        if state_config.key == "conflict_like" and local > 0.72:
            tip += 0.10 * local
        color = interpolate_color(main, secondary, min(1.0, local))
        ax.plot(
            [angle, angle],
            [base, tip],
            color=color,
            linewidth=0.34 + 0.95 * local,
            alpha=0.18 + 0.42 * local,
            solid_capstyle="round",
            zorder=3,
        )

    # Two independently modulated audio rings satisfy the radial waveform layers.
    main_radius = 0.85 + state_config.wave_height * state_env
    second_radius = 1.04 + state_config.second_wave_height * alt_env
    inner_wave = 0.63 + 0.055 * gaussian_filter1d(state_env, sigma=10.0, mode="wrap")
    draw_glow_line(ax, theta, inner_wave, interpolate_color(main, accent, 0.30), scale=0.55, zorder=4)
    draw_glow_line(ax, theta, main_radius, main, scale=1.05, zorder=6)
    draw_glow_line(ax, theta, second_radius, interpolate_color(main, secondary, 0.55), scale=0.70, zorder=5)

    # Discontinuous arcs make the poster feel layered and state-specific.
    for arc_idx in range(state_config.arc_count):
        radius = rng.uniform(0.54, 1.50)
        start = rng.uniform(0, 2 * np.pi)
        if state_config.key == "social_positive":
            length = rng.uniform(0.18, 0.52)
        elif state_config.key == "stress_avoidance":
            length = rng.uniform(0.035, 0.22)
        else:
            length = rng.uniform(0.06, 0.42)
        color = [main, accent, secondary][arc_idx % 3]
        linewidth = rng.uniform(0.55, 2.4 if state_config.key != "conflict_like" else 3.2)
        alpha = rng.uniform(0.18, 0.52)
        draw_arc_segment(ax, start, length, radius, color, linewidth, alpha, zorder=7)

        if state_config.key == "stress_avoidance" and rng.random() < 0.50:
            offset = length * rng.uniform(1.25, 1.75)
            draw_arc_segment(
                ax,
                start + offset,
                length * rng.uniform(0.25, 0.60),
                radius + rng.normal(0, 0.025),
                secondary,
                linewidth * 0.75,
                alpha * 0.80,
                zorder=7,
                points=40,
            )

    # Glowing particles carry local energy information from the envelope.
    particle_idx = rng.choice(env.size, size=state_config.particle_count, replace=False)
    particle_theta = theta[particle_idx] + rng.normal(0, 0.004 + 0.014 * state_config.breakiness, particle_idx.size)
    particle_env = state_env[particle_idx]
    particle_radius = (
        0.79
        + 0.62 * particle_env
        + rng.normal(0, 0.035 + 0.060 * state_config.breakiness, particle_idx.size)
    )
    particle_size = 5.0 + 46.0 * particle_env**2
    if state_config.key == "conflict_like":
        particle_size *= 1.28
    elif state_config.key == "social_positive":
        particle_size *= 0.70

    particle_color = [interpolate_color(main, accent, float(v)) for v in particle_env]
    ax.scatter(
        particle_theta,
        particle_radius,
        s=particle_size * 4.6,
        color=particle_color,
        alpha=0.050,
        linewidths=0,
        zorder=8,
    )
    ax.scatter(
        particle_theta,
        particle_radius,
        s=particle_size,
        color=particle_color,
        alpha=0.38,
        linewidths=0,
        zorder=9,
    )

    # Local peaks become outer burst points and strong pulse accents.
    peak_candidates = np.argsort(state_env)[-state_config.burst_count * 5 :]
    rng.shuffle(peak_candidates)
    chosen: list[int] = []
    min_distance = env.size // max(12, state_config.burst_count * 2)
    for idx in peak_candidates:
        if all(min(abs(idx - j), env.size - abs(idx - j)) > min_distance for j in chosen):
            chosen.append(int(idx))
        if len(chosen) >= state_config.burst_count:
            break

    for idx in chosen:
        local = state_env[idx]
        angle = theta[idx] + rng.normal(0, 0.010)
        radius = 1.25 + 0.31 * local + rng.uniform(-0.015, 0.045)
        burst_color = secondary if state_config.key != "conflict_like" else accent
        size = (70 + 170 * local) * (1.35 if state_config.key == "conflict_like" else 1.0)
        ax.scatter([angle], [radius], s=size * 3.0, color=burst_color, alpha=0.075, linewidths=0, zorder=10)
        ax.scatter([angle], [radius], s=size, color=burst_color, alpha=0.62, linewidths=0, zorder=11)
        ax.plot(
            [angle, angle],
            [1.07, min(1.70, radius + 0.12 + 0.10 * local)],
            color=burst_color,
            linewidth=1.0 + 2.1 * local,
            alpha=0.34,
            solid_capstyle="round",
            zorder=10,
        )

    # A black core preserves the hollow center while a faint rim keeps depth.
    ax.fill_between(theta, 0, 0.34, color="black", alpha=1.0, zorder=20)
    ax.plot(theta, np.full_like(theta, 0.36), color=main, linewidth=1.0, alpha=0.20, zorder=21)
    ax.plot(theta, np.full_like(theta, 0.44), color=accent, linewidth=0.55, alpha=0.10, zorder=21)

    if title:
        ax.set_title(state_config.title, color="white", fontsize=15, pad=24)
    if label:
        ax.text(
            0.055,
            0.055,
            state_config.label,
            transform=ax.transAxes,
            color=interpolate_color(main, "#FFFFFF", 0.30),
            fontsize=8.5,
            fontweight="bold",
            alpha=0.78,
            ha="left",
            va="bottom",
        )


def make_single_state_figure(state_name: str, config: StateConfig, envelopes: dict[str, np.ndarray]) -> None:
    rng = np.random.default_rng(zlib.crc32(f"single-{state_name}".encode("utf-8")))
    fig = plt.figure(figsize=(9, 9), dpi=240)
    ax = fig.add_subplot(111, projection="polar")
    draw_complex_radial_state(ax, envelopes[config.wav_name], config, rng, title=True, label=True)
    fig.savefig(OUT_FIG_DIR / config.output_name, facecolor="black")
    plt.close(fig)


def make_overview_figure(envelopes: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 14),
        dpi=220,
        subplot_kw={"projection": "polar"},
    )
    fig.patch.set_facecolor("black")
    for ax, (state_name, cfg) in zip(axes.ravel(), STATES.items(), strict=True):
        rng = np.random.default_rng(zlib.crc32(f"overview-{state_name}".encode("utf-8")))
        draw_complex_radial_state(ax, envelopes[cfg.wav_name], cfg, rng, title=True, label=True)
        ax.set_ylim(0, 1.74)

    fig.suptitle(
        "Four candidate dolphin acoustic states - complex radial visualization",
        color="white",
        fontsize=20,
        y=0.975,
    )
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.035, top=0.880, wspace=0.045, hspace=0.160)
    fig.savefig(OUT_FIG_DIR / "four_state_complex_radial_overview.png", facecolor="black")
    plt.close(fig)


def write_readme() -> None:
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    README_PATH.write_text(
        """# 复杂版海豚候选声学状态环形声波可视化说明

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
""",
        encoding="utf-8",
    )


def main() -> int:
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    required_audio = sorted({AUDIO_DIR / cfg.wav_name for cfg in STATES.values()})
    missing = [path for path in required_audio if not path.exists()]
    if missing:
        print("[ERROR] Missing required audio file(s):")
        for path in missing:
            print(f"  - {path}")
        print("Please run: python generated_dolphin_audio/scripts/generate_state_audio.py")
        return 1

    envelopes = {
        path.name: get_audio_envelope(read_audio_mono(path), points=POINTS)
        for path in required_audio
    }

    for state_name, cfg in STATES.items():
        make_single_state_figure(state_name, cfg, envelopes)
    make_overview_figure(envelopes)
    write_readme()

    print("[OK] Created complex radial waveform visualizations.")
    print("[OK] Figures saved to: generated_dolphin_audio/visualization_complex/figures")
    print(
        "[OK] README saved to: "
        "generated_dolphin_audio/visualization_complex/reports/README_complex_radial_visualization.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
