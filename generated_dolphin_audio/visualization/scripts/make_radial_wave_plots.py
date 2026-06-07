from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import zlib

import numpy as np
from scipy.io import wavfile
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
OUT_FIG_DIR = PROJECT_ROOT / "generated_dolphin_audio" / "visualization" / "figures"
OUT_README = (
    PROJECT_ROOT
    / "generated_dolphin_audio"
    / "visualization"
    / "reports"
    / "README_radial_wave_visualization.md"
)

N_POINTS = 720


@dataclass(frozen=True)
class StateConfig:
    key: str
    chinese_name: str
    title: str
    wav_name: str
    color: str
    wave_height: float
    smooth_window: int
    density_cycles: int
    irregularity: float
    spike_strength: float
    base_radius: float


STATES = [
    StateConfig(
        key="social_positive",
        chinese_name="社交积极型",
        title="Social-positive acoustic state",
        wav_name="positive_active_dolphin_scene.wav",
        color="#00E5FF",
        wave_height=0.20,
        smooth_window=35,
        density_cycles=8,
        irregularity=0.025,
        spike_strength=0.10,
        base_radius=1.00,
    ),
    StateConfig(
        key="foraging_active",
        chinese_name="觅食活跃型",
        title="Foraging-active acoustic state",
        wav_name="positive_active_dolphin_scene.wav",
        color="#39FF88",
        wave_height=0.27,
        smooth_window=19,
        density_cycles=24,
        irregularity=0.045,
        spike_strength=0.20,
        base_radius=1.00,
    ),
    StateConfig(
        key="stress_avoidance",
        chinese_name="压力/躲避型",
        title="Stress / avoidance acoustic state",
        wav_name="negative_aroused_dolphin_scene.wav",
        color="#B366FF",
        wave_height=0.34,
        smooth_window=11,
        density_cycles=18,
        irregularity=0.090,
        spike_strength=0.33,
        base_radius=1.00,
    ),
    StateConfig(
        key="conflict_like",
        chinese_name="冲突/打斗样高强度型",
        title="Conflict-like high-intensity acoustic state",
        wav_name="negative_aroused_dolphin_scene.wav",
        color="#FF5A36",
        wave_height=0.43,
        smooth_window=5,
        density_cycles=34,
        irregularity=0.125,
        spike_strength=0.55,
        base_radius=1.00,
    ),
]


def load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(path)

    if x.ndim > 1:
        x = x.mean(axis=1)

    if np.issubdtype(x.dtype, np.integer):
        dtype_info = np.iinfo(x.dtype)
        max_abs = max(abs(dtype_info.min), dtype_info.max)
        x = x.astype(np.float32) / max_abs
    else:
        x = x.astype(np.float32)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 1.0:
            x = x / peak

    return sr, x


def circular_smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1

    kernel = np.hanning(window)
    kernel = kernel / kernel.sum()
    pad = window // 2
    padded = np.pad(x, pad_width=pad, mode="wrap")
    return np.convolve(padded, kernel, mode="valid")


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.min(x)
    peak = np.max(x)
    if peak < 1e-12:
        return np.zeros_like(x)
    return x / peak


def extract_envelope(path: Path) -> np.ndarray:
    _, x = load_wav_mono(path)
    if x.size == 0:
        return np.zeros(N_POINTS, dtype=np.float64)

    envelope = np.abs(hilbert(x))
    envelope = resample(envelope, N_POINTS)
    envelope = np.maximum(envelope, 0.0)
    envelope = normalize01(envelope)
    envelope = envelope**0.55
    return normalize01(envelope)


def state_waveform(raw_envelope: np.ndarray, cfg: StateConfig) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(cfg.key.encode("utf-8")))
    theta = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)

    env = circular_smooth(raw_envelope, cfg.smooth_window)

    density_modulation = (
        0.055 * np.sin(cfg.density_cycles * theta)
        + 0.030 * np.sin((cfg.density_cycles * 2 + 3) * theta + 0.8)
    )
    irregular = cfg.irregularity * rng.normal(0.0, 1.0, N_POINTS)

    spike_source = np.maximum(raw_envelope - circular_smooth(raw_envelope, 55), 0.0)
    spikes = cfg.spike_strength * normalize01(spike_source) ** 1.8

    wave = env + density_modulation + irregular + spikes
    wave = circular_smooth(wave, max(3, cfg.smooth_window // 2))
    return normalize01(wave)


def setup_polar_axis(fig: plt.Figure) -> plt.Axes:
    ax = fig.add_subplot(111, projection="polar")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_axis_off()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1.62)
    return ax


def draw_radial_texture(
    ax: plt.Axes,
    theta: np.ndarray,
    color: str,
    inner_radius: float = 0.64,
    outer_radius: float = 1.42,
) -> None:
    for angle in theta[::3]:
        ax.plot(
            [angle, angle],
            [inner_radius, outer_radius],
            color=color,
            alpha=0.055,
            linewidth=0.35,
            solid_capstyle="round",
        )

    dot_theta = theta[::10]
    for radius, alpha, size in [(0.69, 0.28, 3.2), (0.79, 0.18, 2.2)]:
        ax.scatter(
            dot_theta,
            np.full_like(dot_theta, radius),
            s=size,
            color=color,
            alpha=alpha,
            linewidths=0,
        )


def draw_neon_wave(
    ax: plt.Axes,
    theta: np.ndarray,
    radius: np.ndarray,
    color: str,
    linewidth_scale: float = 1.0,
) -> None:
    closed_theta = np.r_[theta, theta[0]]
    closed_radius = np.r_[radius, radius[0]]

    for linewidth, alpha in [
        (18.0 * linewidth_scale, 0.035),
        (11.0 * linewidth_scale, 0.060),
        (6.5 * linewidth_scale, 0.120),
        (3.0 * linewidth_scale, 0.250),
    ]:
        ax.plot(
            closed_theta,
            closed_radius,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_joinstyle="round",
            solid_capstyle="round",
        )

    ax.plot(
        closed_theta,
        closed_radius,
        color=color,
        linewidth=1.35 * linewidth_scale,
        alpha=0.98,
        solid_joinstyle="round",
        solid_capstyle="round",
    )
    ax.plot(
        closed_theta,
        closed_radius - 0.030 * linewidth_scale,
        color=color,
        linewidth=0.65 * linewidth_scale,
        alpha=0.45,
        solid_joinstyle="round",
        solid_capstyle="round",
    )


def draw_individual_plot(cfg: StateConfig, raw_envelope: np.ndarray) -> None:
    theta = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
    wave = state_waveform(raw_envelope, cfg)
    radius = cfg.base_radius + cfg.wave_height * wave

    fig = plt.figure(figsize=(8, 8), dpi=220)
    ax = setup_polar_axis(fig)

    draw_radial_texture(ax, theta, cfg.color)

    # Inner dark disc keeps the center open even when glow layers overlap.
    ax.fill_between(theta, 0, 0.58, color="black", alpha=1.0, zorder=3)
    ax.plot(theta, np.full_like(theta, 0.62), color=cfg.color, alpha=0.16, linewidth=0.8)
    ax.plot(theta, np.full_like(theta, 0.86), color=cfg.color, alpha=0.10, linewidth=0.6)

    draw_neon_wave(ax, theta, radius, cfg.color)

    ax.set_title(cfg.title, color="white", fontsize=15, pad=24)
    out_path = OUT_FIG_DIR / f"{cfg.key}_radial_wave.png"
    fig.savefig(out_path, facecolor="black")
    plt.close(fig)


def draw_overview(envelopes: dict[str, np.ndarray]) -> None:
    theta = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)

    fig = plt.figure(figsize=(8, 8), dpi=220)
    ax = setup_polar_axis(fig)
    ax.set_ylim(0, 1.55)

    overview_radii = {
        "social_positive": 0.46,
        "foraging_active": 0.72,
        "stress_avoidance": 0.99,
        "conflict_like": 1.27,
    }

    for cfg in STATES:
        wave = state_waveform(envelopes[cfg.wav_name], cfg)
        height = cfg.wave_height * 0.42
        radius = overview_radii[cfg.key] + height * wave

        for angle in theta[::6]:
            ax.plot(
                [angle, angle],
                [overview_radii[cfg.key] - 0.035, overview_radii[cfg.key] + height + 0.035],
                color=cfg.color,
                alpha=0.030,
                linewidth=0.28,
            )
        draw_neon_wave(ax, theta, radius, cfg.color, linewidth_scale=0.72)

    ax.fill_between(theta, 0, 0.29, color="black", alpha=1.0, zorder=5)
    ax.set_title(
        "Four candidate dolphin acoustic states",
        color="white",
        fontsize=15,
        pad=24,
    )

    out_path = OUT_FIG_DIR / "four_state_radial_wave_overview.png"
    fig.savefig(out_path, facecolor="black")
    plt.close(fig)


def main() -> int:
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    required_audio = sorted({AUDIO_DIR / cfg.wav_name for cfg in STATES})
    missing_audio = [path for path in required_audio if not path.exists()]
    if missing_audio:
        print("[ERROR] Missing required audio file(s):")
        for path in missing_audio:
            print(f"  - {path}")
        print("Please run the audio generation script first, then rerun this visualization script.")
        return 1

    envelopes = {path.name: extract_envelope(path) for path in required_audio}

    for cfg in STATES:
        draw_individual_plot(cfg, envelopes[cfg.wav_name])
    draw_overview(envelopes)

    print("[OK] Created radial waveform visualizations.")
    print("[OK] Figures saved to: generated_dolphin_audio/visualization/figures")
    print(
        "[OK] README saved to: "
        "generated_dolphin_audio/visualization/reports/README_radial_wave_visualization.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
