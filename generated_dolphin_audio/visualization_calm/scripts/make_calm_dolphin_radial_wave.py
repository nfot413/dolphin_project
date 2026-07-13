from __future__ import annotations

import os
from pathlib import Path
import tempfile

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


AUDIO_PATH = (
    PROJECT_ROOT
    / "generated_dolphin_audio"
    / "audio"
    / "positive_active_dolphin_scene.wav"
)
OUT_DIR = PROJECT_ROOT / "generated_dolphin_audio" / "visualization_calm" / "figures"
OUT_PATH = OUT_DIR / "calm_dolphin_call_radial_wave.png"
N_POINTS = 1080


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


def calm_envelope(audio: np.ndarray) -> np.ndarray:
    env = np.abs(hilbert(audio))
    env = resample(env, N_POINTS)
    env = normalize01(np.maximum(env, 0.0))

    # Strong smoothing and dynamic range compression make the visual calm rather
    # than spiky while still preserving subtle call-like undulations.
    env = gaussian_filter1d(env, sigma=22.0, mode="wrap")
    env = normalize01(env) ** 0.82
    env = gaussian_filter1d(env, sigma=14.0, mode="wrap")

    theta = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
    gentle_swell = 0.08 * np.sin(6 * theta + 0.4) + 0.04 * np.sin(11 * theta + 1.8)
    env = 0.74 * normalize01(env) + gentle_swell
    return normalize01(gaussian_filter1d(env, sigma=10.0, mode="wrap"))


def draw_glow_line(ax: plt.Axes, theta: np.ndarray, radius: np.ndarray, color: str, scale: float = 1.0) -> None:
    closed_theta = np.r_[theta, theta[0]]
    closed_radius = np.r_[radius, radius[0]]
    for linewidth, alpha in [(18, 0.030), (11, 0.055), (6, 0.115), (3, 0.220)]:
        ax.plot(
            closed_theta,
            closed_radius,
            color=color,
            linewidth=linewidth * scale,
            alpha=alpha,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    ax.plot(
        closed_theta,
        closed_radius,
        color=color,
        linewidth=1.25 * scale,
        alpha=0.94,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def make_figure() -> None:
    if not AUDIO_PATH.exists():
        print(f"[ERROR] Missing audio file: {AUDIO_PATH}")
        print("Please run: python generated_dolphin_audio/scripts/generate_state_audio.py")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = calm_envelope(read_audio_mono(AUDIO_PATH))
    theta = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)

    fig = plt.figure(figsize=(9, 9), dpi=240)
    ax = fig.add_subplot(111, projection="polar")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_axis_off()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1.62)

    main = "#7BEFFF"
    accent = "#00B8D9"
    soft_green = "#98FFD8"
    deep_blue = "#1B5CFF"

    # Soft ocean-tech substrate: quiet rings, sparse dots, and restrained radial ticks.
    for radius, color, alpha, lw in [
        (0.43, accent, 0.12, 0.8),
        (0.58, main, 0.10, 0.6),
        (0.74, soft_green, 0.08, 0.55),
        (0.94, accent, 0.10, 0.7),
        (1.12, main, 0.08, 0.55),
        (1.31, deep_blue, 0.07, 0.45),
        (1.45, main, 0.08, 0.45),
    ]:
        ax.plot(theta, np.full_like(theta, radius), color=color, linewidth=lw, alpha=alpha)

    for i in range(0, N_POINTS, 6):
        local = env[i]
        base = 1.18
        tip = base + 0.035 + 0.060 * local
        ax.plot(
            [theta[i], theta[i]],
            [base, tip],
            color=main,
            linewidth=0.35 + 0.35 * local,
            alpha=0.10 + 0.16 * local,
            solid_capstyle="round",
        )

    dot_theta = theta[::9]
    dot_env = env[::9]
    ax.scatter(
        dot_theta,
        0.70 + 0.018 * np.sin(np.arange(dot_theta.size) * 0.4),
        s=2.5 + 5.0 * dot_env,
        color=soft_green,
        alpha=0.18,
        linewidths=0,
    )
    ax.scatter(
        dot_theta,
        1.38 + 0.020 * dot_env,
        s=2.0 + 4.0 * dot_env,
        color=main,
        alpha=0.14,
        linewidths=0,
    )

    base_radius = 0.95
    calm_wave = base_radius + 0.115 * env
    inner_wave = 0.79 + 0.050 * gaussian_filter1d(env, sigma=18.0, mode="wrap")
    outer_wave = 1.08 + 0.060 * gaussian_filter1d(np.roll(env, 70), sigma=16.0, mode="wrap")

    draw_glow_line(ax, theta, inner_wave, accent, scale=0.58)
    draw_glow_line(ax, theta, calm_wave, main, scale=0.92)
    draw_glow_line(ax, theta, outer_wave, soft_green, scale=0.46)

    # Very limited arc accents: enough to match the established style, not enough to feel agitated.
    for start, length, radius, color in [
        (0.30, 0.34, 1.25, soft_green),
        (1.42, 0.26, 1.02, deep_blue),
        (2.65, 0.40, 1.34, main),
        (4.18, 0.30, 0.88, accent),
        (5.35, 0.36, 1.18, soft_green),
    ]:
        arc_theta = np.linspace(start, start + length, 90)
        ax.plot(arc_theta, np.full_like(arc_theta, radius), color=color, linewidth=5.0, alpha=0.035)
        ax.plot(arc_theta, np.full_like(arc_theta, radius), color=color, linewidth=1.1, alpha=0.38)

    # Center remains empty and dark.
    ax.fill_between(theta, 0, 0.38, color="black", alpha=1.0, zorder=20)
    ax.plot(theta, np.full_like(theta, 0.40), color=main, linewidth=0.9, alpha=0.18, zorder=21)
    ax.plot(theta, np.full_like(theta, 0.50), color=accent, linewidth=0.5, alpha=0.08, zorder=21)

    ax.set_title("Calm dolphin call acoustic impression", color="white", fontsize=16, pad=24)
    ax.text(
        0.058,
        0.058,
        "CALM DOLPHIN CALL",
        transform=ax.transAxes,
        color=main,
        fontsize=9,
        fontweight="bold",
        alpha=0.78,
        ha="left",
        va="bottom",
    )

    fig.savefig(OUT_PATH, facecolor="black")
    plt.close(fig)
    print("[OK] Created calm dolphin radial waveform visualization.")
    print("[OK] Figure saved to: generated_dolphin_audio/visualization_calm/figures/calm_dolphin_call_radial_wave.png")


if __name__ == "__main__":
    make_figure()
