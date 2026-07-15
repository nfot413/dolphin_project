#!/usr/bin/env python3
"""Build a high-resolution overview of the dolphin acoustic analysis pipeline."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path(tempfile.gettempdir()) / "dolphin_project_pipeline_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
from scipy.io import wavfile
from scipy.signal import spectrogram


OUTPUT_DIR = PROJECT_ROOT / "project_visualization" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "dolphin_acoustic_state_pipeline_overview.png"

WHISTLE_FEATURES = (
    PROJECT_ROOT
    / "official_whistle_features"
    / "features"
    / "official_whistle_features_with_clusters.csv"
)
CLICK_FEATURES = (
    PROJECT_ROOT
    / "official_click_features"
    / "features"
    / "official_click_features_with_clusters.csv"
)
WHISTLE_SUMMARY = (
    PROJECT_ROOT / "official_whistle_features" / "features" / "cluster_feature_summary.csv"
)
CLICK_SUMMARY = (
    PROJECT_ROOT / "official_click_features" / "features" / "click_cluster_feature_summary.csv"
)
STATE_WINDOWS = (
    PROJECT_ROOT / "behavior_state_analysis" / "features" / "behavior_state_candidates.csv"
)

AUDIO_DIR = PROJECT_ROOT / "generated_dolphin_audio" / "audio"
POSITIVE_AUDIO = AUDIO_DIR / "positive_active_dolphin_scene.wav"
NEGATIVE_AUDIO = AUDIO_DIR / "negative_aroused_dolphin_scene.wav"

RADIAL_DIR = PROJECT_ROOT / "generated_dolphin_audio" / "visualization_complex" / "figures"
CALM_DIR = PROJECT_ROOT / "generated_dolphin_audio" / "visualization_calm" / "figures"
RADIAL_IMAGES = [
    ("CALM", CALM_DIR / "calm_dolphin_call_radial_wave.png", "#58D7E8"),
    ("SOCIAL", RADIAL_DIR / "social_positive_complex_radial_wave.png", "#00E5FF"),
    ("FORAGING", RADIAL_DIR / "foraging_active_complex_radial_wave.png", "#39FF88"),
    ("STRESS", RADIAL_DIR / "stress_avoidance_complex_radial_wave.png", "#B366FF"),
    ("CONFLICT", RADIAL_DIR / "conflict_like_complex_radial_wave.png", "#FF5A36"),
]

BG = "#080B0E"
PANEL = "#0C1115"
GRID = "#26313A"
TEXT = "#E8EEF1"
MUTED = "#89969E"
CYAN = "#00D9F5"
GREEN = "#55E987"
ORANGE = "#FF7A43"
PURPLE = "#B477FF"

STATE_COLORS = {
    "Foraging-like acoustic activity": "#39FF88",
    "Social communication candidate": "#00E5FF",
    "Uncertain mixed acoustic state": "#65747D",
    "Conflict-like high-intensity activity": "#FF5A36",
    "Low-activity / no annotated acoustic event": "#334049",
    "Avoidance / stress-like arousal": "#B366FF",
    "Courtship / affiliative candidate": "#F1D85C",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def validate_inputs() -> None:
    required = [
        WHISTLE_FEATURES,
        CLICK_FEATURES,
        WHISTLE_SUMMARY,
        CLICK_SUMMARY,
        STATE_WINDOWS,
        POSITIVE_AUDIO,
        NEGATIVE_AUDIO,
        *[path for _, path, _ in RADIAL_IMAGES],
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required project outputs are missing:\n{paths}")


def style_axis(ax: plt.Axes, border: bool = True) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(border)
        spine.set_color(GRID)
        spine.set_linewidth(0.8)


def stage_label(fig: plt.Figure, x: float, y: float, number: str, label: str) -> None:
    fig.text(x, y, number, color=CYAN, fontsize=11, fontweight="bold", va="center")
    fig.text(x + 0.017, y, label, color=TEXT, fontsize=12, fontweight="bold", va="center")
    fig.add_artist(
        Rectangle(
            (x, y - 0.012),
            0.145,
            0.0012,
            transform=fig.transFigure,
            color=GRID,
            linewidth=0,
        )
    )


def flow_arrow(fig: plt.Figure, start: tuple[float, float], end: tuple[float, float]) -> None:
    fig.add_artist(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#3D5963",
            connectionstyle="arc3,rad=0",
        )
    )


def draw_background(fig: plt.Figure) -> None:
    for x in np.linspace(0.025, 0.975, 33):
        fig.add_artist(
            plt.Line2D([x, x], [0.045, 0.94], transform=fig.transFigure, color=GRID, alpha=0.13, lw=0.5)
        )
    for y in np.linspace(0.06, 0.93, 19):
        fig.add_artist(
            plt.Line2D([0.02, 0.98], [y, y], transform=fig.transFigure, color=GRID, alpha=0.13, lw=0.5)
        )


def draw_input_panel(fig: plt.Figure, rows: list[dict[str, str]]) -> None:
    ax = fig.add_axes([0.035, 0.595, 0.145, 0.235])
    style_axis(ax)
    minutes = numeric(rows, "window_mid_min")
    whistles = numeric(rows, "whistle_count")
    clicks = numeric(rows, "click_train_count")
    whistles = whistles / max(whistles.max(), 1)
    clicks = clicks / max(clicks.max(), 1)
    ax.fill_between(minutes, 0, clicks, color=GREEN, alpha=0.28, linewidth=0)
    ax.plot(minutes, clicks, color=GREEN, lw=1.0, alpha=0.95)
    ax.plot(minutes, whistles * 0.82, color=CYAN, lw=1.2, alpha=0.95)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 50, 100], labels=["0", "50", "100 min"])
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.55)
    ax.text(0.04, 0.88, "EVENT DENSITY", transform=ax.transAxes, color=MUTED, fontsize=8, fontweight="bold")
    ax.text(0.04, 0.72, "303", transform=ax.transAxes, color=CYAN, fontsize=25, fontweight="bold")
    ax.text(0.04, 0.61, "WHISTLES", transform=ax.transAxes, color=MUTED, fontsize=7)
    ax.text(0.52, 0.72, "3,323", transform=ax.transAxes, color=GREEN, fontsize=25, fontweight="bold")
    ax.text(0.52, 0.61, "CLICK TRAINS", transform=ax.transAxes, color=MUTED, fontsize=7)


def zscore_columns(data: np.ndarray) -> np.ndarray:
    means = np.nanmean(data, axis=0)
    scales = np.nanstd(data, axis=0)
    scales[scales < 1e-12] = 1.0
    return (data - means) / scales


def draw_feature_panel(
    fig: plt.Figure,
    rect: list[float],
    rows: list[dict[str, str]],
    features: list[str],
    title: str,
) -> None:
    ax = fig.add_axes(rect)
    style_axis(ax)
    matrix = np.asarray([[float(row[key]) for key in features] for row in rows])
    matrix = zscore_columns(matrix)
    cmap = LinearSegmentedColormap.from_list("acoustic", ["#142129", "#087B8D", "#5CFFAD"])
    ax.imshow(matrix, cmap=cmap, vmin=-1.45, vmax=1.45, aspect="auto", interpolation="nearest")
    short = ["DUR", "RMS", "FREQ", "BAND", "RATE"]
    ax.set_xticks(range(len(short)), labels=short, fontsize=6.5)
    ax.set_yticks([0, 1, 2], labels=["C0", "C1", "C2"], fontsize=7)
    ax.tick_params(pad=3)
    ax.text(0.03, 0.88, title, transform=ax.transAxes, color=TEXT, fontsize=8, fontweight="bold")


def draw_pca_panel(
    fig: plt.Figure,
    rect: list[float],
    rows: list[dict[str, str]],
    title: str,
    max_points: int,
) -> None:
    ax = fig.add_axes(rect)
    style_axis(ax)
    x = numeric(rows, "PC1")
    y = numeric(rows, "PC2")
    cluster = numeric(rows, "cluster").astype(int)
    if len(rows) > max_points:
        rng = np.random.default_rng(17)
        keep = np.sort(rng.choice(len(rows), max_points, replace=False))
        x, y, cluster = x[keep], y[keep], cluster[keep]
    colors = np.asarray([CYAN, GREEN, ORANGE])
    ax.scatter(x, y, c=colors[cluster], s=7, alpha=0.68, linewidths=0)
    ax.axhline(0, color=GRID, lw=0.6)
    ax.axvline(0, color=GRID, lw=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.04, 0.89, title, transform=ax.transAxes, color=TEXT, fontsize=8, fontweight="bold")
    ax.text(0.94, 0.08, "k=3", transform=ax.transAxes, color=MUTED, fontsize=8, ha="right")


def draw_state_panel(fig: plt.Figure, rows: list[dict[str, str]]) -> None:
    ax = fig.add_axes([0.555, 0.595, 0.205, 0.235])
    style_axis(ax)
    starts = numeric(rows, "window_start_sec") / 60.0
    labels = [row["final_behavior_candidate"] for row in rows]
    for start, label in zip(starts, labels):
        ax.add_patch(Rectangle((start, 0.10), 0.5, 0.22, color=STATE_COLORS.get(label, MUTED), linewidth=0))

    score_keys = ["foraging_score", "social_score", "avoidance_stress_score", "conflict_like_score"]
    score_colors = [GREEN, CYAN, PURPLE, ORANGE]
    for offset, (key, color) in enumerate(zip(score_keys, score_colors)):
        values = numeric(rows, key)
        values = values / max(values.max(), 1e-12)
        ax.plot(starts + 0.25, 0.48 + values * 0.38 + offset * 0.015, color=color, lw=0.75, alpha=0.72)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 25, 50, 75, 100], labels=["0", "25", "50", "75", "100 min"])
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.55)
    ax.text(0.03, 0.88, "30 s WINDOWS", transform=ax.transAxes, color=MUTED, fontsize=8, fontweight="bold")
    ax.text(0.97, 0.88, "197", transform=ax.transAxes, color=TEXT, fontsize=20, fontweight="bold", ha="right")


def load_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float32)
    return sample_rate, audio


def draw_spectrogram(fig: plt.Figure, rect: list[float], path: Path, title: str, accent: str) -> None:
    sample_rate, audio = load_audio(path)
    frequencies, times, power = spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=4096,
        noverlap=3072,
        scaling="spectrum",
        mode="magnitude",
    )
    keep = frequencies <= 70000
    db = 20 * np.log10(power[keep] + 1e-10)
    floor = np.percentile(db, 18)
    ceiling = np.percentile(db, 99.8)
    ax = fig.add_axes(rect)
    style_axis(ax)
    cmap = LinearSegmentedColormap.from_list(
        "spectral",
        ["#080B0E", "#10313B", "#087E8B", accent, "#F3F7D4"],
    )
    ax.pcolormesh(times, frequencies[keep] / 1000.0, db, shading="auto", cmap=cmap, vmin=floor, vmax=ceiling)
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(0, 70)
    ax.set_xticks([0, 10, 20], labels=["0", "10", "20 s"])
    ax.set_yticks([0, 35, 70], labels=["0", "35", "70 kHz"])
    ax.grid(color=GRID, lw=0.45, alpha=0.28)
    ax.text(0.03, 0.86, title, transform=ax.transAxes, color=TEXT, fontsize=8, fontweight="bold")


def crop_radial(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        top = int(height * 0.105)
        bottom = int(height * 0.965)
        side = min(width, bottom - top)
        left = (width - side) // 2
        cropped = np.asarray(image.crop((left, top, left + side, top + side))).copy()
        dark_pixels = np.max(cropped, axis=2) < 14
        cropped[dark_pixels] = np.asarray([8, 11, 14], dtype=np.uint8)
        return cropped


def draw_radial_outputs(fig: plt.Figure) -> None:
    centers = np.linspace(0.12, 0.88, 5)
    axis_width = 0.155
    axis_height = axis_width * (32 / 18)
    bottom = 0.095
    for center, (label, path, color) in zip(centers, RADIAL_IMAGES):
        ax = fig.add_axes([center - axis_width / 2, bottom, axis_width, axis_height])
        ax.imshow(crop_radial(path), interpolation="lanczos")
        ax.set_axis_off()
        ax.text(
            0.5,
            -0.03,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=color,
            fontsize=10,
            fontweight="bold",
        )
        fig.add_artist(
            plt.Line2D(
                [center - 0.035, center + 0.035],
                [bottom - 0.027, bottom - 0.027],
                transform=fig.transFigure,
                color=color,
                alpha=0.65,
                lw=1.2,
            )
        )


def build_figure() -> None:
    whistle_rows = read_csv(WHISTLE_FEATURES)
    click_rows = read_csv(CLICK_FEATURES)
    whistle_summary = read_csv(WHISTLE_SUMMARY)
    click_summary = read_csv(CLICK_SUMMARY)
    state_rows = read_csv(STATE_WINDOWS)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlepad": 8,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": TEXT,
        }
    )
    fig = plt.figure(figsize=(32, 18), dpi=240, facecolor=BG)
    draw_background(fig)

    fig.text(0.035, 0.947, "DOLPHIN ACOUSTIC STATE PIPELINE", fontsize=25, color=TEXT, fontweight="bold", va="top")
    fig.text(0.035, 0.918, "FROM ANNOTATED CALLS TO VISUAL STATE IMPRESSIONS", fontsize=9, color=MUTED, va="top")
    fig.text(0.965, 0.944, "100 MIN  /  192 kHz", fontsize=9, color=MUTED, ha="right", va="top")
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.895, 0.895], transform=fig.transFigure, color=GRID, lw=1.0))

    stage_label(fig, 0.035, 0.865, "01", "ANNOTATED AUDIO")
    stage_label(fig, 0.200, 0.865, "02", "FEATURE PROFILES")
    stage_label(fig, 0.380, 0.865, "03", "PCA + KMEANS")
    stage_label(fig, 0.555, 0.865, "04", "STATE WINDOWS")
    stage_label(fig, 0.785, 0.865, "05", "SYNTHETIC SCENES")
    for x1, x2 in [(0.18, 0.20), (0.345, 0.38), (0.525, 0.555), (0.76, 0.785)]:
        flow_arrow(fig, (x1, 0.71), (x2, 0.71))

    draw_input_panel(fig, state_rows)
    whistle_profile_keys = ["duration_sec", "rms_energy", "spectral_centroid_hz", "bandwidth_est_hz", "zero_crossing_rate"]
    click_profile_keys = ["original_duration_sec", "rms_energy", "spectral_centroid_hz", "bandwidth_est_hz", "estimated_click_rate_per_sec"]
    draw_feature_panel(fig, [0.200, 0.715, 0.145, 0.115], whistle_summary, whistle_profile_keys, "WHISTLE")
    draw_feature_panel(fig, [0.200, 0.595, 0.145, 0.105], click_summary, click_profile_keys, "CLICK TRAIN")
    draw_pca_panel(fig, [0.380, 0.595, 0.070, 0.235], whistle_rows, "WHISTLE", 303)
    draw_pca_panel(fig, [0.455, 0.595, 0.070, 0.235], click_rows, "CLICK", 1500)
    draw_state_panel(fig, state_rows)
    draw_spectrogram(fig, [0.785, 0.715, 0.180, 0.115], POSITIVE_AUDIO, "POSITIVE-ACTIVE", GREEN)
    draw_spectrogram(fig, [0.785, 0.595, 0.180, 0.105], NEGATIVE_AUDIO, "NEGATIVE-AROUSED", ORANGE)

    flow_arrow(fig, (0.875, 0.575), (0.875, 0.535))
    fig.text(0.035, 0.535, "06", color=CYAN, fontsize=11, fontweight="bold", va="center")
    fig.text(0.052, 0.535, "RADIAL STATE IMPRESSIONS", color=TEXT, fontsize=12, fontweight="bold", va="center")
    fig.text(0.965, 0.535, "AUDIO ENVELOPE  /  ARTISTIC MAPPING", color=MUTED, fontsize=8, ha="right", va="center")
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.518, 0.518], transform=fig.transFigure, color=GRID, lw=1.0))
    draw_radial_outputs(fig)

    fig.text(
        0.965,
        0.035,
        "ACOUSTIC-STATE CANDIDATES  ·  NOT BEHAVIORAL GROUND TRUTH",
        color="#66737B",
        fontsize=7,
        ha="right",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=240, facecolor=BG, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main() -> int:
    try:
        validate_inputs()
        build_figure()
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    print("[OK] Created dolphin acoustic-state pipeline visualization.")
    print(f"[OK] Figure saved to: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
