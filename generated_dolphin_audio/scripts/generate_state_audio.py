from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUT_AUDIO_DIR = PROJECT_ROOT / "generated_dolphin_audio" / "audio"
OUT_FIG_DIR = PROJECT_ROOT / "generated_dolphin_audio" / "figures"

OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


SAMPLE_RATE = 192_000
SCENE_DURATION = 20.0


def normalize_audio(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = np.max(np.abs(x))
    if max_abs < 1e-12:
        return x
    return x / max_abs * peak


def fade_in_out(x: np.ndarray, sr: int, fade_sec: float = 0.005) -> np.ndarray:
    n = len(x)
    fade_n = int(fade_sec * sr)

    if fade_n <= 0 or 2 * fade_n >= n:
        return x

    envelope = np.ones(n)
    envelope[:fade_n] = np.linspace(0, 1, fade_n)
    envelope[-fade_n:] = np.linspace(1, 0, fade_n)

    return x * envelope


def synth_whistle(
    sr: int,
    duration: float,
    f_start: float,
    f_end: float,
    amplitude: float,
    modulation_depth: float = 0.0,
    modulation_rate: float = 2.0,
    noise_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    n = int(duration * sr)
    t = np.arange(n) / sr

    base_freq = np.linspace(f_start, f_end, n)
    modulation = modulation_depth * np.sin(2 * np.pi * modulation_rate * t)

    if noise_jitter > 0:
        random_curve = rng.normal(0, 1, n)
        kernel_size = max(5, int(0.01 * sr))
        kernel = np.ones(kernel_size) / kernel_size
        smooth_noise = np.convolve(random_curve, kernel, mode="same")
        jitter = noise_jitter * smooth_noise
    else:
        jitter = 0.0

    freq = base_freq + modulation + jitter
    freq = np.clip(freq, 500, sr / 2 - 1000)

    phase = 2 * np.pi * np.cumsum(freq) / sr
    x = amplitude * np.sin(phase)

    # 加入轻微二次谐波，让 whistle 不那么单薄
    x += 0.15 * amplitude * np.sin(2 * phase)

    x = fade_in_out(x, sr, fade_sec=0.01)
    return x


def synth_click(
    sr: int,
    duration: float = 0.0015,
    center_freq: float = 45000,
    amplitude: float = 0.6,
) -> np.ndarray:
    n = max(8, int(duration * sr))
    t = np.arange(n) / sr

    sigma = duration / 6
    center = duration / 2

    envelope = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    carrier = np.sin(2 * np.pi * center_freq * t)

    x = amplitude * envelope * carrier
    x = fade_in_out(x, sr, fade_sec=0.0002)
    return x


def add_signal(scene: np.ndarray, signal: np.ndarray, start_sec: float, sr: int) -> None:
    start = int(start_sec * sr)
    end = start + len(signal)

    if start >= len(scene):
        return

    if end > len(scene):
        signal = signal[: len(scene) - start]
        end = len(scene)

    scene[start:end] += signal


def add_click_train(
    scene: np.ndarray,
    sr: int,
    start_sec: float,
    duration: float,
    ici_mean: float,
    ici_jitter: float,
    amplitude: float,
    center_freq: float,
    rng: np.random.Generator,
) -> None:
    t = start_sec

    while t < start_sec + duration:
        click_duration = rng.uniform(0.0008, 0.002)
        click = synth_click(
            sr=sr,
            duration=click_duration,
            center_freq=center_freq * rng.uniform(0.85, 1.15),
            amplitude=amplitude * rng.uniform(0.7, 1.2),
        )
        add_signal(scene, click, t, sr)

        ici = rng.normal(ici_mean, ici_jitter)
        ici = max(0.003, ici)
        t += ici


def add_background_noise(
    scene: np.ndarray,
    rng: np.random.Generator,
    level: float = 0.015,
) -> np.ndarray:
    white = rng.normal(0, 1, len(scene))

    # 简单低频平滑背景，模拟水下环境噪声
    kernel_size = 512
    kernel = np.ones(kernel_size) / kernel_size
    low_noise = np.convolve(white, kernel, mode="same")

    mixed_noise = 0.7 * low_noise + 0.3 * white
    mixed_noise = mixed_noise / (np.max(np.abs(mixed_noise)) + 1e-12)

    return scene + level * mixed_noise


def generate_positive_active_scene(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scene = np.zeros(int(SCENE_DURATION * SAMPLE_RATE), dtype=np.float32)

    # 稳定、中频、较平滑 whistle
    whistle_times = [1.5, 4.2, 7.0, 10.8, 14.5, 17.2]

    for start in whistle_times:
        duration = rng.uniform(0.45, 0.95)
        f_start = rng.uniform(4500, 8000)
        f_end = rng.uniform(6500, 12000)

        whistle = synth_whistle(
            sr=SAMPLE_RATE,
            duration=duration,
            f_start=f_start,
            f_end=f_end,
            amplitude=rng.uniform(0.18, 0.32),
            modulation_depth=rng.uniform(300, 900),
            modulation_rate=rng.uniform(1.0, 3.0),
            noise_jitter=50,
            rng=rng,
        )
        add_signal(scene, whistle, start, SAMPLE_RATE)

    # 中等密度 click train
    click_train_specs = [
        (2.4, 0.8, 0.030, 0.006, 0.18, 42000),
        (8.2, 1.2, 0.022, 0.005, 0.22, 50000),
        (13.0, 1.0, 0.018, 0.004, 0.25, 52000),
        (18.0, 0.7, 0.026, 0.006, 0.18, 46000),
    ]

    for spec in click_train_specs:
        add_click_train(
            scene=scene,
            sr=SAMPLE_RATE,
            start_sec=spec[0],
            duration=spec[1],
            ici_mean=spec[2],
            ici_jitter=spec[3],
            amplitude=spec[4],
            center_freq=spec[5],
            rng=rng,
        )

    scene = add_background_noise(scene, rng, level=0.012)
    return normalize_audio(scene, peak=0.90)


def generate_negative_aroused_scene(seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scene = np.zeros(int(SCENE_DURATION * SAMPLE_RATE), dtype=np.float32)

    # 高频、宽带、不规则 whistle
    whistle_times = [0.8, 2.6, 4.0, 6.4, 9.1, 11.0, 13.3, 15.1, 17.6]

    for start in whistle_times:
        duration = rng.uniform(0.25, 0.75)

        f_start = rng.uniform(9000, 18000)
        f_end = rng.uniform(14000, 28000)

        if rng.random() < 0.5:
            f_start, f_end = f_end, f_start

        whistle = synth_whistle(
            sr=SAMPLE_RATE,
            duration=duration,
            f_start=f_start,
            f_end=f_end,
            amplitude=rng.uniform(0.28, 0.48),
            modulation_depth=rng.uniform(1200, 3500),
            modulation_rate=rng.uniform(2.5, 7.0),
            noise_jitter=rng.uniform(150, 450),
            rng=rng,
        )
        add_signal(scene, whistle, start, SAMPLE_RATE)

    # 密集、不规则、高能 click / pulse train
    click_train_specs = [
        (1.5, 1.4, 0.010, 0.004, 0.35, 56000),
        (5.0, 1.8, 0.007, 0.003, 0.42, 62000),
        (8.0, 2.0, 0.006, 0.0025, 0.45, 68000),
        (12.0, 1.6, 0.009, 0.004, 0.40, 58000),
        (16.2, 1.8, 0.006, 0.003, 0.48, 70000),
    ]

    for spec in click_train_specs:
        add_click_train(
            scene=scene,
            sr=SAMPLE_RATE,
            start_sec=spec[0],
            duration=spec[1],
            ici_mean=spec[2],
            ici_jitter=spec[3],
            amplitude=spec[4],
            center_freq=spec[5],
            rng=rng,
        )

    # 额外加入少量强瞬态脉冲，模拟高强度/干扰样片段
    for _ in range(18):
        t = rng.uniform(0.5, SCENE_DURATION - 0.5)
        click = synth_click(
            sr=SAMPLE_RATE,
            duration=rng.uniform(0.001, 0.003),
            center_freq=rng.uniform(30000, 75000),
            amplitude=rng.uniform(0.25, 0.55),
        )
        add_signal(scene, click, t, SAMPLE_RATE)

    scene = add_background_noise(scene, rng, level=0.025)
    return normalize_audio(scene, peak=0.95)


def save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    audio = normalize_audio(audio, peak=0.95)
    audio_int16 = np.int16(np.clip(audio, -1, 1) * 32767)
    wavfile.write(path, sr, audio_int16)


def plot_spectrogram(audio: np.ndarray, sr: int, out_path: Path, title: str) -> None:
    f, t, sxx = spectrogram(
        audio,
        fs=sr,
        window="hann",
        nperseg=2048,
        noverlap=1024,
        scaling="spectrum",
        mode="magnitude",
    )

    sxx_db = 20 * np.log10(sxx + 1e-12)

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f / 1000, sxx_db, shading="auto")
    plt.ylim(0, 96)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.title(title)
    plt.colorbar(label="Magnitude (dB)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    positive = generate_positive_active_scene(seed=42)
    negative = generate_negative_aroused_scene(seed=123)

    positive_wav = OUT_AUDIO_DIR / "positive_active_dolphin_scene.wav"
    negative_wav = OUT_AUDIO_DIR / "negative_aroused_dolphin_scene.wav"

    save_wav(positive_wav, positive, SAMPLE_RATE)
    save_wav(negative_wav, negative, SAMPLE_RATE)

    plot_spectrogram(
        positive,
        SAMPLE_RATE,
        OUT_FIG_DIR / "positive_active_spectrogram.png",
        "Synthetic positive-active dolphin acoustic scene",
    )

    plot_spectrogram(
        negative,
        SAMPLE_RATE,
        OUT_FIG_DIR / "negative_aroused_spectrogram.png",
        "Synthetic negative-aroused dolphin acoustic scene",
    )

    print("[OK] Generated synthetic dolphin acoustic scenes.")
    print(f"[OK] Positive audio: {positive_wav}")
    print(f"[OK] Negative audio: {negative_wav}")
    print(f"[OK] Figures: {OUT_FIG_DIR}")


if __name__ == "__main__":
    main()