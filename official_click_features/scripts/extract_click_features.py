from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram, find_peaks


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEGMENT_DIR = PROJECT_ROOT / "official_click_features" / "segments"
INDEX_CSV = SEGMENT_DIR / "click_segments_index.csv"

OUT_DIR = PROJECT_ROOT / "official_click_features" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / "official_click_features.csv"


def load_wav_mono(wav_path: Path):
    sample_rate, audio = wavfile.read(wav_path)

    if audio.ndim > 1:
        audio = audio[:, 0]

    if np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)

    return sample_rate, audio


def basic_time_features(audio: np.ndarray, sample_rate: int) -> dict:
    duration_sec = len(audio) / sample_rate if sample_rate > 0 else np.nan

    if len(audio) == 0:
        return {
            "duration_sec": np.nan,
            "rms_energy": np.nan,
            "peak_amplitude": np.nan,
            "mean_abs_amplitude": np.nan,
            "zero_crossing_rate": np.nan,
        }

    rms_energy = float(np.sqrt(np.mean(audio ** 2)))
    peak_amplitude = float(np.max(np.abs(audio)))
    mean_abs_amplitude = float(np.mean(np.abs(audio)))

    signs = np.sign(audio)
    signs[signs == 0] = 1
    zcr = float(np.mean(signs[:-1] != signs[1:])) if len(audio) > 1 else np.nan

    return {
        "duration_sec": duration_sec,
        "rms_energy": rms_energy,
        "peak_amplitude": peak_amplitude,
        "mean_abs_amplitude": mean_abs_amplitude,
        "zero_crossing_rate": zcr,
    }


def spectral_features(audio: np.ndarray, sample_rate: int) -> dict:
    if len(audio) < 256:
        return {
            "spectral_centroid_hz": np.nan,
            "spectral_bandwidth_hz": np.nan,
            "spectral_rolloff_85_hz": np.nan,
            "dominant_frequency_hz": np.nan,
            "low_freq_est_hz": np.nan,
            "high_freq_est_hz": np.nan,
            "bandwidth_est_hz": np.nan,
        }

    nperseg = min(2048, len(audio))
    noverlap = nperseg // 2

    freqs, times, spec = spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )

    power = spec ** 2
    mean_power = np.mean(power, axis=1)
    total_power = np.sum(mean_power)

    if total_power <= 0:
        return {
            "spectral_centroid_hz": np.nan,
            "spectral_bandwidth_hz": np.nan,
            "spectral_rolloff_85_hz": np.nan,
            "dominant_frequency_hz": np.nan,
            "low_freq_est_hz": np.nan,
            "high_freq_est_hz": np.nan,
            "bandwidth_est_hz": np.nan,
        }

    centroid = float(np.sum(freqs * mean_power) / total_power)

    bandwidth = float(
        np.sqrt(np.sum(((freqs - centroid) ** 2) * mean_power) / total_power)
    )

    cumulative = np.cumsum(mean_power)
    rolloff_idx = np.searchsorted(cumulative, 0.85 * cumulative[-1])
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    dominant = float(freqs[np.argmax(mean_power)])

    threshold = 0.10 * np.max(mean_power)
    active = mean_power >= threshold

    if np.any(active):
        active_freqs = freqs[active]
        low_freq = float(active_freqs.min())
        high_freq = float(active_freqs.max())
        bw_est = high_freq - low_freq
    else:
        low_freq = np.nan
        high_freq = np.nan
        bw_est = np.nan

    return {
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_rolloff_85_hz": rolloff,
        "dominant_frequency_hz": dominant,
        "low_freq_est_hz": low_freq,
        "high_freq_est_hz": high_freq,
        "bandwidth_est_hz": bw_est,
    }


def estimate_click_peaks(audio: np.ndarray, sample_rate: int) -> dict:
    """
    粗略估计片段内的脉冲峰数量和 ICI。
    注意：这是从切片音频中用简单峰值法估计，不等同于官方精确标注。
    """
    if len(audio) == 0:
        return {
            "estimated_peak_count": 0,
            "estimated_click_rate_per_sec": np.nan,
            "estimated_mean_ici_sec": np.nan,
            "estimated_min_ici_sec": np.nan,
        }

    envelope = np.abs(audio)

    # 自适应阈值，避免把背景噪声全部当成峰
    threshold = np.mean(envelope) + 3.0 * np.std(envelope)

    # 限制最小峰间距，避免一个 click 被重复数多次
    min_distance_samples = int(0.001 * sample_rate)  # 1 ms

    peaks, _ = find_peaks(
        envelope,
        height=threshold,
        distance=max(1, min_distance_samples),
    )

    duration_sec = len(audio) / sample_rate

    if len(peaks) >= 2:
        peak_times = peaks / sample_rate
        ici = np.diff(peak_times)
        mean_ici = float(np.mean(ici))
        min_ici = float(np.min(ici))
    else:
        mean_ici = np.nan
        min_ici = np.nan

    click_rate = float(len(peaks) / duration_sec) if duration_sec > 0 else np.nan

    return {
        "estimated_peak_count": int(len(peaks)),
        "estimated_click_rate_per_sec": click_rate,
        "estimated_mean_ici_sec": mean_ici,
        "estimated_min_ici_sec": min_ici,
    }


def extract_features(wav_path: Path) -> dict:
    sample_rate, audio = load_wav_mono(wav_path)

    row = {
        "file_name": wav_path.name,
        "file_path": str(wav_path),
        "sample_rate": sample_rate,
        "num_samples": len(audio),
    }

    row.update(basic_time_features(audio, sample_rate))
    row.update(spectral_features(audio, sample_rate))
    row.update(estimate_click_peaks(audio, sample_rate))

    return row


def main() -> None:
    if not SEGMENT_DIR.exists():
        raise FileNotFoundError(f"Segment directory not found: {SEGMENT_DIR}")

    wav_files = sorted(
        p for p in SEGMENT_DIR.rglob("*.wav")
        if p.is_file()
    )

    if not wav_files:
        raise FileNotFoundError(f"No wav files found in: {SEGMENT_DIR}")

    print(f"[INFO] Found {len(wav_files)} click train wav files.")

    rows = []

    for i, wav_path in enumerate(wav_files, start=1):
        try:
            feat = extract_features(wav_path)
            rows.append(feat)
            print(f"[{i}/{len(wav_files)}] OK: {wav_path.name}")
        except Exception as exc:
            print(f"[{i}/{len(wav_files)}] ERROR: {wav_path.name}: {exc}")

    features = pd.DataFrame(rows)

    if INDEX_CSV.exists():
        index_df = pd.read_csv(INDEX_CSV)
        features = features.merge(index_df, on="file_name", how="left")

    features.to_csv(OUTPUT_CSV, index=False)

    print(f"[OK] Saved click features:")
    print(f"     {OUTPUT_CSV}")
    print("[INFO] Columns:")
    for col in features.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    main()