from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "segments"
OUTPUT_DIR = PROJECT_ROOT / "official_whistle_features" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "official_whistle_features.csv"


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


def zero_crossing_rate(audio: np.ndarray) -> float:
    if len(audio) < 2:
        return np.nan

    signs = np.sign(audio)
    signs[signs == 0] = 1
    crossings = np.sum(signs[:-1] != signs[1:])

    return crossings / len(audio)


def spectral_features(audio: np.ndarray, sample_rate: int):
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

    spectral_centroid = np.sum(freqs * mean_power) / total_power

    spectral_bandwidth = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * mean_power) / total_power
    )

    cumulative_power = np.cumsum(mean_power)
    rolloff_threshold = 0.85 * cumulative_power[-1]
    rolloff_idx = np.searchsorted(cumulative_power, rolloff_threshold)
    spectral_rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    dominant_frequency = freqs[np.argmax(mean_power)]

    # 用能量阈值粗略估计有效频率范围
    threshold = 0.10 * np.max(mean_power)
    active = mean_power >= threshold

    if np.any(active):
        active_freqs = freqs[active]
        low_freq = float(active_freqs.min())
        high_freq = float(active_freqs.max())
        bandwidth = high_freq - low_freq
    else:
        low_freq = np.nan
        high_freq = np.nan
        bandwidth = np.nan

    return {
        "spectral_centroid_hz": float(spectral_centroid),
        "spectral_bandwidth_hz": float(spectral_bandwidth),
        "spectral_rolloff_85_hz": float(spectral_rolloff),
        "dominant_frequency_hz": float(dominant_frequency),
        "low_freq_est_hz": low_freq,
        "high_freq_est_hz": high_freq,
        "bandwidth_est_hz": bandwidth,
    }


def extract_features_from_file(wav_path: Path) -> dict:
    sample_rate, audio = load_wav_mono(wav_path)

    duration_sec = len(audio) / sample_rate

    rms_energy = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else np.nan
    peak_amplitude = float(np.max(np.abs(audio))) if len(audio) > 0 else np.nan
    mean_amplitude = float(np.mean(np.abs(audio))) if len(audio) > 0 else np.nan
    zcr = float(zero_crossing_rate(audio))

    spec_feats = spectral_features(audio, sample_rate)

    row = {
        "file_name": wav_path.name,
        "file_path": str(wav_path),
        "sample_rate": sample_rate,
        "num_samples": len(audio),
        "duration_sec": duration_sec,
        "rms_energy": rms_energy,
        "peak_amplitude": peak_amplitude,
        "mean_abs_amplitude": mean_amplitude,
        "zero_crossing_rate": zcr,
    }

    row.update(spec_feats)
    return row


def main():
    wav_files = sorted(INPUT_DIR.rglob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(f"No wav files found in: {INPUT_DIR}")

    print(f"[INFO] Found {len(wav_files)} wav files in {INPUT_DIR}")

    rows = []

    for idx, wav_path in enumerate(wav_files, start=1):
        try:
            features = extract_features_from_file(wav_path)
            rows.append(features)
            print(f"[{idx}/{len(wav_files)}] OK: {wav_path.name}")
        except Exception as exc:
            print(f"[{idx}/{len(wav_files)}] ERROR: {wav_path.name}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print()
    print(f"[OK] Saved feature table:")
    print(f"     {OUTPUT_CSV}")
    print()
    print("[INFO] Feature columns:")
    for col in df.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    main()