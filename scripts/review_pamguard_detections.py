import os
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


# 当前文件假设放在：~/dolphin_project/scripts/review_pamguard_detections.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

WAV_PATH = PROJECT_ROOT / "data" / "raw" / "full_recording.wav"
CLICK_CSV = PROJECT_ROOT / "exports" / "pamguard_csv" / "pamguard_clicks.csv"
WHISTLE_CSV = PROJECT_ROOT / "exports" / "pamguard_csv" / "pamguard_whistles.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "review_spectrograms"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_wav_segment(wav_path: Path, start_sec: float, duration_sec: float = 2.0):
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        total_frames = wf.getnframes()

        if sampwidth != 2:
            raise ValueError(
                f"Only 16-bit PCM WAV is supported. Got sample width = {sampwidth}"
            )

        start_frame = max(0, int(start_sec * sr))
        start_frame = min(start_frame, total_frames)

        n_frames = int(duration_sec * sr)
        n_frames = min(n_frames, total_frames - start_frame)

        wf.setpos(start_frame)
        raw = wf.readframes(n_frames)

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)[:, 0]

    audio = audio / 32768.0
    return audio, sr


def plot_spectrogram(
    audio: np.ndarray,
    sr: int,
    title: str,
    out_path: Path,
    max_freq: float,
    nperseg: int = 2048,
    noverlap: int = 1024,
):
    if len(audio) == 0:
        raise ValueError("Empty audio segment.")

    f, t, Sxx = spectrogram(
        audio,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )

    Sxx_db = 20 * np.log10(Sxx + 1e-12)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f / 1000, Sxx_db, shading="auto")
    plt.ylim(0, max_freq / 1000)
    plt.xlabel("Time within segment (s)")
    plt.ylabel("Frequency (kHz)")
    plt.title(title)
    plt.colorbar(label="Magnitude (dB)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def load_detection_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "startSeconds" not in df.columns:
        raise ValueError(f"{csv_path} does not contain startSeconds column.")

    df = df.copy()
    df["startSeconds"] = pd.to_numeric(df["startSeconds"], errors="coerce")
    df = df.dropna(subset=["startSeconds"])
    df = df.sort_values("startSeconds").reset_index(drop=True)

    for col in ["duration", "lowFreq", "highFreq", "amplitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def sample_detections(
    df: pd.DataFrame,
    n: int,
    random_state: int = 42,
    min_start: float | None = None,
    max_start: float | None = None,
    min_duration: float | None = None,
    min_amplitude: float | None = None,
) -> pd.DataFrame:
    filtered = df.copy()

    if min_start is not None:
        filtered = filtered[filtered["startSeconds"] >= min_start]

    if max_start is not None:
        filtered = filtered[filtered["startSeconds"] <= max_start]

    if min_duration is not None and "duration" in filtered.columns:
        filtered = filtered[filtered["duration"] >= min_duration]

    if min_amplitude is not None and "amplitude" in filtered.columns:
        filtered = filtered[filtered["amplitude"] >= min_amplitude]

    if len(filtered) == 0:
        return filtered

    return filtered.sample(n=min(n, len(filtered)), random_state=random_state)


def make_review_images(
    df: pd.DataFrame,
    kind: str,
    n: int,
    duration_sec: float,
    pre_context_sec: float,
    max_freq: float,
    random_state: int = 42,
):
    sample_df = sample_detections(df, n=n, random_state=random_state)

    if len(sample_df) == 0:
        print(f"[WARN] No {kind} detections to review.")
        return pd.DataFrame()

    rows = []

    for _, row in sample_df.iterrows():
        detection_id = int(row["Id"]) if "Id" in row and not pd.isna(row["Id"]) else -1
        start = float(row["startSeconds"])
        segment_start = max(0.0, start - pre_context_sec)

        audio, sr = read_wav_segment(
            WAV_PATH,
            start_sec=segment_start,
            duration_sec=duration_sec,
        )

        out_name = f"{kind}_id_{detection_id}_start_{start:.2f}s.png"
        out_path = OUT_DIR / out_name

        low_freq = row.get("lowFreq", np.nan)
        high_freq = row.get("highFreq", np.nan)
        amp = row.get("amplitude", np.nan)

        title = (
            f"{kind} | Id={detection_id} | start={start:.2f}s | "
            f"low={low_freq:.0f}Hz high={high_freq:.0f}Hz amp={amp:.1f}"
        )

        plot_spectrogram(
            audio=audio,
            sr=sr,
            title=title,
            out_path=out_path,
            max_freq=max_freq,
        )

        rows.append(
            {
                "kind": kind,
                "Id": detection_id,
                "startSeconds": start,
                "segmentStartSeconds": segment_start,
                "duration": row.get("duration", None),
                "lowFreq": row.get("lowFreq", None),
                "highFreq": row.get("highFreq", None),
                "amplitude": row.get("amplitude", None),
                "image": str(out_path),
            }
        )

    return pd.DataFrame(rows)


def main():
    print("[INFO] Loading PAMGuard CSV files...")
    print(f"[INFO] Click CSV: {CLICK_CSV}")
    print(f"[INFO] Whistle CSV: {WHISTLE_CSV}")

    whistles = load_detection_csv(WHISTLE_CSV)
    clicks = load_detection_csv(CLICK_CSV)

    print(f"[INFO] Whistle rows: {len(whistles)}")
    print(f"[INFO] Click rows: {len(clicks)}")

    # whistle：看 0–40 kHz，截取 3 秒
    whistle_review = make_review_images(
        whistles,
        kind="whistle",
        n=10,
        duration_sec=3.0,
        pre_context_sec=0.5,
        max_freq=40000,
        random_state=42,
    )

    # click：看 0–96 kHz，截取 1 秒
    click_review = make_review_images(
        clicks,
        kind="click",
        n=10,
        duration_sec=1.0,
        pre_context_sec=0.2,
        max_freq=96000,
        random_state=43,
    )

    review = pd.concat([whistle_review, click_review], ignore_index=True)
    review_csv = OUT_DIR / "review_index.csv"
    review.to_csv(review_csv, index=False)

    print("[OK] Done.")
    print(f"[OK] Generated review images in: {OUT_DIR}")
    print(f"[OK] Review index saved to: {review_csv}")


if __name__ == "__main__":
    main()