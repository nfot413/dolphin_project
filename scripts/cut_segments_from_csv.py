from pathlib import Path
import wave

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 原始完整录音
WAV_PATH = PROJECT_ROOT / "data" / "raw" / "full_recording.wav"

# PAMGuard 导出的 whistle CSV
CSV_PATH = PROJECT_ROOT / "exports" / "pamguard_csv" / "pamguard_whistles.csv"

# 输出目录
OUT_DIR = PROJECT_ROOT / "outputs" / "cut_segments" / "whistles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 每个片段前后额外保留一点时间，方便后续看频谱图
PRE_CONTEXT_SEC = 0.10
POST_CONTEXT_SEC = 0.10

# 过滤太短或太长的片段
MIN_DURATION_SEC = 0.05
MAX_DURATION_SEC = 5.00


def get_wav_info(wav_path: Path) -> dict:
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        duration_sec = n_frames / sample_rate

    return {
        "sample_rate": sample_rate,
        "n_channels": n_channels,
        "sample_width": sample_width,
        "n_frames": n_frames,
        "duration_sec": duration_sec,
    }


def cut_wav_segment(
    wav_path: Path,
    out_path: Path,
    start_sec: float,
    end_sec: float,
) -> bool:
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        total_frames = wf.getnframes()

        start_frame = max(0, int(start_sec * sample_rate))
        end_frame = min(total_frames, int(end_sec * sample_rate))

        if end_frame <= start_frame:
            return False

        wf.setpos(start_frame)
        audio_bytes = wf.readframes(end_frame - start_frame)

    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(n_channels)
        out.setsampwidth(sample_width)
        out.setframerate(sample_rate)
        out.writeframes(audio_bytes)

    return True


def main() -> None:
    if not WAV_PATH.exists():
        raise FileNotFoundError(f"Raw wav not found: {WAV_PATH}")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    wav_info = get_wav_info(WAV_PATH)

    print("[INFO] WAV file:")
    print(f"  path: {WAV_PATH}")
    print(f"  sample_rate: {wav_info['sample_rate']}")
    print(f"  channels: {wav_info['n_channels']}")
    print(f"  duration_sec: {wav_info['duration_sec']:.3f}")

    df = pd.read_csv(CSV_PATH)

    required_cols = ["Id", "startSeconds", "duration"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    df = df.copy()
    df["startSeconds"] = pd.to_numeric(df["startSeconds"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    df = df.dropna(subset=["startSeconds", "duration"])
    df = df.sort_values("startSeconds").reset_index(drop=True)

    # 这里假设 duration 单位是秒
    df["endSeconds"] = df["startSeconds"] + df["duration"]

    # 基础过滤
    df = df[df["duration"] >= MIN_DURATION_SEC]
    df = df[df["duration"] <= MAX_DURATION_SEC]

    print(f"[INFO] Valid segments to cut: {len(df)}")
    print(f"[INFO] Output directory: {OUT_DIR}")

    index_rows = []

    for _, row in df.iterrows():
        det_id = int(row["Id"])
        start_sec = float(row["startSeconds"])
        end_sec = float(row["endSeconds"])

        cut_start = max(0.0, start_sec - PRE_CONTEXT_SEC)
        cut_end = min(wav_info["duration_sec"], end_sec + POST_CONTEXT_SEC)

        out_name = f"whistle_id_{det_id:06d}_start_{start_sec:.3f}s.wav"
        out_path = OUT_DIR / out_name

        ok = cut_wav_segment(
            wav_path=WAV_PATH,
            out_path=out_path,
            start_sec=cut_start,
            end_sec=cut_end,
        )

        if not ok:
            continue

        index_rows.append(
            {
                "Id": det_id,
                "original_startSeconds": start_sec,
                "original_endSeconds": end_sec,
                "original_duration": float(row["duration"]),
                "cut_startSeconds": cut_start,
                "cut_endSeconds": cut_end,
                "cut_duration": cut_end - cut_start,
                "lowFreq": row.get("lowFreq", None),
                "highFreq": row.get("highFreq", None),
                "amplitude": row.get("amplitude", None),
                "file": str(out_path),
            }
        )

    index_df = pd.DataFrame(index_rows)
    index_path = OUT_DIR / "segments_index.csv"
    index_df.to_csv(index_path, index=False)

    print(f"[OK] Cut {len(index_df)} segments.")
    print(f"[OK] Index saved to: {index_path}")


if __name__ == "__main__":
    main()