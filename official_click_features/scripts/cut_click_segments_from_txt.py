from pathlib import Path
import re
import wave

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_WAV_PATH = PROJECT_ROOT / "data" / "raw" / "full_recording.wav"
CLICKS_TXT_PATH = PROJECT_ROOT / "data" / "labels" / "clicks.txt"

OUT_DIR = PROJECT_ROOT / "official_click_features" / "segments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CSV = OUT_DIR / "click_segments_index.csv"

# 前后额外保留一点上下文，方便后续看频谱
PRE_CONTEXT_SEC = 0.02
POST_CONTEXT_SEC = 0.02

# 防止极端异常片段
MIN_DURATION_SEC = 0.001
MAX_DURATION_SEC = 30.0


def parse_decimal_comma(value: str) -> float:
    return float(value.replace(",", "."))


def load_clicks_txt(txt_path: Path) -> pd.DataFrame:
    rows = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = re.split(r"\s+", line)

            if len(parts) < 3:
                print(f"[WARN] Skip line {line_no}: {line}")
                continue

            start_sec = parse_decimal_comma(parts[0])
            end_sec = parse_decimal_comma(parts[1])
            ici = float(parts[2].replace(",", "."))

            rows.append(
                {
                    "line_no": line_no,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": end_sec - start_sec,
                    "ici": ici,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values("start_sec").reset_index(drop=True)
    return df


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
    if not RAW_WAV_PATH.exists():
        raise FileNotFoundError(f"Raw wav not found: {RAW_WAV_PATH}")

    if not CLICKS_TXT_PATH.exists():
        raise FileNotFoundError(f"clicks.txt not found: {CLICKS_TXT_PATH}")

    wav_info = get_wav_info(RAW_WAV_PATH)

    print("[INFO] RAW WAV:")
    print(f"  path: {RAW_WAV_PATH}")
    print(f"  sample_rate: {wav_info['sample_rate']}")
    print(f"  channels: {wav_info['n_channels']}")
    print(f"  duration_sec: {wav_info['duration_sec']:.3f}")

    clicks = load_clicks_txt(CLICKS_TXT_PATH)

    clicks = clicks[
        (clicks["duration_sec"] >= MIN_DURATION_SEC)
        & (clicks["duration_sec"] <= MAX_DURATION_SEC)
    ].copy()

    print(f"[INFO] Valid click train segments: {len(clicks)}")
    print(f"[INFO] Output directory: {OUT_DIR}")

    index_rows = []

    for idx, row in clicks.iterrows():
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])

        cut_start = max(0.0, start_sec - PRE_CONTEXT_SEC)
        cut_end = min(wav_info["duration_sec"], end_sec + POST_CONTEXT_SEC)

        out_name = (
            f"click_train_{idx:05d}_"
            f"start_{start_sec:.3f}s_"
            f"ici_{row['ici']:.1f}.wav"
        )
        out_path = OUT_DIR / out_name

        ok = cut_wav_segment(
            wav_path=RAW_WAV_PATH,
            out_path=out_path,
            start_sec=cut_start,
            end_sec=cut_end,
        )

        if ok:
            index_rows.append(
                {
                    "segment_id": idx,
                    "source_line_no": int(row["line_no"]),
                    "original_start_sec": start_sec,
                    "original_end_sec": end_sec,
                    "original_duration_sec": float(row["duration_sec"]),
                    "cut_start_sec": cut_start,
                    "cut_end_sec": cut_end,
                    "cut_duration_sec": cut_end - cut_start,
                    "ici": float(row["ici"]),
                    "file_name": out_name,
                    "file_path": str(out_path),
                }
            )

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(INDEX_CSV, index=False)

    print(f"[OK] Cut {len(index_df)} click train segments.")
    print(f"[OK] Index saved to: {INDEX_CSV}")


if __name__ == "__main__":
    main()