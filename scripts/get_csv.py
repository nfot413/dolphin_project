from pathlib import Path
import sqlite3

import pandas as pd


# 当前文件位置：~/dolphin_project/scripts/get_csv.py
# 项目根目录：~/dolphin_project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据库默认位置：~/dolphin_project/exports/pamguard/database.sqlite3
# 如果你的 database.sqlite3 不在根目录，请修改这一行
DB_PATH = PROJECT_ROOT / "exports" / "pamguard" / "database.sqlite3"

# CSV 输出目录：~/dolphin_project/exports/pamguard_csv
OUT_DIR = PROJECT_ROOT / "exports" / "pamguard_csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLICK_TABLE = "Click_Detector_Clicks"
WHISTLE_TABLE = "Whistle_and_Moan_Detector"


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    query = """
    SELECT name
    FROM sqlite_master
    WHERE type = 'table' AND name = ?;
    """
    return conn.execute(query, (table_name,)).fetchone() is not None


def export_clicks(conn: sqlite3.Connection) -> pd.DataFrame:
    query = f"""
    SELECT
        Id,
        UID,
        UTC,
        UTCMilliseconds,
        PCLocalTime,
        PCTime,
        ChannelBitmap,
        SequenceBitmap,
        channelMap,
        startSample,
        startSeconds,
        duration,
        lowFreq,
        highFreq,
        amplitude,
        detectionType,
        ClickNumber,
        SpeciesCode
    FROM {CLICK_TABLE}
    ORDER BY startSeconds ASC;
    """

    df = pd.read_sql_query(query, conn)
    out_path = OUT_DIR / "pamguard_clicks.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Exported clicks: {len(df)} rows -> {out_path}")
    return df


def export_whistles(conn: sqlite3.Connection) -> pd.DataFrame:
    query = f"""
    SELECT
        Id,
        UID,
        UTC,
        UTCMilliseconds,
        PCLocalTime,
        PCTime,
        ChannelBitmap,
        SequenceBitmap,
        UpdateOf,
        channelMap,
        startSample,
        startSeconds,
        duration,
        lowFreq,
        highFreq,
        amplitude,
        detectionType
    FROM {WHISTLE_TABLE}
    ORDER BY startSeconds ASC;
    """

    df = pd.read_sql_query(query, conn)
    out_path = OUT_DIR / "pamguard_whistles.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Exported whistles: {len(df)} rows -> {out_path}")
    return df


def save_summary(clicks: pd.DataFrame | None, whistles: pd.DataFrame | None) -> None:
    rows = []

    if clicks is not None:
        rows.append(
            {
                "type": "click",
                "count": len(clicks),
                "min_startSeconds": clicks["startSeconds"].min() if len(clicks) else None,
                "max_startSeconds": clicks["startSeconds"].max() if len(clicks) else None,
                "mean_amplitude": clicks["amplitude"].mean() if len(clicks) else None,
                "output_file": str(OUT_DIR / "pamguard_clicks.csv"),
            }
        )

    if whistles is not None:
        rows.append(
            {
                "type": "whistle",
                "count": len(whistles),
                "min_startSeconds": whistles["startSeconds"].min() if len(whistles) else None,
                "max_startSeconds": whistles["startSeconds"].max() if len(whistles) else None,
                "mean_amplitude": whistles["amplitude"].mean() if len(whistles) else None,
                "output_file": str(OUT_DIR / "pamguard_whistles.csv"),
            }
        )

    summary = pd.DataFrame(rows)
    out_path = OUT_DIR / "summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"[OK] Saved summary -> {out_path}")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "请确认 database.sqlite3 是否在项目根目录下；"
            "如果不在，请修改 get_csv.py 中的 DB_PATH。"
        )

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Reading database: {DB_PATH}")
    print(f"[INFO] Output directory: {OUT_DIR}")

    with sqlite3.connect(DB_PATH) as conn:
        clicks = None
        whistles = None

        if table_exists(conn, CLICK_TABLE):
            clicks = export_clicks(conn)
        else:
            print(f"[WARN] Table not found: {CLICK_TABLE}")

        if table_exists(conn, WHISTLE_TABLE):
            whistles = export_whistles(conn)
        else:
            print(f"[WARN] Table not found: {WHISTLE_TABLE}")

        save_summary(clicks, whistles)


if __name__ == "__main__":
    main()