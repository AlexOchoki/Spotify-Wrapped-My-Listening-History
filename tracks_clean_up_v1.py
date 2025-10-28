

#!/usr/bin/env python3
"""
Spotify Takeout Cleaner — Local Version (final)
Run in VS Code. No CLI flags needed.
"""

from pathlib import Path
from fnmatch import fnmatch
import json
import logging
import pandas as pd

# ====== EDIT THESE ======
INPUT_DIR  = Path(r"C:\Users\HomePC\Desktop\Projects\Spotify Account Data")  # folder with JSONs
OUTPUT_DIR = Path(r"C:\Users\HomePC\Desktop\Projects\Spotify Account Data")  # where cleaned files go
# ========================

TZ_LOCAL = "Africa/Nairobi"

# Only pick streaming history files
from fnmatch import fnmatch

# match with or without extension; tolerate underscores/spaces/case
STREAM_PATTERNS = [
    "*streaming*history*",
    "*extended*streaming*history*",
    "*streaming*history*music*",
]

def is_streaming_history_file(path: Path) -> bool:
    name = path.name.lower()
    return any(fnmatch(name, pat) for pat in STREAM_PATTERNS)

def discover_json_files(input_dir: Path):
    # include files with or without .json
    all_files = sorted([p for p in input_dir.iterdir() if p.is_file()])
    files = [p for p in all_files if is_streaming_history_file(p)]
    if not files:
        raise FileNotFoundError(
            f"No streaming history files found in {input_dir} "
            f"(looked for patterns: {', '.join(STREAM_PATTERNS)})"
        )
    # log what we’ll load so you can verify both files got picked up
    for f in files:
        logging.info("Selected file: %s", f.name)
    return files


def looks_like_stream_row(x: dict) -> bool:
    return ("ts" in x or "endTime" in x) and ("ms_played" in x or "msPlayed" in x)

def load_json_records(files):
    all_records = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    filtered = [r for r in data if isinstance(r, dict) and looks_like_stream_row(r)]
                    if not filtered:
                        logging.warning("No streaming rows in %s; skipping", f.name)
                        continue
                    all_records.extend(filtered)
                else:
                    logging.warning("%s is not a JSON array; skipping", f.name)
        except Exception as e:
            logging.warning("Skipping %s: %s", f, e)
    return all_records

def normalize_record(rec: dict) -> dict:
    """
    Map classic schema to extended-like keys.
    Classic keys: endTime, msPlayed, artistName, trackName
    Extended keys: ts, ms_played, master_metadata_* (plus episode/show for podcasts)
    """
    out = dict(rec)
    if "ts" not in out and "endTime" in out:
        out["ts"] = out.get("endTime")                 # naive string; treat as UTC on parse
        out["ms_played"] = out.get("msPlayed")
        if out.get("trackName"):
            out["master_metadata_track_name"] = out["trackName"]
        if out.get("artistName"):
            out["master_metadata_album_artist_name"] = out["artistName"]
    return out

def normalize_records(records):
    return [normalize_record(r) for r in records if isinstance(r, dict) and ("ts" in r or "endTime" in r)]

def _coalesce_cols(df: pd.DataFrame, cols, default=None) -> pd.Series:
    """Return first non-null across given columns; if none exist, return default series."""
    s = None
    for c in cols:
        if c in df.columns:
            s = df[c] if s is None else s.where(s.notna(), df[c])
    if s is None:
        s = pd.Series([default] * len(df))
    return s.fillna(default)

def build_dataframe(records) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("No valid records found.")

    if "ts" not in df.columns:
        raise ValueError("No 'ts' found after normalization. Are these streaming history files?")

    # Parse UTC -> local tz BEFORE feature engineering
    ts_parsed = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.loc[ts_parsed.notna()].copy()
    ts_local = ts_parsed.loc[ts_parsed.notna()].dt.tz_convert(TZ_LOCAL)
    df["Played At"] = ts_local

    # Core fields (music+podcasts)
    df["Track Name"] = _coalesce_cols(df, ["master_metadata_track_name", "episode_name"], "").astype(str).str.strip()
    df["Artist(s)"]  = _coalesce_cols(df,
                         ["master_metadata_album_artist_name", "artist_name", "show_name"], ""
                       ).astype(str).str.strip()
    df["Album"]      = _coalesce_cols(df, ["master_metadata_album_album_name", "show_name"], "").astype(str).str.strip()
    df["Platform"]   = _coalesce_cols(df, ["platform"], "")

    # Duration
    ms = _coalesce_cols(df, ["ms_played", "msPlayed"], 0)
    df["Duration (ms)"]   = pd.to_numeric(ms, errors="coerce")
    df["Duration (mins)"] = df["Duration (ms)"] / 60000.0

    # Time features
    df["Year"]        = df["Played At"].dt.year
    df["Month"]       = df["Played At"].dt.month_name().str[:3]
    df["Month Year"]  = df["Played At"].dt.strftime("%b %Y")
    df["Date"]        = df["Played At"].dt.date
    df["Hour"]        = df["Played At"].dt.hour
    df["Day"]         = df["Played At"].dt.day_name().str[:3]
    df["Day Date"]    = df["Played At"].dt.day
    df["MonthStart"]  = df["Played At"].dt.to_period("M").dt.to_timestamp()

    # Safer record_id: ISO8601 local + track/episode + duration
    iso_ts = df["Played At"].dt.strftime("%Y-%m-%dT%H:%M:%S%z").str.replace(
        r"(\+|\-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True
    )
    df["record_id"] = (
        iso_ts + "_" +
        df["Track Name"].fillna("").astype(str).str.strip() + "_" +
        df["Duration (ms)"].fillna(-1).astype("Int64").astype(str)
    )

    # Dedupe
    before = len(df)
    df = df.drop_duplicates(subset=["record_id"], keep="first").reset_index(drop=True)
    logging.info("Dropped %d duplicates.", before - len(df))

    # Final order
    cols = [
        "record_id",
        "Played At", "Platform",
        "Track Name", "Artist(s)", "Album",
        "Duration (ms)", "Duration (mins)",
        "Year", "Month", "Month Year", "MonthStart",
        "Date", "Hour", "Day", "Day Date",
    ]
    return df[cols]

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    files = discover_json_files(INPUT_DIR)
    logging.info("Found %d valid Spotify JSON files.", len(files))

    records = load_json_records(files)
    records = normalize_records(records)
    logging.info("Loaded %,d raw records.", len(records))

    df = build_dataframe(records)

        # === Export section (CSV only) ===
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "spotify_clean.csv"

    # Save to CSV
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    logging.info("✅ Done! Cleaned CSV saved to: %s", out_csv)
    print(f"Rows exported: {len(df):,}\nFile: {out_csv}")


if __name__ == "__main__":
    main()
