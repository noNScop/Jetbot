from pathlib import Path
import pandas as pd
import sys


def process_csv(path: Path) -> pd.DataFrame | None:
    """
    Loads a single CSV, shifts frame_timestamp down by 1 row,
    and discards the first and last rows. Returns None if the file
    is missing the expected column or has too few rows to be useful.
    """
    df = pd.read_csv(path)
 
    if "frame_timestamp" not in df.columns:
        print(f"  [skip] {path} — no 'frame_timestamp' column")
        return None
 
    if len(df) < 3:
        print(f"  [skip] {path} — too few rows ({len(df)}), nothing left after trim")
        return None
 
    df["frame_timestamp"] = df["frame_timestamp"].shift(1)
    df = df.iloc[1:-1].reset_index(drop=True)
    return df
 
 
def process_directory(input_dir: str, output_path: str):
    """
    Recursively finds all CSV files under input_dir, processes each one,
    and concatenates the results into a single output CSV.
    """
    root = Path(input_dir)
    csv_files = sorted(root.rglob("*.csv"))
 
    if not csv_files:
        print(f"No CSV files found under: {root}")
        sys.exit(1)
 
    print(f"Found {len(csv_files)} CSV file(s) under: {root}")
 
    frames = []
    for path in csv_files:
        print(f"  Processing: {path}")
        df = process_csv(path)
        if df is not None:
            frames.append(df)
            print(f"    -> {len(df)} rows kept")
 
    if not frames:
        print("No usable data found after processing.")
        sys.exit(1)
 
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("frame_timestamp").reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    print(f"\nDone. Concatenated {len(frames)} file(s), {len(combined)} total rows -> {output_path}")

if __name__ == "__main__":
    source = "../../../../../data/recordings_with_controls_raw"
    dest = "../../../../../data/dataset_raw.csv"

    # process_directory(source, dest)
    df = process_csv("../../../../../data/dataset_retarded.csv").sort_values("frame_timestamp").reset_index(drop=True)
    df.to_csv("../../../../../data/dataset_retarded.csv", index=False)