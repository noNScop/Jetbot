import csv
from pathlib import Path


def normalize_ts(ts: str) -> str:
    """Normalize timestamp to exactly 6 decimal digits (round-half-to-even, zero-pad)."""
    return f"{float(ts):.6f}"

def find_matching_image(csv_ts: str, png_files: dict) -> str | None:
    """
    Given a normalized CSV timestamp, find a matching image stem within +/-1
    of the last decimal digit (i.e. +/-0.000001). Returns the matching stem
    or None if no match found.
    """
    ts_float = float(csv_ts)
    # Check exact match first
    if csv_ts in png_files:
        return csv_ts
    # Search within +/-1 in the 6th decimal place (1e-6)
    for stem in png_files:
        if abs(float(stem) - ts_float) <= 1e-6:
            return stem
    return None
 
 
def sync_csv_and_images(csv_path: str, images_dir: str):
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
 
    # Load CSV rows and normalize frame_timestamp to 6 decimal digits
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
 
    for row in rows:
        row["frame_timestamp"] = normalize_ts(row["frame_timestamp"])
 
    # Load all PNGs and map stem -> path
    png_files = {p.stem: p for p in images_dir.glob("*.png")}
 
    print(f"CSV rows: {len(rows)}")
    print(f"Images:   {len(png_files)}")
 
    # Match each CSV row to an image, standardizing both to the CSV timestamp
    matched_images = set()   # image stems that got matched
    rows_to_keep = []
 
    for row in rows:
        csv_ts = row["frame_timestamp"]
        matched_stem = find_matching_image(csv_ts, png_files)
 
        if matched_stem is None:
            print(f"  [drop row] {csv_ts} — no image within ±1e-6")
            continue
 
        # Rename image to match the CSV timestamp if they differ
        if matched_stem != csv_ts:
            old_path = png_files[matched_stem]
            new_path = old_path.with_name(f"{csv_ts}.png")
            old_path.rename(new_path)
            png_files[csv_ts] = new_path
            del png_files[matched_stem]
            print(f"  [renamed] {matched_stem}.png -> {csv_ts}.png")
 
        matched_images.add(csv_ts)
        rows_to_keep.append(row)
 
    # Delete images that had no matching CSV row
    unmatched_images = set(png_files.keys()) - matched_images
    for stem in unmatched_images:
        png_files[stem].unlink()
        print(f"  [deleted image] {stem}.png")
 
    # Rewrite CSV
    rows_to_keep.sort(key=lambda r: float(r["frame_timestamp"]))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_keep)
 
    print(f"\nDone. {len(rows_to_keep)} matched rows, {len(unmatched_images)} images deleted.")


if __name__ == "__main__":
    sync_csv_and_images("/home/noNScop/Desktop/sem6/ROB/Jetbot/data/dataset_raw.csv", "/home/noNScop/Desktop/sem6/ROB/Jetbot/data/dataset_images_raw")