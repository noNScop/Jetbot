from pathlib import Path
import re
import sys
import shutil
from pathlib import Path

# Change this to your root directory
ROOT_DIR = "/home/noNScop/Desktop/sem6/ROB/Jetbot/data/recordings_with_controls_raw"
OUT_DIR = "/home/noNScop/Desktop/sem6/ROB/Jetbot/data/dataset_images_raw"

def collect_pngs(input_dir: str, output_dir: str):
    root = Path(input_dir)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
 
    png_files = sorted(root.rglob("*.png"))
 
    if not png_files:
        print(f"No PNG files found under: {root}")
        sys.exit(1)
 
    print(f"Found {len(png_files)} PNG file(s) under: {root}")
 
    copied, skipped = 0, 0
    for path in png_files:
        target = dest / path.name
        if target.exists():
            print(f"  [skip] {path.name} — already exists in destination")
            skipped += 1
        else:
            shutil.copy2(path, target)
            copied += 1
 
    print(f"\nDone. Copied {copied} file(s), skipped {skipped} duplicate(s) -> {dest}")

if __name__ == "__main__":
    collect_pngs(ROOT_DIR, OUT_DIR)