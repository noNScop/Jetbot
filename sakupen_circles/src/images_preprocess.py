from pathlib import Path
import re
import sys
import shutil
from pathlib import Path

# Change this to your root directory
ROOT_DIR = "/home/noNScop/Desktop/sem6/ROB/Jetbot/data/recordings_with_controls_raw"
OUT_DIR = "/home/noNScop/Desktop/sem6/ROB/Jetbot/data/dataset_images_raw"

def rename():
    pattern = re.compile(r"^\d+_(\d+)\.(\d+)\.png$")

    for file_path in Path(ROOT_DIR).rglob("*.png"):
        match = pattern.match(file_path.name)

        if match:
            time1 = match.group(1)
            time2 = match.group(2)

            # Keep only first 2 digits of time2
            new_name = f"{time1}.{time2}.png"

            new_path = file_path.with_name(new_name)

            print(f"{file_path.name} -> {new_name}")

            file_path.rename(new_path)

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
    rename()
    collect_pngs(ROOT_DIR, OUT_DIR)