import os
import subprocess
from pathlib import Path

# ================= Configuration =================
# Directory containing the source (reference) point clouds
BASE_DIR = Path(r".\Dataset")
SRC_DIR = BASE_DIR / "SRC"

# Separate output directory for source point cloud features to avoid confusion
OUTPUT_DIR = Path(r".\Dataset\output_src")

# Path to the compiled C++ executable
EXE_PATH = Path(r".\ms_feature_cal.exe")

# =================================================

def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SRC_DIR.exists():
        print(f"Error: SRC directory not found at: {SRC_DIR}")
        return

    # Collect all .ply files in the SRC directory
    src_files = list(SRC_DIR.glob("*.ply"))
    total_files = len(src_files)
    print(f"Found {total_files} reference point cloud files. Starting processing...\n")

    success_count = 0

    for idx, src_file in enumerate(src_files, 1):
        filename = src_file.name
        csv_filename = f"{src_file.stem}.csv"
        csv_file = OUTPUT_DIR / csv_filename

        print(f"[{idx}/{total_files}] Processing reference: {filename}")

        # Execute command: passing the same SRC file as both test and reference
        # to establish a baseline (Self-comparison)
        command = [
            str(EXE_PATH),
            str(src_file),  # Path for test data
            str(src_file),  # Path for reference data
            str(csv_file)  # Output CSV path
        ]

        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            success_count += 1

            # Log specific performance metrics from C++ output if available
            for line in result.stdout.splitlines():
                if "Algorithm processing time" in line:
                    print(f"    -> {line}")
        except subprocess.CalledProcessError as e:
            print(f"[{idx}/{total_files}] Runtime error for: {filename}")
            print(f"Error details: {e.stderr}")

    print(f"\nProcessing complete! Successfully processed {success_count}/{total_files} files.")
    print(f"Baseline results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()