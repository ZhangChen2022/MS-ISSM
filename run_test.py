import os
import subprocess
from pathlib import Path

# ================= Configuration =================
# 1. Define dataset and directory paths
BASE_DIR = Path(r".\Dataset")
SRC_DIR = BASE_DIR / "SRC"
PPC_DIR = BASE_DIR / "PPC"
OUTPUT_DIR = Path(r".\Dataset\output")

# 2. Path to the compiled C++ executable
EXE_PATH = Path(r"\ms_feature_cal.exe")
# =================================================

def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate input paths
    if not PPC_DIR.exists() or not SRC_DIR.exists():
        print(f"Error: PPC or SRC directory not found at: {BASE_DIR}")
        return

    # Collect all .ply files from the processed point cloud (PPC) directory
    ppc_files = list(PPC_DIR.glob("*.ply"))
    total_files = len(ppc_files)
    print(f"Found {total_files} distorted point cloud files. Starting processing...\n")

    success_count = 0

    for idx, ppc_file in enumerate(ppc_files, 1):
        # 1. Parse filename to match the reference (SRC) point cloud
        # Example: "p01_noise_level1.ply" -> prefix "p01" -> "p01.ply"
        ppc_filename = ppc_file.name
        src_prefix = ppc_filename.split('_')[0]
        src_filename = f"{src_prefix}.ply"

        # 2. Construct source file path
        src_file = SRC_DIR / src_filename

        # 3. Check if corresponding source file exists
        if not src_file.exists():
            print(f"[{idx}/{total_files}] Warning: Reference {src_filename} not found. Skipping {ppc_filename}")
            continue

        # 4. Construct output CSV path
        csv_filename = f"{ppc_file.stem}.csv"
        csv_file = OUTPUT_DIR / csv_filename

        print(f"[{idx}/{total_files}] Processing: {ppc_filename} -> Ref: {src_filename}")

        # 5. Prepare command line arguments for the C++ executable
        # Arguments mapping: argv[1]=testPath, argv[2]=originPath, argv[3]=outputPath
        command = [
            str(EXE_PATH),
            str(ppc_file),
            str(src_file),
            str(csv_file)
        ]

        try:
            # Execute the C++ feature calculation program
            subprocess.run(command, check=True, text=True, capture_output=True)
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"[{idx}/{total_files}] Error processing {ppc_filename}")
            print(f"Details: {e.stderr}")

    print(f"\nProcessing complete! Successfully processed {success_count}/{total_files} files.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()