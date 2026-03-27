import os
import numpy as np
import pandas as pd
from scipy.io import savemat

# ================= Configuration =================
BASE_DIR = r"F:\study\testOutPut\20260303MS_ISSM\ICIP"
DIR_DISTORTED = os.path.join(BASE_DIR, "output")  # Distorted point cloud CSVs
DIR_REFERENCE = os.path.join(BASE_DIR, "output_src")  # Reference point cloud CSVs
MOS_CSV = os.path.join(BASE_DIR, "ICIP_mos.csv")  # Ground truth MOS file

OUTPUT_CSV = os.path.join(BASE_DIR, "ICIP_features_merged.csv")
OUTPUT_MAT = os.path.join(BASE_DIR, "ICIP_features_tensor.mat")
OUTPUT_MOS_MAT = os.path.join(BASE_DIR, "ICIP_mos_tensor.mat")


# =================================================

def compute_feature_diff_means(ref_path, dist_path):
    """
    Calculates the relative difference between reference and distorted feature tables.
    Returns the mean of 45 feature columns. Rows containing NaN or Inf are dropped.
    """
    try:
        ref_df = pd.read_csv(ref_path)
        dist_df = pd.read_csv(dist_path)
    except Exception as e:
        print(f"  -> [Error] Failed to read files: {e}")
        return None

    if ref_df.shape != dist_df.shape:
        print(f"  -> [Warning] Dimension mismatch: Ref {ref_df.shape} vs Dist {dist_df.shape}")
        return None

    # Convert to numeric and handle invalid values
    ref_df = ref_df.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    dist_df = dist_df.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

    # Align rows by dropping any row that contains NaN in either dataframe
    valid_mask = ~(ref_df.isna().any(axis=1) | dist_df.isna().any(axis=1))
    ref_df = ref_df[valid_mask]
    dist_df = dist_df[valid_mask]

    if ref_df.empty:
        print(f"  -> [Warning] Data empty after filtering NaN/Inf. Skipping.")
        return None

    ref_vals = ref_df.to_numpy(dtype=float)
    dist_vals = dist_df.to_numpy(dtype=float)

    # Calculate relative difference: (Ref - Dist) / max(|Ref|, |Dist|)
    denominator = np.maximum(np.abs(ref_vals), np.abs(dist_vals)) + 1e-5
    diff = (ref_vals - dist_vals) / denominator

    # Calculate column-wise means (Keeping signs for log-modulus transformation in NN)
    return np.mean(diff, axis=0)


def main():
    if not os.path.exists(DIR_DISTORTED) or not os.path.exists(DIR_REFERENCE):
        print(f"Error: Input directories not found in: {BASE_DIR}")
        return

    if not os.path.exists(MOS_CSV):
        print(f"Error: MOS file not found: {MOS_CSV}")
        return

    # 1. Parse MOS Labels into a dictionary for O(1) lookup
    try:
        mos_df = pd.read_csv(MOS_CSV)
        mos_df.columns = mos_df.columns.str.strip().str.lower()

        # Identify filename and score columns automatically
        name_col = next((col for col in mos_df.columns if 'name' in col or 'file' in col), None)
        score_col = next((col for col in mos_df.columns if 'mos' in col or 'score' in col), None)

        if not name_col or not score_col:
            print(f"Error: Could not identify name/score columns. Headers: {mos_df.columns.tolist()}")
            return

        mos_dict = dict(zip(mos_df[name_col], mos_df[score_col]))
        print(f"MOS file loaded. Found {len(mos_dict)} records.\n")
    except Exception as e:
        print(f"Error parsing MOS file: {e}")
        return

    distorted_files = sorted([f for f in os.listdir(DIR_DISTORTED) if f.endswith(".csv")])
    print(f"Found {len(distorted_files)} distorted feature files. Starting alignment...\n")

    results_features = []
    results_mos = []
    filenames = []

    # 2. Extract features and bind with MOS labels
    for idx, dist_file in enumerate(distorted_files, 1):
        name_without_ext = os.path.splitext(dist_file)[0]

        # Match score by full filename or stem
        if dist_file in mos_dict:
            current_mos = mos_dict[dist_file]
        elif name_without_ext in mos_dict:
            current_mos = mos_dict[name_without_ext]
        else:
            print(f"[{idx}/{len(distorted_files)}] Warning: {dist_file} missing score in MOS file. Skipping.")
            continue

        # Match with reference file (e.g., "p01_noise.csv" -> "p01.csv")
        ref_prefix = dist_file.split('_')[0]
        ref_file = f"{ref_prefix}.csv"
        ref_path = os.path.join(DIR_REFERENCE, ref_file)

        if not os.path.exists(ref_path):
            print(f"[{idx}/{len(distorted_files)}] Warning: Reference {ref_file} not found. Skipping.")
            continue

        print(f"[{idx}/{len(distorted_files)}] Processing: {dist_file}")

        col_means = compute_feature_diff_means(ref_path, dist_path := os.path.join(DIR_DISTORTED, dist_file))

        if col_means is not None:
            if len(col_means) != 45:
                print(f"  -> [Warning] Expected 45 features, got {len(col_means)}. Skipping.")
                continue

            filenames.append(dist_file)
            results_features.append(col_means)
            results_mos.append(current_mos)

    if not results_features:
        print("\nNo data aligned. Exiting.")
        return

    # 3. Aggregate results and export
    M = len(results_features)
    print("\nAlignment complete. Generating outputs...")

    # Save merged CSV
    header = ["Filename", "MOS"] + [f"C{i + 1}" for i in range(45)]
    combined_data = np.column_stack((results_mos, results_features))
    df_out = pd.DataFrame(combined_data, columns=header[1:])
    df_out.insert(0, "Filename", filenames)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"1. Merged CSV saved: {OUTPUT_CSV}")

    # Map features to 4D Tensor: (Samples, Params, Scales, Components)
    # Step A: Reshape to internal logic (M, 3 Components, 3 Scales, 5 Params)
    tensor_2d = np.array(results_features, dtype=np.float32)
    tensor_4d_raw = np.reshape(tensor_2d, (M, 3, 3, 5))
    # Step B: Transpose to desired training format (M, 5, 3, 3)
    tensor_4d = np.transpose(tensor_4d_raw, (0, 3, 2, 1))

    savemat(OUTPUT_MAT, {'components': tensor_4d})
    print(f"2. Feature tensor saved: {OUTPUT_MAT} | Shape: {tensor_4d.shape}")

    # Save MOS labels as tensor (M, 1)
    mos_array = np.array(results_mos, dtype=np.float32).reshape(-1, 1)
    savemat(OUTPUT_MOS_MAT, {'mos': mos_array})
    print(f"3. MOS tensor saved: {OUTPUT_MOS_MAT} | Shape: {mos_array.shape}")

    print(f"\n[Done] Total valid samples: {M}")


if __name__ == "__main__":
    main()