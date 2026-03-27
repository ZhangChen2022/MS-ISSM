# MS-ISSM: Multi-scale Implicit Structural Similarity for Point Cloud Quality Assessment

This repository contains the official implementation of the **Multi-scale Implicit Structural Similarity Method (MS-ISSM)** for Point Cloud Quality Assessment (PCQA). 

Our approach bypasses the error-prone point-to-point matching process by representing point cloud features as continuous implicit functions using Radial Basis Functions (RBF). We also introduce a `ResGrouped-MLP` network to robustly map these multi-scale coefficient differences to human perceptual quality scores (MOS).

---

## 📂 Repository Structure

* `main.cpp`: The core C++ algorithm. Extracts multi-scale features (Luma, Chroma, Curvature) and computes their RBF implicit representations.
* `CMakeLists.txt`: Build configuration for the C++ feature extractor.
* `nanoflann.hpp`: A lightweight, header-only library for fast KD-Tree searches (included for your convenience).
* `run_test_org.py` & `run_test.py`: Python wrappers to batch process Reference and Distorted point clouds through the C++ executable.
* `total_csv_to_mat2.py`: Aggregates C++ CSV outputs, computes relative feature differences (preserving signs for Log-Modulus), and packs them into `.mat` tensors.
* `MLP_with_train.py`: The PyTorch backend featuring the `ResGrouped-MLP` network, K-Fold cross-validation, and VQEG standard logistic fitting.

---

## 🚀 Quick Start: Easy Verification (No C++ Required)

We understand that configuring C++ Point Cloud Library (PCL) environments can be time-consuming. **To facilitate immediate verification of our network architecture and algorithm performance, we provide pre-extracted feature and MOS tensors for 4 major public datasets: WPC, SJTU, MPCCD, and ICIP.**

You can completely skip the feature extraction step and directly train/evaluate the network:

1. Ensure the provided `.mat` files are in your working directory (e.g., `WPC_features_tensor.mat` and `WPC_mos_tensor.mat`).
2. Open `MLP_with_train.py` and set your desired dataset at the bottom of the script:
   `DATASET_NAME = 'WPC' # Options: 'WPC', 'SJTU', 'MPCCD', 'ICIP'`
3. Run the training script:
   `python MLP_with_train.py`

The script will perform strict K-Fold cross-validation (preventing data leakage), apply VQEG 4-parameter logistic fitting, and output the final **PLCC, SROCC, KROCC, and RMSE** scores.

---

## 🛠️ Full Pipeline: Feature Extraction to Training

If you wish to run the pipeline on your own custom point clouds, follow these steps sequentially.

### Prerequisites
* **C++ Environment:** CMake (>= 3.10), PCL (Point Cloud Library, >= 1.8), Eigen3. *(Note: `nanoflann.hpp` is already included in this repo).*
* **Python Environment:** Python 3.8+, PyTorch, NumPy, Pandas, SciPy, scikit-learn.

### Step 1: Compile the C++ Feature Extractor

**Windows (Visual Studio):**
Open `x64 Native Tools Command Prompt for VS` and run the following commands:
* `mkdir build`
* `cd build`
* `cmake ..`
* `cmake --build . --config Release`

Move the generated `ms_feature_cal.exe` (from `build\Release\`) to the root directory.

**Linux / Ubuntu:**
Run the following commands in your terminal:
* `mkdir build`
* `cd build`
* `cmake -DCMAKE_BUILD_TYPE=Release ..`
* `make -j4`

Move the compiled `ms_feature_cal` executable to the root directory.

### Step 2: Prepare Your Directory Structure
Organize your raw `.ply` files and your MOS label file as follows:
* `.\Dataset\SRC\` : Pristine/reference point clouds (e.g., p01.ply)
* `.\Dataset\PPC\` : Distorted point clouds (e.g., p01_noise.ply)
* `.\Dataset\mos_labels.csv` : Ground truth subjective scores

### Step 3: Extract Baseline Features (Reference)
Establish the clean baseline by processing the reference point clouds:
* `python run_test_org.py`
*(Outputs will be saved as CSVs in `.\Dataset\output_src\`)*

### Step 4: Extract Distorted Features
Process all distorted point clouds:
* `python run_test.py`
*(Outputs will be saved as CSVs in `.\Dataset\output\`)*

### Step 5: Align and Pack into Tensors
Calculate the relative difference between distorted and reference features, align them with your MOS labels, and generate the final `.mat` tensors:
* `python total_csv_to_mat2.py`
*(Please update the `BASE_DIR` and `MOS_CSV` paths inside the script to match your local setup before running).*

### Step 6: Train and Evaluate
Feed the generated `_features_tensor.mat` and `_mos_tensor.mat` files into the PyTorch network:
* `python MLP_with_train.py`

---

## 📝 Citation
If you find our work useful in your research, please consider citing our paper:
```bibtex
@article{chen2025msissm,
  title={MS-ISSM: Objective Quality Assessment of Point Clouds Using Multi-scale Implicit Structural Similarity},
  author={Chen, Zhang and Wan, Shuai and Zhang, Yuezhe and Ren, Siyu and Yang, Fuzheng and Hou, Junhui},
  year={2025}
}