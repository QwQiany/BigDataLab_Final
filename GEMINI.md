# BigDataLab_Final Project Context

## Project Overview
This project is a comprehensive experiment for a Big Data Processing course ("大数据处理综合实验"), focusing on classifying student data (likely related to GPA or happiness levels). It implements and compares two main categories of machine learning models:
1.  **Hypergraph Neural Networks (HGNN):** Advanced graph-based learning using hypergraphs to model complex high-order correlations between data points (students and their behaviors).
2.  **Traditional Machine Learning Baselines:** Standard classifiers including Random Forest, SVM, Adaboost, Logistic Regression, KNN, and Decision Trees.

The project utilizes **PyTorch** for the deep learning components (HGNN) and **Scikit-Learn** for the traditional machine learning baselines.

## Directory Structure

### Root Directory
Contains the main execution scripts for different models:
*   **Deep Learning Entry Points:**
    *   `HGCN.py`, `HGCN_2.py`: Main scripts to train and evaluate the Hypergraph Convolutional Network.
    *   `GCN.py`: Graph Convolutional Network implementation (likely for comparison).
*   **Traditional ML Entry Points:**
    *   `RF.py`: Random Forest Classifier (supports CLI arguments).
    *   `Adaboost.py`, `DT.py` (Decision Tree), `KNN.py`, `LR.py` (Logistic Regression), `SVM.py`.
    *   `Tabnet.py`: Implementation of TabNet (attention-based network for tabular data).

### `data/`
Contains the datasets used for training and testing:
*   `student_features_min_oversampler.csv`: The primary preprocessed dataset used in `HGCN.py`.
*   `features.csv`, `features_2class.csv`: Feature sets for 3-class and 2-class classification tasks.
*   `train_idx.csv`, `test_idx.csv`: Pre-defined split indices to ensure consistent evaluation.

### `models/`
Contains the PyTorch model definitions:
*   `HGNN.py`: The core Hypergraph Neural Network architecture.
*   `layers.py`: Custom graph convolution layers (`HGNN_conv`).
*   `HGNN+ATT.py`: Variant of HGNN with attention mechanisms.

### `utils/`
*   `hypergraph_utils.py`: Critical utility functions for constructing hypergraphs (incidence matrices) from tabular data, often using KNN (K-Nearest Neighbors) to find correlations.

### `config/`
*   `config.yaml`: Configuration file for hyperparameters (learning rate, epochs, neighbor count `K_neigs`, etc.). *Note: Some scripts like `HGCN.py` may override these settings with hardcoded values.*

## Technical Stack & Dependencies
*   **Language:** Python 3.x
*   **Deep Learning:** PyTorch
*   **Machine Learning:** Scikit-Learn
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib

## Usage Guidelines

### Running Traditional Baselines
Scripts like `RF.py` are designed to be run from the CLI with arguments.
**Example (Random Forest):**
```bash
python RF.py --data_path data/features.csv --dataset_type 3class --n_estimators 100
```
*   `--dataset_type`: `3class` (Poor/Medium/Good) or `2class` (Poor/Good).
*   `--data_path`: Path to the CSV file.

### Running Hypergraph Models (HGNN)
The `HGCN.py` script is the main entry point.
**Example:**
```bash
python HGCN.py
```
*   **Important:** `HGCN.py` currently has hardcoded paths (e.g., `data/student_features_min_oversampler.csv`). Ensure data exists at this location or modify the script/config.
*   It uses `utils.hypergraph_utils` to dynamically construct the hypergraph structure based on feature similarity (KNN).

## Development Conventions
*   **Data Format:** Input data is generally expected to be CSV with the first column being the label (or explicitly separated in the script).
*   **Model Config:** Hyperparameters are partially managed in `config/config.yaml` but check the `_main()` function in scripts as they often define specific training loops and overrides.
*   **Output:** Scripts typically print classification reports (Precision, Recall, F1-score) and accuracy to stdout. Some may generate plots in the `picture/` directory or display them via `matplotlib`.
