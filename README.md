# Machine Learning & Deep Learning From Scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/built%20with-NumPy-013243.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive educational repository containing implementations of fundamental **Machine Learning** and **Deep Learning** algorithms from scratch using **pure NumPy and Python**. 

No high-level frameworks (PyTorch, TensorFlow, or Keras) are used for model forward/backward computations. Every layer, loss function, optimization step, and backpropagation pass is derived directly from mathematical first principles.

---

## 🌟 Key Highlights

- **Pure NumPy Implementation**: All core mathematical transformations and tensor calculations are built with vectorized matrix operations.
- **Manual Backpropagation**: Explicit analytical gradient derivations for all layers, including Convolutions, Multi-Head Attention, and Recurrent layers.
- **Numerical Gradient Checking**: Finite-difference gradient verification ensures mathematical correctness of custom backpropagation routines (e.g. MHA relative error $< 10^{-5}$).
- **Modular & Readable**: Clean, self-contained implementations designed for deep understanding and study.

---

## 📂 Project Structure

```text
numpy/
├── Dl_FromScratch/                  # Deep Learning Architectures
│   ├── Transformer/                 # Multi-Head Attention (MHA) & Self-Attention
│   │   └── MHA.ipynb
│   ├── Cnnfs/                       # Convolutional Neural Networks
│   │   ├── dogs_and_cats/           # CNN on Cats vs Dogs
│   │   └── mnist/                   # CNN on MNIST handwritten digits
│   ├── Rnnfs/                       # Recurrent Neural Networks
│   │   └── rnnfs.ipynb
│   └── nnfs/                        # Multi-Layer Perceptrons & Modular NN Components
│       ├── my_ver.py
│       └── nn from scratch.ipynb
│
└── ml_FromScratch/                  # Classic Machine Learning Algorithms
    ├── decision_tree/               # Decision Tree (Entropy, Information Gain)
    ├── random_frs/                  # Random Forest (Ensemble & Bootstrap Aggregation)
    ├── pca/                         # Principal Component Analysis (Covariance & Eigendecomposition)
    ├── k_mean_clustering/           # K-Means Clustering
    ├── knn/                         # K-Nearest Neighbors
    ├── svm/                         # Support Vector Machine (Subgradient Descent)
    ├── linear_regression/           # Linear Regression (MSE & Gradient Descent)
    └── logistic_regression/         # Logistic Regression (Sigmoid & Binary Cross-Entropy)
```

---

## 🔬 Implemented Algorithms & Architectures

### 1. Deep Learning (`Dl_FromScratch/`)

| Model / Component | Description | Highlights |
| :--- | :--- | :--- |
| **Multi-Head Attention** | Scaled Dot-Product Self-Attention | Manual gradient derivation for $W_q, W_k, W_v, W_o$, masked attention, and finite-difference gradient check. |
| **Convolutional Neural Network** | 2D Conv & Pooling Layers | Custom 2D convolution forward/backward, max pooling, flattening, and multi-class classification. |
| **Recurrent Neural Network** | Sequence modeling & generation | Hidden state recurrence, BPTT (Backpropagation Through Time), next-word prediction. |
| **Modular Neural Network** | Dense Layers & Activations | Dense (Linear), ReLU, Sigmoid, Softmax with Categorical Cross-Entropy, SGD with Momentum. |

### 2. Machine Learning (`ml_FromScratch/`)

| Algorithm | Method / Formulation | Key Concepts |
| :--- | :--- | :--- |
| **Decision Tree** | Classification Tree | Shannon Entropy, Information Gain, recursive best-split search. |
| **Random Forest** | Ensemble Bagging | Bootstrap sampling of data, majority voting, variance reduction. |
| **PCA** | Dimensionality Reduction | Data centering, sample covariance matrix, eigenvalue & eigenvector decomposition. |
| **K-Means Clustering** | Unsupervised Clustering | Random centroid initialization, Euclidean distance assignment, convergence loop. |
| **Support Vector Machine** | Linear SVM Classifier | Hinge loss objective, subgradient descent, $L_2$ regularization penalty. |
| **K-Nearest Neighbors** | Non-parametric Classifier | Vectorized Euclidean distance, majority voting. |
| **Linear Regression** | Ordinary Least Squares | Mean Squared Error (MSE), gradient descent optimization. |
| **Logistic Regression** | Binary Classification | Sigmoid activation, log loss (cross-entropy), gradient descent. |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. Install the minimal dependencies:

```bash
pip install numpy pandas matplotlib opencv-python
```

### Running an Example

#### 1. Machine Learning Scripts (Python)
Run any standalone algorithm directly from the terminal:

```bash
# Principal Component Analysis (PCA)
python ml_FromScratch/pca/pca.py

# Decision Tree Classification
python ml_FromScratch/decision_tree/decision_tree.py

# Random Forest
python ml_FromScratch/random_frs/random_forest.py

# K-Means Clustering
python ml_FromScratch/k_mean_clustering/k_mean.py
```

#### 2. Deep Learning Experiments (Jupyter Notebooks)
Launch Jupyter Notebook or VS Code to explore interactive notebooks:

```bash
jupyter notebook
```
* Navigate to `Dl_FromScratch/Transformer/MHA.ipynb` to inspect the Multi-Head Attention implementation and numerical gradient checks.
* Navigate to `Dl_FromScratch/Cnnfs/` or `Dl_FromScratch/Rnnfs/` to inspect CNN and RNN training routines.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

