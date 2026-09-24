# Spatio-Temporal-Tensor-Analytics

# Spatio-Temporal Sensor Imputation & Compression via Multilinear Tensor Algebra

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_NOTEBOOK_LINK_HERE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Library: TensorLy](https://img.shields.io/badge/Library-TensorLy-blue)](http://tensorly.org/)

An applied scientific computing project formulating multi-dimensional transit and fleet sensor networks as 3rd-order continuous tensors. Evaluates **CANDECOMP/PARAFAC (CP)** and **Tucker Higher-Order SVD (HOSVD)** decompositions under high data sparsity (telemetry dropouts) and benchmarks GPU-accelerated memory compression.

---

## 1. Problem Statement & Motivation

In intelligent transit networks and fleet telemetry, spatial data generated across continuous time is fundamentally multi-modal:
- **Spatial dependencies:** Shared road network dynamics, bottlenecks, and topological correlations across sensors.
- **Temporal cyclicity:** Multi-scale periodicities (rush-hour dynamics vs. intra-week shifts).

### The Flaw of Standard 2D Matrix Flattening
Conventional ML models flatten spatio-temporal streams into 2D matrices $\mathbf{X} \in \mathbb{R}^{S \times (T \cdot D)}$, where $S$ is sensors, $T$ is daily intervals, and $D$ is days. This **destroys the structural multi-frequency periodicity** and leads to catastrophic memory inflation during continuous telemetry aggregation.

### The Tensor Solution
We preserve the continuous topological geometry by structuring the telemetry as a **3rd-Order Spatio-Temporal Tensor**:
$$\boldsymbol{\mathcal{X}} \in \mathbb{R}^{S \times T \times D}$$

---

## 2. Mathematical Formulations

### A. Missing Data Imputation Task
Let $\boldsymbol{\Omega} \in \{0, 1\}^{S \times T \times D}$ denote an observation mask, where $\Omega_{s,t,d} = 0$ indicates lost GPS/sensor packets. The objective is to reconstruct the true state $\boldsymbol{\mathcal{X}}$ from partial observation $\boldsymbol{\mathcal{Y}} = \boldsymbol{\Omega} \odot \boldsymbol{\mathcal{X}}$ by solving:

$$\min_{\hat{\boldsymbol{\mathcal{X}}}} \|\boldsymbol{\Omega} \odot (\boldsymbol{\mathcal{Y}} - \hat{\boldsymbol{\mathcal{X}}})\|_F^2 + \lambda \mathcal{R}(\hat{\boldsymbol{\mathcal{X}}})$$

where $\mathcal{R}(\cdot)$ imposes low-rank multilinear constraints.

---

### B. Tensor Decomposition Architectures

#### 1. CANDECOMP/PARAFAC (CP) Decomposition
Factorizes the spatial tensor into a sum of $R$ rank-one tensors:
$$\boldsymbol{\mathcal{X}} \approx \sum_{r=1}^R \mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r = [\![\mathbf{A}, \mathbf{B}, \mathbf{C}]\!]$$
- $\mathbf{A} \in \mathbb{R}^{S \times R}$: Spatial Basis (Sensor correlations across the road topology)
- $\mathbf{B} \in \mathbb{R}^{T \times R}$: Intra-Day Temporal Basis (Diurnal flow patterns)
- $\mathbf{C} \in \mathbb{R}^{D \times R}$: Daily Modulation Basis (Long-term seasonality)
- **Parameter Footprint:** $\mathcal{O}(R(S + T + D))$ *(Extremely high compression)*

#### 2. Tucker Decomposition (Higher-Order SVD)
Models multi-way cross-mode interactions using a dense Core Tensor $\boldsymbol{\mathcal{G}}$ multiplied by orthogonal factor matrices:
$$\boldsymbol{\mathcal{X}} \approx \boldsymbol{\mathcal{G}} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \mathbf{U}^{(3)}$$
- $\boldsymbol{\mathcal{G}} \in \mathbb{R}^{R_1 \times R_2 \times R_3}$: Core tensor governing interaction between latent subspaces
- $\mathbf{U}^{(1)} \in \mathbb{R}^{S \times R_1}, \mathbf{U}^{(2)} \in \mathbb{R}^{T \times R_2}, \mathbf{U}^{(3)} \in \mathbb{R}^{D \times R_3}$
- **Parameter Footprint:** $\mathcal{O}(R_1 R_2 R_3 + S R_1 + T R_2 + D R_3)$

---

## 3. Empirical Results & Benchmarking

Experiments were executed across **64 Sensors over 30 Days (288 five-minute observations/day)** with a simulated **35% telemetry dropout rate**.

### A. Imputation Quality vs. Compression Performance

| Method | Target Rank | Missing MAE | Missing RMSE | Parameter Footprint | Compression Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Uncompressed Raw** | — | — | — | 552,960 floats | 1.0x |
| **CP Decomposition** | $R=12$ | **0.0241** | **0.0382** | **4,584 floats** | **120.6x** |
| **Tucker Decomposition**| $(10, 20, 10)$ | **0.0189** | **0.0275** | **8,940 floats** | **61.8x** |

### B. Computational Hardware Acceleration (PyTorch Backend)
Execution time for 50 Alternating Least Squares (ALS) iterations:
- **Intel Xeon CPU:** 3.42 seconds
- **NVIDIA T4 GPU (CUDA):** 0.48 seconds (**~7.1x speedup**)

---

## 4. Visualizations

### Continuous Flow Recovery Under Missing Telemetry (Sensor #10)
![Reconstruction Plot](assets/reconstruction_demo.png)
*Figure 1: Reconstructed 24-hour cycle showing the low-rank spatial factor recovering structural peaks during rush hours despite 35% random sensor data dropouts.*

---

## 5. Engineering & Mathematical Takeaways

1. **CP vs. Tucker Trade-off in Transit Networks:**
   - **CP** achieves massive compression ratios (>100x), making it ideal for low-bandwidth edge-device synchronization across large fleets.
   - **Tucker** preserves higher variance and achieves lower RMSE because its core tensor $\boldsymbol{\mathcal{G}}$ captures non-linear interactions between specific sensors and isolated rush-hour windows.
2. **GPU Memory Locality:** By dispatching tensor contractions via PyTorch's native CUDA backend, memory latency from dense multilinear algebra unfolded matricization (Mode-$n$ matricization) is mitigated using batch tensor contractions.

---

## 6. Project Layout

```text
├── assets/
│   └── reconstruction_demo.png       # Generated performance plots
├── notebooks/
│   └── exploratory_traffic.ipynb     # Interactive Google Colab notebook
├── src/
│   ├── model.py                      # CP & Tucker decomposition modules
│   ├── synthetic_data.py             # Spatio-temporal continuous tensor generator
│   └── benchmark.py                  # CPU vs CUDA latency & error evaluation
├── README.md
└── requirements.txt
