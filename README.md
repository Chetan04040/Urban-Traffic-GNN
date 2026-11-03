# 🚦 Urban Traffic Flow Prediction Using Graph Neural Networks (GNNs)

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/PyG-torch__geometric-3C2179?style=for-the-badge&logo=python&logoColor=white" alt="PyG" />
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License" />
  <img src="https://img.shields.io/badge/Tests-Passing%20(7%2F7)-brightgreen?style=for-the-badge" alt="Tests" />
  <img src="https://img.shields.io/badge/Domain-Smart%20Cities%20%26%20ITS-00E5FF?style=for-the-badge" alt="Smart Cities" />
</p>

---

## 📌 Executive Summary & Novelty

Modern metropolitan road networks are **non-Euclidean graph topologies** where intersections and traffic sensors are nodes, and roadways are directed edges. 

* **Why CNNs Fail**: Standard 2D CNNs assume a regular grid with uniform 8-neighbor adjacency. In a city, two sensors 500 meters apart straight-line may sit on opposite sides of a divided highway or river with no direct connection.
* **Why LSTMs Fail**: Standard RNNs/LSTMs process temporal histories of sensors independently. When a bottleneck or traffic incident occurs downstream, an LSTM is spatially blind and cannot predict upstream shockwave delays until the congestion physically reaches that sensor.
* **The Real-World Imbalance Challenge**: Real-world traffic speed datasets (e.g., California PeMS / METR-LA) are **heavily class-imbalanced**:
  * ~**82.9% Free-Flow Traffic** ($\ge 50$ mph)
  * ~**11.8% Moderate Slowdown** ($30-50$ mph)
  * ~**5.3% Severe Congestion & Shockwave Bottlenecks** ($< 30$ mph)  
  Under standard MSE loss, standard deep learning models suffer from **regression-to-the-mean**—they accurately predict free-flow speeds but catastrophically underestimate traffic jams and shockwaves.

---

## 🏗️ Architectural Evolution

This project implements a complete comparative deep learning progression:

```
+-----------------------------------------------------------------------------------------+
|                                1. Baseline Model (LSTM)                                 |
|                                                                                         |
|  [X_t-11 ... X_t] (Nodes flattened) ---> Bi-LSTM ---> Linear Regressor ---> Multi-Step  |
|  * Limitation: Completely blind to spatial road graph connectivity                      |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|                               2. Base Model (ST-GCN)                                    |
|                                                                                         |
|  Temporal 1D Conv ---> Spatial Graph Conv (Normalized Laplacian) ---> Output Head        |
|  * Uses physical road adjacency matrix: A_hat = D^(-0.5) * A * D^(-0.5)                 |
|  * Limitation: Evaluated with standard MSE; heavily biased towards free-flow speeds    |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|                     3. Improved Model: Attention ST-GCN (A-STGCN)                       |
|                                                                                         |
|  Feature Projection ---> Adaptive Graph Attention Conv ---> Bi-GRU ---> Multi-Head     |
|  Temporal Self-Attention ---> Congestion-Focal Loss Head                                 |
|                                                                                         |
|  * Improvement 1: Dynamic Latent Topology Discovery (A_total = alpha*A_phys + (1-alpha)*A_adapt)
|  * Improvement 2: Multi-Head Temporal Self-Attention over rush hour transition steps   |
|  * Improvement 3: Congestion-Focal Loss heavily penalizes bottleneck under-prediction    |
+-----------------------------------------------------------------------------------------+
```

---

## 📊 Empirical Benchmark Results

Evaluated on the 25-node highway network test split across 3-step forecasting horizons (15 to 45 mins ahead):

| Model Architecture | Spatial Graph Awareness | Overall MAE | RMSE | Severe Jam MAE (<30 mph) | Jam Recall (%) | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline LSTM** | ❌ Blind (Independent) | 1.569 mph | 2.061 | 1.999 mph | 94.2% | 0.942 |
| **Base ST-GCN** | ✓ Static Road Laplacian | 1.600 mph | 2.117 | 2.626 mph | 87.5% | 0.920 |
| **Improved A-STGCN** *(Ours)* | **✓ Adaptive Attn + Focal Loss** | **1.497 mph** | **1.945** | **1.852 mph** | **92.6%** | **0.942** |

> **Key Finding**: Base ST-GCN with standard MSE loss had an error of **2.626 mph** during severe traffic jams because the majority free-flow class dominated the gradient. Our **Improved A-STGCN with Congestion-Focal Loss reduced severe congestion error down to 1.852 mph (a 29.5% error reduction)** while simultaneously improving overall network RMSE to 1.945!

---

## 🎛️ Interactive Full-Stack Presentation & Simulation Dashboard

The project includes a built-in full-stack web application featuring dual modes:
* **🌐 Live Highway Simulator**: Real-time road graph canvas with 24-hour time-scrubber, animated traffic shockwaves, and interactive node inspector comparing all 3 models against ground truth.
* **🎤 Presentation Deck Mode**: 6-slide presentation deck designed for academic defense, complete with keyboard controls (`←` / `→` arrows), theoretical comparisons, loss formulation, and interactive viva defense accordion cards.

```bash
# Launch the dashboard locally:
python3 app.py
# Open: http://localhost:8080
```

---

## 🚀 Quick Start & How to Run

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Chetan04040/Urban-Traffic-GNN.git
cd Urban-Traffic-GNN
pip install -r requirements.txt
```

### 2. Run the Automated Test Suite (7/7 Passing)
To verify tensor shapes, gradient flow, and graph Laplacian normalization:
```bash
python3 test_models.py
```

### 3. Re-train & Benchmark Models
To rerun dataset generation, training epochs, and plot generation:
```bash
python3 train_and_compare.py
```

### 4. Single-Command Master Runner
```bash
# Run tests, train models, and launch web dashboard:
python3 run_pipeline.py --all
```

---

## 📁 Repository Structure

```
Urban_Traffic_GNN/
├── app.py                       # Full-stack HTTP server & REST API
├── run_pipeline.py              # Master orchestration CLI script
├── train_and_compare.py         # End-to-end training and comparative evaluation
├── test_models.py               # Unit & integration test suite (7 tests)
├── VIVA_DEFENSE_GUIDE.md        # Comprehensive oral defense & theory Q&A guide
├── requirements.txt             # Project dependencies
├── LICENSE                      # MIT License
├── CITATION.cff                 # Academic citation metadata
├── README.md                    # Project documentation
├── models/
│   ├── baseline_lstm.py         # Pure Temporal baseline (no graph)
│   ├── base_stgcn.py            # Standard ST-GCN (Kipf GCN + 1D Conv)
│   ├── improved_stgcn.py        # Improved A-STGCN + Adaptive Topology + Focal Loss
│   ├── Baseline_LSTM.pt         # Saved PyTorch checkpoint
│   ├── Base_STGCN.pt            # Saved PyTorch checkpoint
│   └── Improved_A_STGCN.pt      # Saved PyTorch checkpoint
├── src/
│   ├── dataset.py               # Highway graph builder & imbalanced traffic generator
│   └── metrics.py               # MAE, RMSE, MAPE, and Congestion Regime metrics
├── data/
│   ├── metadata.json            # Graph coordinates & imbalance statistics
│   ├── adj_matrix.npy           # Distance-decay Gaussian adjacency matrix
│   └── traffic_features.npy     # Sensor time-series features
├── results/
│   ├── comparative_results.json # Full benchmark metrics breakdown
│   ├── traffic_comparison.png   # 4-panel publication-grade evaluation plot
│   └── ground_truth_test.npy    # Test split arrays
└── web_app/
    ├── index.html               # Responsive dark dashboard & presentation deck
    ├── style.css                # Transportation cyber theme styling
    └── app.js                   # Canvas road graph visualizer & Chart.js engine
```

---

## 📚 Viva Voce & Oral Defense Guide

A complete viva voce preparation guide is included in [`VIVA_DEFENSE_GUIDE.md`](VIVA_DEFENSE_GUIDE.md). It covers:
1. **GNN vs. LSTM**: Why temporal sequences fail without topological message passing.
2. **Graph Laplacian Mathematics**: Why $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ symmetric normalization is necessary.
3. **Imbalance Resolution**: How Congestion-Focal Loss eliminates regression-to-the-mean.
4. **Megacity Scalability**: Cluster-GCN and GraphSAGE neighborhood sampling for 20,000+ node networks.

---

## 📄 Citation

```bibtex
@software{rupchandani2026urbantrafficgnn,
  author = {Rupchandani, Chetan},
  title = {Urban Traffic Flow Prediction Using Graph Neural Networks (GNNs)},
  url = {https://github.com/Chetan04040/Urban-Traffic-GNN},
  year = {2026}
}
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
