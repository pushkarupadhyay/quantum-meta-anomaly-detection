# Hybrid Quantum-Classical Meta-Reinforcement Learning for Few-Shot Anomaly Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-QML-brightgreen.svg)](https://pennylane.ai/)
[![Framework](https://img.shields.io/badge/LangGraph-Orchestration-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end research framework investigating hybrid quantum-classical meta-learning for few-shot anomaly detection across heterogeneous tabular domains. The architecture pairs an 8-qubit Parameterized Quantum Circuit (PQC) encoder, optimized via episodic Evolution Strategies (ES), with an adaptive few-shot classification head to enable cross-domain transfer under strict 5-shot adaptation regimes.

---

## Architecture Overview

```
Tabular Anomaly Datasets (Breast Cancer, Ionosphere, Credit Card, NSL-KDD, etc.)
                          │
                          ▼
            Preprocessing & PCA Alignment
          (StandardScaler + Angle Rescaling [-π, π])
                          │
                          ▼
           Parameterized Quantum Circuit (PQC)
        (8 Qubits, 3 Variational Layers, CNOT Ring)
                          │
                          ▼
           Measurement: Pauli-Z Expectations
                          │
                          ▼
             Adapted Few-Shot Linear Head
         (Support-Set F1-Calibrated Threshold)
                          │
                          ▼
            Evaluation (ROC-AUC & F1-Score)
```

- **Encoder Circuit:** 8-qubit PQC with single-qubit $R_Y$ angle encoding, trainable $R_Y$ and $R_Z$ rotations per layer, closed CNOT entangling chains, and Pauli-$Z$ expectation readouts.
- **Optimization:** Episodic Evolution Strategies (ES) using batch-aligned perturbations to bypass barren plateaus without relying on parameter-shift quantum gradients.
- **Few-Shot Adaptation:** A class-weighted logistic head calibrated dynamically on support-set F1 scores across decision thresholds from 0.01 to 0.99.
- **Workflow Orchestration:** Managed through LangGraph state machines with logging support.

---

## Core Benchmark Results

Evaluation on unseen holdout datasets under a strict 5-shot adaptation constraint ($K=5$ normal, $K=5$ anomalous samples):

### Satimage Benchmark (OpenML ID: 182)

| Model / Framework | Configuration | ROC-AUC | F1-Score |
| :--- | :--- | :---: | :---: |
| **Meta-Trained Quantum PQC (Ours)** | **8 Qubits, 3 Layers + ES** | **0.8122** | **0.8591** |
| Classical Meta-NN Ablation | Multi-Layer Perceptron + ES | 0.5899 | 0.7259 |
| From-Scratch Quantum PQC | 8 Qubits (No Meta-Training) | 0.5173 | 0.5054 |
| Isolation Forest Baseline | Classical Unsupervised | 0.6431 | 0.6289 |
| One-Class SVM Baseline | RBF Kernel | 0.6034 | 0.6727 |
| Quantum One-Class Baseline | QSVDD-inspired Kernel | 0.5312 | 0.4952 |

### Noise Robustness (Depolarizing Noise on Satimage)

| Depolarizing Error Rate | ROC-AUC | F1-Score |
| :---: | :---: | :---: |
| 0.00 (Noiseless) | 0.8122 | 0.8591 |
| 0.01 (1%) | 0.7932 | 0.8421 |
| 0.05 (5%) | 0.7511 | 0.8122 |
| 0.10 (10%) | 0.6909 | 0.7623 |

---

## Repository Structure

```
.
├── streamlit_app.py                # Interactive Streamlit experimentation dashboard
├── test_on_new_data.py             # Holdout-domain evaluation CLI
├── generate_premium_plots.py       # Publication-ready ROC and convergence figures
├── generate_trained_quirk_links.py # Quirk browser simulator URL generators
├── read_scores.py                  # Summaries and parsing for metric objects
├── requirements.txt                # Pinned project dependencies
│
├── src/                            # Core framework source code
│   ├── datasets.py                 # OpenML dataset ingestion and preprocessing
│   ├── pqc_encoder.py              # PennyLane variational circuit implementation
│   ├── fewshot_head.py             # Adaptive thresholding and few-shot heads
│   ├── meta_rl_es.py               # Evolution Strategies meta-learning loop
│   ├── baselines.py                # Classical (IForest, OC-SVM) and quantum baselines
│   ├── experiments_cross_domain.py # Cross-domain benchmarking routines
│   ├── utils_metrics.py            # ROC-AUC, F1, and threshold calibration
│   └── langgraph_workflow.py       # Agentic state machine orchestration
│
├── models/                         # Pre-trained weights (quantum_brain_weights.npy)
├── images/                         # Generated plots, diagrams, and circuit states
└── result_of_simulation/           # Experiment output metrics and evaluation CSVs
```

---

## Installation and Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/pushkarupadhyay/quantum-meta-anomaly-detection.git
cd quantum-meta-anomaly-detection
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv

# On macOS / Linux:
source venv/bin/activate

# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pennylane scikit-learn pandas numpy matplotlib seaborn streamlit openml
```

---

## Quickstart and Usage

### Launch Interactive Streamlit Dashboard
To explore training curves, tune hyperparameters, run holdout evaluations, and view hardware scaling metrics:
```bash
streamlit run streamlit_app.py
```
Access the application at `http://localhost:8501`.

### Evaluate on Unseen Holdout Data
Run evaluation across benchmark datasets (e.g., Satimage, Diabetes, Cardiotocography) using pre-trained weights:
```bash
python test_on_new_data.py
```

### Generate Figures and Quirk Circuit Visualizations
```bash
# Generate ROC curves and convergence plots into images/
python generate_premium_plots.py

# Generate Quirk simulator URLs for state inspections
python generate_trained_quirk_links.py

# Print summarized performance tables
python read_scores.py
```

---

## Pre-trained Model Configuration

The bundled weights file `models/quantum_brain_weights.npy` is shaped `(3, 8, 2)`:
- **Qubits:** 8
- **Variational Layers:** 3
- **Parameters per Qubit per Layer:** 2 ($R_Y$, $R_Z$)

Ensure evaluation parameters match this configuration when testing pre-trained models.

---

## Citation & Academic Report

This work was conducted as an MSc in Artificial Intelligence research practicum. For technical inquiries, contact [Pushkar Upadhyay](https://github.com/pushkarupadhyay).

```bibtex
@article{upadhyay2026hybrid,
  title   = {Hybrid Quantum-Classical Meta-Reinforcement Learning for Few-Shot Anomaly Detection on Tabular Data},
  author  = {Upadhyay, Pushkar},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.