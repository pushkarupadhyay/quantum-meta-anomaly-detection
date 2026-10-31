"""
Unified Configuration File for Quantum Meta-RL Anomaly Detection Project
"""

import os

# 1. QUANTUM CIRCUIT CONFIGURATIONS (PQC)
N_QUBITS = 8          # Dimension mapping limit (Number of qubits in the register)
N_LAYERS = 2          # Number of variational layers in the PQC encoder
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10] # Depolarizing noise levels for robustness analysis

# 2. META-REINFORCEMENT LEARNING CONFIGURATIONS (EVOLUTION STRATEGIES)
META_ITERATIONS = 15   # Default meta-training iterations (epochs)
POPULATION_SIZE = 8    # Number of perturbed parameter vectors evaluated per iteration
SIGMA = 0.1            # Perturbation step-size (standard deviation of mutation noise)
LEARNING_RATE = 0.03   # Optimization learning rate for ES update step
K_SUPPORT = 5          # Labelled support samples per class (K-shot adaptation: e.g. 5 normal, 5 anomaly)
Q_QUERY = 30           # Unlabeled query samples per class for episodic reward evaluation

# 3. DATASET MAPPING CONFIGURATIONS (OPENML ID REGISTRY)
DATASET_IDS = {
    "breast_cancer": 15,       # Breast Cancer (UCI) - Healthcare Domain
    "ionosphere": 59,         # Ionosphere (UCI) - Healthcare/Signal Domain
    "credit_card": 1597,       # Credit Card Fraud - Finance Domain
    "nsl_kdd": 1481,           # NSL-KDD - Cybersecurity Domain
    "diabetes": 37,            # Diabetes (UCI) - Generalization Holdout
    "ctg": 1481,               # Cardiotocography (UCI) - Generalization Holdout
    "satimage": 182            # Satimage (OpenML) - Generalization Holdout
}

DEFAULT_TRAIN_DOMAINS = ["breast_cancer", "ionosphere"]
DEFAULT_HOLDOUT_DOMAIN = "satimage"

# 4. PATHS AND DIRECTORY CONFIGURATIONS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
RESULTS_DIR = os.path.join(REPORTS_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Default weight filenames
META_PQC_WEIGHTS_FILE = "meta_pqc_weights.npy"
QUANTUM_BRAIN_WEIGHTS_FILE = "quantum_brain_weights.npy"
