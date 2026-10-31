import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.datasets import fetch_openml

from src.datasets import DatasetManager
from src.pqc_encoder import PQCEncoder
from src.fewshot_head import FewShotHead
from src.meta_rl_es import EvolutionStrategies
from src.utils_metrics import compute_metrics

WEIGHTS_FILE = "quantum_brain_weights.npy"
N_QUBITS = 8
N_LAYERS = 3

def train_brain(iterations=40, population_size=12):
    """
    TRAINING PHASE:
    Exposes the PQC to the full datasets of multiple training domains.
    Updates the PQC parameters so it learns a universal, domain-agnostic feature map.
    """
    print("\n" + "="*60)
    print("PHASE 1: TRAINING THE QUANTUM META-LEARNING BRAIN")
    print("="*60)
    
    # Ingest full datasets for training domains
    dm = DatasetManager(n_qubits=N_QUBITS)
    dm.load_all()
    train_datasets = ['breast_cancer', 'ionosphere', 'credit_card']
    
    print(f"\nInitializing Evolution Strategies (ES) optimizer...")
    es = EvolutionStrategies(params_shape=(N_LAYERS, N_QUBITS, 2), 
                             population_size=population_size, 
                             sigma=0.1, 
                             learning_rate=0.2)
    
    print(f"Beginning meta-training across {train_datasets} for {iterations} iterations...")
    for i in range(iterations):
        population, epsilons = es.sample_population()
        
        # Sample a batch of tasks (one from each domain) for this iteration
        # This ensures all candidates in the population are evaluated on the exact same tasks!
        tasks = []
        for ds_name in train_datasets:
            X_s, y_s, X_q, y_q = dm.sample_episode(ds_name, k_support=5, q_query=20)
            tasks.append((X_s, y_s, X_q, y_q))
            
        rewards = []
        for params in population:
            pqc = PQCEncoder(n_qubits=N_QUBITS, n_layers=N_LAYERS)
            pqc.set_params(params)
            
            task_rewards = []
            for X_s, y_s, X_q, y_q in tasks:
                X_s_pqc = pqc.forward(X_s)
                X_q_pqc = pqc.forward(X_q)
                
                # Simple linear head adaptation
                head = FewShotHead()
                head.fit(X_s_pqc, y_s)
                y_probas = head.predict_proba(X_q_pqc)
                
                metrics = compute_metrics(y_q, y_probas)
                task_rewards.append(metrics['f1'])
            
            # Average reward across the tasks in the batch
            rewards.append(np.mean(task_rewards))
            
        es.update(rewards, epsilons)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Iteration {i+1}/{iterations} - Mean F1 Score: {np.mean(rewards):.4f}")
            
    # Save the trained brain parameter weights
    trained_params = es.get_params()
    np.save(WEIGHTS_FILE, trained_params)
    print("="*60)
    print(f"SUCCESS: The Quantum Brain has finished learning!")
    print(f"Saved weights state to '{WEIGHTS_FILE}'")
    print("="*60)

def test_unseen_data(dataset_id_or_path, is_openml=True, target_col="target"):
    """
    ADAPTATION PHASE:
    Loads the saved weights, maps a new dataset to the quantum space,
    and performs few-shot adaptation to classify anomalies.
    """
    print("\n" + "="*60)
    print("PHASE 2: DETECTING ANOMALIES ON UNSEEN DATA")
    print("="*60)
    
    if not os.path.exists(WEIGHTS_FILE):
        print(f"Error: Weights file '{WEIGHTS_FILE}' not found. Please train the brain first using: --mode train")
        return
        
    # 1. Load the pre-trained weights
    print(f"Loading weights state from '{WEIGHTS_FILE}'...")
    meta_params = np.load(WEIGHTS_FILE)
    
    # 2. Ingest the unseen dataset
    if is_openml:
        print(f"Fetching OpenML ID {dataset_id_or_path}...")
        data = fetch_openml(data_id=dataset_id_or_path, as_frame=True, parser='auto')
        df = data.frame
        target_col = data.target.name if hasattr(data, 'target') and data.target is not None else df.columns[-1]
        
        # Convert non-target category/object columns to float if possible, or label encode them
        for col in df.columns:
            if col != target_col:
                if df[col].dtype.name in ['category', 'object']:
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if converted.isnull().sum() < 0.5 * len(df):
                        df[col] = converted.fillna(0)
                    else:
                        df[col] = df[col].astype('category').cat.codes
                        
        df_numeric = df.select_dtypes(include=[np.number])
        if target_col in df_numeric.columns:
            X_new = df_numeric.drop(columns=[target_col]).values
        else:
            X_new = df_numeric.values
        y_raw = df[target_col].values
        
        # Binary anomaly mapping (minority classes mapped to 1)
        unique_classes, counts = np.unique(y_raw, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        y_new = (y_raw != majority_class).astype(int)
    else:
        print(f"Loading custom CSV file from '{dataset_id_or_path}'...")
        df = pd.read_csv(dataset_id_or_path)
        
        # Convert non-target category/object columns to float if possible, or label encode them
        for col in df.columns:
            if col != target_col:
                if df[col].dtype.name in ['category', 'object']:
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if converted.isnull().sum() < 0.5 * len(df):
                        df[col] = converted.fillna(0)
                    else:
                        df[col] = df[col].astype('category').cat.codes
                        
        X_new = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
        y_new = df[target_col].values
        
    print(f"Loaded dataset containing {X_new.shape[0]} samples with {X_new.shape[1]} features.")
    print(f"  Class breakdown -> Normal (0): {np.sum(y_new == 0)}, Anomaly (1): {np.sum(y_new == 1)}")
    
    # 3. Preprocess features to match the circuit size (N_QUBITS)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_new)
    
    n_components = min(N_QUBITS, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # If the dataset has fewer features than our qubits, pad with zeros
    if X_pca.shape[1] < N_QUBITS:
        padding = np.zeros((X_pca.shape[0], N_QUBITS - X_pca.shape[1]))
        X_pca = np.concatenate([X_pca, padding], axis=1)
        
    # Map coordinates to [-pi, pi] for rotational gates
    min_val, max_val = np.min(X_pca, axis=0), np.max(X_pca, axis=0)
    X_normalized = 2 * np.pi * (X_pca - min_val) / (max_val - min_val + 1e-8) - np.pi
    
    # 4. Extract quantum features using the frozen brain parameters
    print("Running PQC forward pass to extract quantum representation...")
    pqc = PQCEncoder(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    pqc.set_params(meta_params)
    X_quantum = pqc.forward(X_normalized)
    
    # 5. Sample Few-Shot Support Set (5 normal, 5 anomaly)
    normal_idx = np.where(y_new == 0)[0]
    anomaly_idx = np.where(y_new == 1)[0]
    
    k_support = min(5, len(normal_idx), len(anomaly_idx))
    
    # Set seed for reproducible evaluation
    np.random.seed(42)
    support_normal = np.random.choice(normal_idx, k_support, replace=False)
    support_anomaly = np.random.choice(anomaly_idx, k_support, replace=False)
    support_indices = np.concatenate([support_normal, support_anomaly])
    query_indices = np.setdiff1d(np.arange(len(y_new)), support_indices)
    
    X_support = X_quantum[support_indices]
    y_support = y_new[support_indices]
    X_query = X_quantum[query_indices]
    y_query = y_new[query_indices]
    
    # 6. Adapt classification head on support set (No training of PQC weights occurs here)
    print(f"Fitting linear few-shot head on {len(support_indices)} support samples...")
    head = FewShotHead()
    head.fit(X_support, y_support)
    
    # 3. Sweep thresholds to find the best one on the support set
    y_prob_support = head.predict_proba(X_support)
    best_thresholds = []
    best_support_f1 = -1.0
    for thresh in np.linspace(0.01, 0.99, 100):
        y_pred_support = (y_prob_support >= thresh).astype(int)
        f1_s = f1_score(y_support, y_pred_support, zero_division=0)
        if f1_s > best_support_f1:
            best_support_f1 = f1_s
            best_thresholds = [thresh]
        elif f1_s == best_support_f1:
            best_thresholds.append(thresh)
    best_thresh = np.mean(best_thresholds) if best_thresholds else 0.5
    print(f"Optimal threshold found on support set: {best_thresh:.4f} (Support F1: {best_support_f1:.4f})")
    
    # 7. Predict and evaluate
    print("Classifying unseen query samples...")
    y_prob = head.predict_proba(X_query)
    y_pred = (y_prob >= best_thresh).astype(int)
    
    print("\n" + "="*50)
    print("DETECTION RESULTS ON UNSEEN DATA")
    print("="*50)
    print(classification_report(y_query, y_pred, target_names=["Normal", "Anomaly"]))
    
    auc = roc_auc_score(y_query, y_prob)
    f1 = f1_score(y_query, y_pred)
    print(f"ROC AUC Score : {auc:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Meta-Learning Brain CLI Utility")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Choose 'train' to train the brain parameters, or 'test' to run few-shot classification.")
    parser.add_argument("--dataset", type=str, default="37",
                        help="Dataset parameter: OpenML ID (e.g. 37 for Diabetes, 1481 for CTG) or absolute file path to a CSV.")
    parser.add_argument("--target", type=str, default="target",
                        help="Only used for CSV files: The column name representing the binary labels.")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_brain(iterations=40, population_size=12)
    elif args.mode == "test":
        # Check if argument is a file path ending in .csv
        is_openml = not args.dataset.endswith(".csv")
        dataset_param = int(args.dataset) if is_openml else args.dataset
        test_unseen_data(dataset_param, is_openml=is_openml, target_col=args.target)
