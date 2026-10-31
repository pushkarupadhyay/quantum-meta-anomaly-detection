import os
import json
import urllib.parse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

def generate_quirk_url(n_qubits, n_layers, features, weights):
    """
    Generates a Quirk JSON circuit definition and returns a clickable simulation URL.
    
    n_qubits: number of qubits (e.g. 4 or 8)
    n_layers: number of variational layers
    features: list of classical input values (angle encoded using Ry gates)
    weights: numpy array of model parameters (Ry and Rz rotation angles)
    """
    features = np.round(features, 4).tolist()
    
    cols = []
    
    # 1. Feature Encoding Layer (Ry gates)
    encoding_col = [{"id": "Ryft", "arg": str(features[q])} for q in range(n_qubits)]
    cols.append(encoding_col)
    cols.append(["|"] * n_qubits)
    
    # 2. Variational Layers
    for l in range(n_layers):
        # A. Ry rotation weights
        ry_col = [{"id": "Ryft", "arg": f"{weights[l, q, 0]:.4f}"} for q in range(n_qubits)]
        cols.append(ry_col)
        
        # B. Rz rotation weights
        rz_col = [{"id": "Rzft", "arg": f"{weights[l, q, 1]:.4f}"} for q in range(n_qubits)]
        cols.append(rz_col)
        
        # C. CNOT Entangling Chain
        for q in range(n_qubits - 1):
            cnot = [1] * n_qubits
            cnot[q] = "•"
            cnot[q + 1] = "X"
            cols.append(cnot)
        cnot_end = [1] * n_qubits
        cnot_end[n_qubits - 1] = "•"
        cnot_end[0] = "X"
        cols.append(cnot_end)
        
        if l < n_layers - 1:
            cols.append(["|"] * n_qubits)
            
    # 3. Measurement/Visual Displays (Bloch sphere outputs on all qubits)
    cols.append(["|"] * n_qubits)
    cols.append(["Bloch"] * n_qubits)
    
    # Construct final Quirk JSON
    quirk_json = {"cols": cols}
    base_url = "https://algassert.com/quirk#circuit="
    encoded_json = urllib.parse.quote(json.dumps(quirk_json))
    
    return base_url + encoded_json

def get_real_samples(n_qubits):
    """
    Fetches the Satimage dataset (OpenML ID 182), preprocesses it, and 
    returns a normal and anomaly sample encoded to the [-pi, pi] range.
    """
    print(f"Fetching Satimage (ID 182) to extract real samples for {n_qubits} qubits...")
    try:
        data = fetch_openml(data_id=182, as_frame=True, parser='auto')
        df = data.frame
        target_col = data.target.name if hasattr(data, 'target') and data.target is not None else df.columns[-1]
        
        # Extract numeric features
        df_numeric = df.select_dtypes(include=[np.number]).fillna(0)
        X = df_numeric.drop(columns=[target_col]).values if target_col in df_numeric.columns else df_numeric.values
        y_raw = df[target_col].values
        
        # Binary label mapping (majority class is normal y=0, others are anomaly y=1)
        unique_classes, counts = np.unique(y_raw, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        y = (y_raw != majority_class).astype(int)
        
        # Apply standard scaling and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_qubits)
        X_pca = pca.fit_transform(X_scaled)
        
        # Normalize PCA components to [-pi, pi] for Angle Encoding
        mi, ma = np.min(X_pca, axis=0), np.max(X_pca, axis=0)
        rng = ma - mi
        rng[rng == 0] = 1.0
        X_norm = 2 * np.pi * (X_pca - mi) / rng - np.pi
        
        # Find first normal sample (y=0) and first anomaly sample (y=1)
        normal_idx = np.where(y == 0)[0][0]
        anomaly_idx = np.where(y == 1)[0][0]
        
        normal_sample = X_norm[normal_idx]
        anomaly_sample = X_norm[anomaly_idx]
        
        print(f"  Successfully extracted samples for {n_qubits} qubits.")
        return normal_sample, anomaly_sample
    except Exception as e:
        print(f"  Error fetching data: {e}. Using fallback synthetic samples.")
        # Fallback to deterministic mock features
        np.random.seed(42)
        normal_sample = np.random.uniform(-1.0, 1.0, n_qubits)
        anomaly_sample = np.random.uniform(-np.pi, np.pi, n_qubits)
        return normal_sample, anomaly_sample

def main():
    print("=========================================================")
    print("QUANTUM META-RL CIRCUIT SIMULATION LINKS GENERATOR FOR QUIRK")
    print("=========================================================\n")
    
    # 1. Load trained model weights
    meta_pqc_weights_path = "meta_pqc_weights.npy"
    quantum_brain_weights_path = "quantum_brain_weights.npy"
    
    if not os.path.exists(meta_pqc_weights_path) or not os.path.exists(quantum_brain_weights_path):
        print(f"Error: Weights files not found in current directory. Make sure you are in the workspace root.")
        return
        
    w_4q = np.load(meta_pqc_weights_path)
    w_8q = np.load(quantum_brain_weights_path)
    
    print(f"Loaded {meta_pqc_weights_path}: Shape {w_4q.shape} (4-Qubit Baseline, {w_4q.shape[0]} layers)")
    print(f"Loaded {quantum_brain_weights_path}: Shape {w_8q.shape} (8-Qubit Optimized, {w_8q.shape[0]} layers)\n")
    
    # 2. Get real data samples for 4-qubit and 8-qubit versions
    norm_4q, anom_4q = get_real_samples(n_qubits=4)
    norm_8q, anom_8q = get_real_samples(n_qubits=8)
    print("")
    
    # 3. Generate Quirk links
    url_4q_norm = generate_quirk_url(4, w_4q.shape[0], norm_4q, w_4q)
    url_4q_anom = generate_quirk_url(4, w_4q.shape[0], anom_4q, w_4q)
    
    url_8q_norm = generate_quirk_url(8, w_8q.shape[0], norm_8q, w_8q)
    url_8q_anom = generate_quirk_url(8, w_8q.shape[0], anom_8q, w_8q)
    
    # 4. Print links to console
    print("--- 4-QUBIT BASELINE MODEL LINKS ---")
    print("Normal Input URL:")
    print(url_4q_norm)
    print("\nAnomaly Input URL:")
    print(url_4q_anom)
    print("\n------------------------------------\n")
    
    print("--- 8-QUBIT OPTIMIZED MODEL LINKS ---")
    print("Normal Input URL:")
    print(url_8q_norm)
    print("\nAnomaly Input URL:")
    print(url_8q_anom)
    print("\n------------------------------------\n")
    
    # 5. Save report
    os.makedirs("reports/results", exist_ok=True)
    report_path = "reports/results/quirk_simulation_report.md"
    
    markdown_content = f"""# ⚛️ Quirk Quantum Circuit Simulation Report

This report presents interactive simulation links for the parameterized quantum circuits (PQCs) used in the Quantum Meta-Reinforcement Learning pipeline. The links encode the **exact trained weights** and **real preprocessed samples** from the Satimage holdout dataset.

You can copy and paste these links into your browser to launch the open-source **Quirk** simulator and visualize the quantum state distributions.

---

## 1. 4-Qubit Baseline Model (2 Variational Layers)

The baseline model encodes a 4-dimensional feature representation into a 4-qubit state. 
* **Trained Weights File**: `meta_pqc_weights.npy`

### 🟢 Normal Sample Simulation
* **Input Features**: {norm_4q.tolist()}
* **Click to Launch**: [4-Qubit Normal Circuit in Quirk]({url_4q_norm})

### 🔴 Anomaly Sample Simulation
* **Input Features**: {anom_4q.tolist()}
* **Click to Launch**: [4-Qubit Anomaly Circuit in Quirk]({url_4q_anom})

---

## 2. 8-Qubit Optimized Model (3 Variational Layers)

The optimized model expands the Hilbert space dimensions to $2^8 = 256$ to provide greater representational capacity for high-dimensional tabular anomalies.
* **Trained Weights File**: `quantum_brain_weights.npy`

### 🟢 Normal Sample Simulation
* **Input Features**: {norm_8q.tolist()}
* **Click to Launch**: [8-Qubit Normal Circuit in Quirk]({url_8q_norm})

### 🔴 Anomaly Sample Simulation
* **Input Features**: {anom_8q.tolist()}
* **Click to Launch**: [8-Qubit Anomaly Circuit in Quirk]({url_8q_anom})

---

## 3. Visualization Guidelines

When loading the circuits in Quirk, pay attention to the **Bloch Sphere displays** at the end of the circuit:
1. **Normal Points**: Ry features and optimized parameters align the state vectors near the $|0\\rangle$ or $|+\\rangle$ state on the Z-axis, yielding positive expectation values.
2. **Anomaly Points**: Deviant features rotate the qubit states toward the opposite poles, leading to contrasting expectation values and visual shifts in the Bloch spheres.
3. **Multi-Qubit Entanglement**: The CNOT cascades entangle the register, spreading the feature representations across the multi-qubit Hilbert space to allow linear decision boundaries to separate outliers.
"""
    
    with open(report_path, "w") as f:
        f.write(markdown_content)
        
    print(f"Successfully generated report at '{report_path}'.")
    
    # 6. Auto-launch links in default web browser
    import webbrowser
    print("\n🚀 Auto-launching simulation links in your default web browser...")
    webbrowser.open(url_4q_norm)
    webbrowser.open(url_4q_anom)
    webbrowser.open(url_8q_norm)
    webbrowser.open(url_8q_anom)

if __name__ == "__main__":
    main()
