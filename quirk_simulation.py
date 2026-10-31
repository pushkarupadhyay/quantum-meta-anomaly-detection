import json
import urllib.parse
import numpy as np

def generate_quirk_circuit(n_qubits=4, n_layers=1, features=None, weights=None):
    """
    Generates a Quirk JSON circuit definition and returns a clickable simulation URL.
    
    n_qubits: number of qubits (e.g. 4 or 8)
    n_layers: number of variational layers
    features: list of classical input values (angle encoded using Ry gates)
    weights: model parameters (Ry and Rz rotation angles)
    """
    if features is None:
        # Default mock features scaled to [-pi, pi]
        features = np.round(np.random.uniform(-np.pi, np.pi, n_qubits), 3).tolist()
    else:
        features = np.round(features, 3).tolist()
        
    if weights is None:
        # Default mock weights
        weights = np.round(np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 2)), 3).tolist()
    
    cols = []
    
    # 1. Feature Encoding Layer (Ry gates)
    encoding_col = []
    for q in range(n_qubits):
        encoding_col.append({"id": "Ryft", "arg": str(features[q])})
    cols.append(encoding_col)
    
    # Divider to visualize encoding separation
    cols.append(["|"] * n_qubits)
    
    # 2. Variational Layers
    for l in range(n_layers):
        # A. Ry rotation weights
        ry_col = []
        for q in range(n_qubits):
            ry_col.append({"id": "Ryft", "arg": str(weights[l][q][0])})
        cols.append(ry_col)
        
        # B. Rz rotation weights
        rz_col = []
        for q in range(n_qubits):
            rz_col.append({"id": "Rzft", "arg": str(weights[l][q][1])})
        cols.append(rz_col)
        
        # C. CNOT Entangling Chain
        for q in range(n_qubits - 1):
            cnot = [1] * n_qubits
            cnot[q] = "•"
            cnot[q + 1] = "X"
            cols.append(cnot)
        # End CNOT loop connection
        cnot_end = [1] * n_qubits
        cnot_end[n_qubits - 1] = "•"
        cnot_end[0] = "X"
        cols.append(cnot_end)
        
        # Divider between layers
        if l < n_layers - 1:
            cols.append(["|"] * n_qubits)
            
    # 3. Measurement/Visual Displays (Bloch sphere outputs on all qubits)
    cols.append(["|"] * n_qubits)
    display_col = ["Bloch"] * n_qubits
    cols.append(display_col)
    
    # Construct final Quirk JSON
    quirk_json = {"cols": cols}
    
    # Generate URL
    base_url = "https://algassert.com/quirk#circuit="
    encoded_json = urllib.parse.quote(json.dumps(quirk_json))
    
    return base_url + encoded_json

def main():
    print("--- Testing Quirk Quantum Circuit Simulator Export ---")
    
    # 1. Test 4-Qubit Circuit
    features_4 = [0.5, -1.2, 0.8, 1.5]
    weights_4 = [[[1.2, -0.5], [0.8, 1.4], [-0.9, 0.3], [0.4, -0.7]]]
    url_4 = generate_quirk_circuit(n_qubits=4, n_layers=1, features=features_4, weights=weights_4)
    print("\n🚀 4-Qubit Quirk Simulation URL:")
    print(url_4)
    
    # 2. Test 8-Qubit Circuit (Project Standard)
    features_8 = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
    weights_8 = [
        [[0.5, -0.5]] * 8, # Layer 1
        [[0.2, 0.8]] * 8   # Layer 2
    ]
    url_8 = generate_quirk_circuit(n_qubits=8, n_layers=2, features=features_8, weights=weights_8)
    print("\n🚀 8-Qubit Quirk Simulation URL:")
    print(url_8)
    
    # Save the links locally for inspection
    with open("reports/quirk_simulation_links.txt", "w") as f:
        f.write("4-Qubit Simulation Link:\n" + url_4 + "\n\n" + "8-Qubit Simulation Link:\n" + url_8 + "\n")
    print("\nLinks saved to 'reports/quirk_simulation_links.txt'")

if __name__ == "__main__":
    main()
