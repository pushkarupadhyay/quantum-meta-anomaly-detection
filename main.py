import matplotlib.pyplot as plt
import numpy as np
from src.pqc_encoder import PQCEncoder
from src.experiments_cross_domain import MetaExperiment
import os

def plot_pipeline_diagram():
    # Simple ASCII/Matplotlib pipeline diagram
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    pipeline_steps = [
        "Raw Data\n(4 Domains)",
        "Preprocessing\n(Scaler + PCA)",
        "Quantum Encoder\n(PQC)",
        "Few-Shot Head\n(LogReg)",
        "Anomaly Prediction\n(Score/F1)"
    ]
    
    for i, step in enumerate(pipeline_steps):
        ax.text(i * 0.25, 0.5, step, bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black"), 
                ha='center', va='center', fontsize=10)
        if i < len(pipeline_steps) - 1:
            ax.annotate("", xy=((i + 1) * 0.25 - 0.05, 0.5), xytext=(i * 0.25 + 0.05, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=2))
            
    plt.title("Quantum Meta-RL Anomaly Detection Pipeline")
    plt.tight_layout()
    plt.savefig("images/pipeline_diagram.png")
    plt.close()
    print("Pipeline diagram saved to images/pipeline_diagram.png")

def plot_superposition_demo():
    # Show the effect of H-gates or RY gates on probabilities
    # We'll use a 4-qubit system and show the probabilities of all 16 states
    import pennylane as qml
    dev = qml.device("default.qubit", wires=4)
    
    @qml.qnode(dev)
    def superposition_circuit(inputs):
        for i in range(4):
            qml.RY(inputs[i], wires=i)
        return qml.probs(wires=range(4))
    
    inputs = np.array([np.pi/4, np.pi/3, np.pi/2, np.pi/6])
    probs = superposition_circuit(inputs)
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(16), probs, color='purple', alpha=0.7)
    plt.xticks(range(16), [format(i, '04b') for i in range(16)], rotation=45)
    plt.xlabel("Basis State")
    plt.ylabel("Probability")
    plt.title("Quantum Superposition: Probability Distribution across Basis States")
    plt.tight_layout()
    plt.savefig("images/quantum_superposition.png")
    plt.close()
    print("Superposition plot saved to images/quantum_superposition.png")

def run_full_project():
    print("Starting full project execution...")
    
    # 1. Visualization
    plot_pipeline_diagram()
    plot_superposition_demo()
    
    pqc = PQCEncoder(n_qubits=8, n_layers=2)
    pqc.draw_circuit()
    
    # 2. Experiments
    # Using 3 qubits and 15 iterations for a balance of realism and speed
    exp = MetaExperiment(n_qubits=3, iterations=15, population_size=6)
    results = exp.run_leave_one_out()
    print("\nCross-domain results summary:")
    print(results)
    
    noise_results = exp.noise_analysis()
    print("\nNoise analysis summary:")
    print(noise_results)
    
    print("\nProject execution complete. Check 'reports/results' and 'images' folders.")

if __name__ == "__main__":
    run_full_project()
