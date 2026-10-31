import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import os

# Create images folder if not exists
os.makedirs("images", exist_ok=True)


# 1. QUANTUM EFFECTS: DEMONSTRATION


print("="*60)
print("1. DEMONSTRATING CORE QUANTUM EFFECTS")
print("="*60)

# --- QISKIT ---
print("\n--- Qiskit Implementation ---")
# 1.1 Superposition and Entanglement (Bell State |Phi+>)
qc_bell = QuantumCircuit(2)
qc_bell.h(0)       # Superposition
qc_bell.cx(0, 1)   # Entanglement
sv_bell = Statevector.from_instruction(qc_bell)
print("Bell Circuit Diagram:")
print(qc_bell.draw(output='text'))
print("Statevector:", sv_bell.data)
print("State probabilities:", np.round(np.abs(sv_bell.data)**2, 4))

# 1.2 Quantum Interference
# We apply: H -> RZ(phi) -> H
# At phi = 0, constructive interference returns state to |0>
# At phi = pi, destructive interference shifts state to |1>
def qiskit_interference(phi):
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(phi, 0)
    qc.h(0)
    sv = Statevector.from_instruction(qc)
    prob_0 = np.abs(sv.data[0])**2
    return prob_0, sv.data

prob_0_qiskit_0, sv_0 = qiskit_interference(0)
prob_0_qiskit_pi, sv_pi = qiskit_interference(np.pi)
print("\nInterference at phi = 0 (constructive for |0>):")
print("  Prob(|0>):", prob_0_qiskit_0, "State:", sv_0)
print("Interference at phi = pi (destructive for |0>, constructive for |1>):")
print("  Prob(|0>):", np.round(prob_0_qiskit_pi, 4), "State:", sv_pi)


# --- PENNYLANE ---
print("\n--- PennyLane Implementation ---")
# 2.1 Superposition and Entanglement
dev_bell = qml.device("default.qubit", wires=2)
@qml.qnode(dev_bell)
def pennylane_bell():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

pl_sv_bell = pennylane_bell()
print("Statevector:", pl_sv_bell)

# 2.2 Quantum Interference
dev_interf = qml.device("default.qubit", wires=1)
@qml.qnode(dev_interf)
def pennylane_interference_circuit(phi):
    qml.Hadamard(wires=0)
    qml.RZ(phi, wires=0)
    qml.Hadamard(wires=0)
    return qml.probs(wires=0)

print("\nInterference at phi = 0 (PennyLane):")
print("  Probs [|0>, |1>]:", pennylane_interference_circuit(0))
print("Interference at phi = pi (PennyLane):")
print("  Probs [|0>, |1>]:", np.round(pennylane_interference_circuit(np.pi), 4))


# Plotting the Interference Pattern to show the wave effect
phases = np.linspace(0, 2 * np.pi, 100)
probs_qiskit = [qiskit_interference(phi)[0] for phi in phases]
probs_pennylane = [pennylane_interference_circuit(phi)[0] for phi in phases]

plt.figure(figsize=(10, 5))
plt.plot(phases, probs_qiskit, 'b-', label='Qiskit Prob(|0>)')
plt.plot(phases, probs_pennylane, 'r--', label='PennyLane Prob(|0>)')
plt.axvline(x=0, color='gray', linestyle=':')
plt.axvline(x=np.pi, color='green', linestyle=':', label='phi = pi (Destructive)')
plt.axvline(x=2*np.pi, color='gray', linestyle=':')
plt.xlabel("Phase Offset phi (radians)")
plt.ylabel("Probability of measuring State |0>")
plt.title("Quantum Interference Waveform: H -> RZ(phi) -> H")
plt.legend()
plt.grid(True)
plt.savefig("images/quantum_interference_pattern.png")
plt.close()
print("\nInterference waveform plot saved to 'images/quantum_interference_pattern.png'")


# ==========================================
# 2. MULTIPLE FEATURE ENCODING TECHNIQUES
# ==========================================

print("\n" + "="*60)
print("2. ENCODING MULTIPLE FEATURES")
print("="*60)

# Sample Multi-Feature Data
features = np.array([0.5, -1.2, 0.8, 1.5])
print(f"Classical features to encode: {features}")

# 2.1 BASIS ENCODING (For Binary/Categorical Features)
# Encodes discrete binary state. E.g., [1, 0, 1, 0] maps to state |1010>
binary_features = np.array([1, 0, 1, 0])
print(f"\nDiscrete features for Basis Encoding: {binary_features}")

# Qiskit Basis Encoding
def qiskit_basis_encode(binary_arr):
    qc = QuantumCircuit(len(binary_arr))
    for i, bit in enumerate(binary_arr):
        if bit == 1:
            qc.x(i)
    sv = Statevector.from_instruction(qc)
    return qc, sv.data

qc_basis, sv_basis = qiskit_basis_encode(binary_features)
print("[Qiskit] Basis encoded circuit statevector:")
print("  Active Basis State index:", np.argmax(np.abs(sv_basis)**2), "Statevector:", sv_basis)

# PennyLane Basis Encoding
dev_basis = qml.device("default.qubit", wires=len(binary_features))
@qml.qnode(dev_basis)
def pennylane_basis_encode(binary_arr):
    qml.BasisEmbedding(binary_arr, wires=range(len(binary_arr)))
    return qml.state()

pl_sv_basis = pennylane_basis_encode(binary_features)
print("[PennyLane] Basis encoded statevector:")
print("  Active Basis State index:", np.argmax(np.abs(pl_sv_basis)**2), "Statevector:", pl_sv_basis)


# 2.2 ANGLE ENCODING
# Encodes N features into N qubits using single-qubit rotations
# Here features act as rotation angles.

# Qiskit Angle Encoding
def qiskit_angle_encode(feats, rot='y'):
    n_q = len(feats)
    qc = QuantumCircuit(n_q)
    for i, val in enumerate(feats):
        if rot == 'x':
            qc.rx(val, i)
        elif rot == 'y':
            qc.ry(val, i)
        elif rot == 'z':
            qc.rz(val, i)
    sv = Statevector.from_instruction(qc)
    return qc, sv.data

qc_angle, sv_angle = qiskit_angle_encode(features, rot='y')
print("\n[Qiskit] Angle Encoding (RY) Statevector (first 4 amplitudes):")
print(" ", sv_angle[:4])

# PennyLane Angle Encoding
dev_angle = qml.device("default.qubit", wires=len(features))
@qml.qnode(dev_angle)
def pennylane_angle_encode(feats):
    qml.AngleEmbedding(feats, wires=range(len(feats)), rotation='Y')
    return qml.state()

pl_sv_angle = pennylane_angle_encode(features)
print("[PennyLane] Angle Encoding (RY) Statevector (first 4 amplitudes):")
print(" ", pl_sv_angle[:4])


# 2.3 AMPLITUDE ENCODING
# Encodes N features into log2(N) qubits.
# Highly qubit efficient, but mathematically requires the features to represent a normalized statevector.
norm_features = features / np.linalg.norm(features)
print(f"\nNormalized features for Amplitude Encoding: {norm_features}")

# Qiskit Amplitude Encoding
def qiskit_amplitude_encode(feats):
    n_q = int(np.ceil(np.log2(len(feats))))
    qc = QuantumCircuit(n_q)
    qc.initialize(feats, range(n_q))
    sv = Statevector.from_instruction(qc)
    return qc, sv.data

qc_amp, sv_amp = qiskit_amplitude_encode(norm_features)
print("[Qiskit] Amplitude encoded statevector (2 qubits):")
print(" ", sv_amp)

# PennyLane Amplitude Encoding
dev_amp = qml.device("default.qubit", wires=int(np.log2(len(features))))
@qml.qnode(dev_amp)
def pennylane_amplitude_encode(feats):
    qml.AmplitudeEmbedding(feats, wires=range(int(np.log2(len(feats)))), normalize=True)
    return qml.state()

pl_sv_amp = pennylane_amplitude_encode(features)
print("[PennyLane] Amplitude encoded statevector (2 qubits):")
print(" ", pl_sv_amp)



# 3. ENCODING TRADE-OFFS ANALYSIS


print("\n" + "="*60)
print("3. ENCODING TRADE-OFFS & COMPLEXITY")
print("="*60)

trade_offs = """
| Encoding Method    | Qubits Required      | Circuit Depth      | Gate Complexity   | Best Use Case                         |
| :----------------- | :------------------- | :----------------- | :---------------- | :------------------------------------ |
| **Basis Encoding**  | N (1 per bit)        | O(1)               | O(N) (X gates)    | Binary/Categorical features           |
| **Angle Encoding**  | N (1 per feature)    | O(1) (Parallel)    | O(N) (1-Q rotations)| Low-dimensional continuous features   |
| **Amplitude**       | log2(N) (Logarithmic)| O(N) (Deep)        | O(2^n) (Multiplex) | High-dimensional dense feature vectors |
"""
print(trade_offs)



# 4. VARIATIONAL QUANTUM CIRCUIT (VQC) COMPARISON


print("\n" + "="*60)
print("4. VARIATIONAL QUANTUM CIRCUIT (VQC) IN QISKIT & PENNYLANE")
print("="*60)

vqc_features = np.array([0.2, -0.6, 0.4, 0.9])
vqc_weights = np.array([1.2, -0.5, 0.8, 1.4]) # 1 layer of weights
print(f"Features: {vqc_features}")
print(f"Weights:  {vqc_weights}")

# 4.1 Qiskit VQC
def qiskit_vqc(feats, w):
    n_q = len(feats)
    qc = QuantumCircuit(n_q)
    
    # Encoder
    for i, val in enumerate(feats):
        qc.ry(val, i)
        
    # Ansatz
    for i in range(n_q):
        qc.ry(w[i], i)
    for i in range(n_q - 1):
        qc.cx(i, i+1)
    qc.cx(n_q - 1, 0)
    
    sv = Statevector.from_instruction(qc)
    
    # Compute Pauli-Z expectation values manually from Statevector
    expectations = []
    for qubit in range(n_q):
        exp_val = 0.0
        for idx in range(2**n_q):
            prob = np.abs(sv.data[idx])**2
            bit = (idx >> qubit) & 1
            exp_val += prob * (1.0 if bit == 0 else -1.0)
        expectations.append(exp_val)
        
    return qc, expectations

qc_vqc, qiskit_expvals = qiskit_vqc(vqc_features, vqc_weights)
print("\n[Qiskit VQC] Pauli-Z Expectation Values:")
print(" ", np.round(qiskit_expvals, 6))
print("[Qiskit VQC] Circuit Layout:")
print(qc_vqc.draw(output='text'))

# 4.2 PennyLane VQC
dev_vqc = qml.device("default.qubit", wires=len(vqc_features))
@qml.qnode(dev_vqc)
def pennylane_vqc(feats, w):
    # Encoder
    qml.AngleEmbedding(feats, wires=range(len(feats)), rotation='Y')
    
    # Ansatz
    for i in range(len(feats)):
        qml.RY(w[i], wires=i)
    for i in range(len(feats) - 1):
        qml.CNOT(wires=[i, i+1])
    qml.CNOT(wires=[len(feats)-1, 0])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(len(feats))]

pennylane_expvals = pennylane_vqc(vqc_features, vqc_weights)
print("\n[PennyLane VQC] Pauli-Z Expectation Values:")
print(" ", np.round(np.array(pennylane_expvals), 6))
print("[PennyLane VQC] Circuit Layout:")
print(qml.draw(pennylane_vqc)(vqc_features, vqc_weights))

# Assert that both expectation values are identical
np.testing.assert_allclose(qiskit_expvals, pennylane_expvals, atol=1e-5)
print("\nVerification Succeeded: Qiskit and PennyLane VQC expectation values match exactly!")
