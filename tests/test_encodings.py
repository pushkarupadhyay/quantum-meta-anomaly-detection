import unittest
import numpy as np
import pennylane as qml
from qiskit.quantum_info import Statevector
from quantum_effects_and_encodings import (
    qiskit_interference,
    pennylane_interference_circuit,
    qiskit_basis_encode,
    pennylane_basis_encode,
    qiskit_angle_encode,
    pennylane_angle_encode,
    qiskit_amplitude_encode,
    pennylane_amplitude_encode,
    qiskit_vqc,
    pennylane_vqc
)

class TestQuantumEncodings(unittest.TestCase):
    def test_interference_match(self):
        phases = [0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi]
        for phi in phases:
            prob_q = qiskit_interference(phi)[0]
            prob_pl = pennylane_interference_circuit(phi)[0]
            self.assertAlmostEqual(prob_q, float(prob_pl), places=5)

    def test_basis_encoding(self):
        binary_arr = np.array([1, 0, 1, 0])
        # Qiskit is little-endian: binary_arr -> index 5 (0101)
        _, sv_q = qiskit_basis_encode(binary_arr)
        self.assertEqual(np.argmax(np.abs(sv_q)**2), 5)
        
        # PennyLane is big-endian: binary_arr -> index 10 (1010)
        sv_pl = pennylane_basis_encode(binary_arr)
        self.assertEqual(np.argmax(np.abs(sv_pl)**2), 10)

    def test_amplitude_encoding_match(self):
        features = np.array([0.5, -1.2, 0.8, 1.5])
        norm_features = features / np.linalg.norm(features)
        
        _, sv_q = qiskit_amplitude_encode(norm_features)
        sv_pl = pennylane_amplitude_encode(features)
        
        np.testing.assert_allclose(sv_q, sv_pl, atol=1e-6)

    def test_vqc_expectation_match(self):
        features = np.array([0.2, -0.6, 0.4, 0.9])
        weights = np.array([1.2, -0.5, 0.8, 1.4])
        
        _, exp_q = qiskit_vqc(features, weights)
        exp_pl = pennylane_vqc(features, weights)
        
        np.testing.assert_allclose(exp_q, exp_pl, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
