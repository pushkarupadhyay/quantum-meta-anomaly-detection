from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import numpy as np
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC

class ClassicalBaselines:
    @staticmethod
    def isolation_forest(X_train, X_test):
        model = IsolationForest(contamination='auto', random_state=42)
        model.fit(X_train)
        # Decision function: lower values are more anomalous
        # We want scores where higher is more anomalous
        scores = -model.decision_function(X_test)
        return scores

    @staticmethod
    def one_class_svm(X_train, X_test):
        # Train on normal data only
        model = OneClassSVM(gamma='auto')
        model.fit(X_train)
        scores = -model.decision_function(X_test)
        return scores

class QuantumKernelBaseline:
    def __init__(self, n_qubits=8):
        self.n_qubits = n_qubits
        self.feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
        self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)

    def train_and_score(self, X_train, X_test):
        # For simplicity and performance, we'll use a subset of training data if it's too large
        if len(X_train) > 100:
            indices = np.random.choice(len(X_train), 100, replace=False)
            X_train_sub = X_train[indices]
        else:
            X_train_sub = X_train
            
        # One-class SVM with precomputed kernel
        # FidelityQuantumKernel.evaluate(x, y)
        K_train = self.kernel.evaluate(X_train_sub)
        model = OneClassSVM(kernel='precomputed')
        model.fit(K_train)
        
        K_test = self.kernel.evaluate(X_test, X_train_sub)
        scores = -model.decision_function(K_test)
        return scores
