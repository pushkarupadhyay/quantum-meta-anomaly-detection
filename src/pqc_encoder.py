import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

class PQCEncoder:
    def __init__(self, n_qubits=8, n_layers=2, noise_level=0.0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.noise_level = noise_level
        if noise_level > 0.0:
            self.dev = qml.device("default.mixed", wires=n_qubits)
        else:
            self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize parameters
        self.params = np.random.uniform(low=-np.pi, high=np.pi, size=(n_layers, n_qubits, 2), requires_grad=True)
        
        @qml.qnode(self.dev)
        def circuit(inputs, params):
            # Angle Encoding
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Variational Layers
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[l, i, 0], wires=i)
                    qml.RZ(params[l, i, 1], wires=i)
                    if self.noise_level > 0:
                        qml.DepolarizingChannel(self.noise_level, wires=i)
                
                # Entangling layer (CNOT chain)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
                
                if self.noise_level > 0:
                    for i in range(self.n_qubits):
                        qml.DepolarizingChannel(self.noise_level, wires=i)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.qnode = circuit

    def forward(self, x):
        # x is a single sample or batch?
        # PennyLane qnodes can handle batches with some setups, but here we'll do it sample by sample if needed
        # or use qml.batch_input
        if x.ndim == 1:
            return np.array(self.qnode(x, self.params))
        else:
            return np.array([self.qnode(sample, self.params) for sample in x])

    def get_params(self):
        return self.params

    def set_params(self, new_params):
        self.params = new_params

    def draw_circuit(self, sample_input=None):
        if sample_input is None:
            sample_input = np.zeros(self.n_qubits)
        
        print(qml.draw(self.qnode)(sample_input, self.params))
        
        # Also save to file
        fig, ax = qml.draw_mpl(self.qnode)(sample_input, self.params)
        plt.title(f"PQC Circuit ({self.n_qubits} qubits, {self.n_layers} layers)")
        # plt.show() # Removed to prevent timeout
        fig.savefig("images/pqc_circuit.png")
        plt.close(fig)

if __name__ == "__main__":
    pqc = PQCEncoder(n_qubits=4, n_layers=1)
    x = np.random.uniform(0, 1, 4)
    output = pqc.forward(x)
    print("PQC Output:", output)
    pqc.draw_circuit(x)
