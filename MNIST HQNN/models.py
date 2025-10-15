# File: models.py

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from config import MODEL_CONFIG, TRAINING_CONFIG

# ====================================================================
# --- 1. Quantum Circuit Definition ---
# ====================================================================

N_QUBITS = MODEL_CONFIG['n_qubits']
N_LAYERS_ANSATZ = MODEL_CONFIG['n_layers_ansatz']

# Use PennyLane's numpy for operations compatible with auto-differentiation in the QNode
# Use PennyLane's high-performance simulator
dev = qml.device("lightning.qubit", wires=N_QUBITS) 

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """The quantum circuit that will be embedded in the PyTorch model."""
    # Encode classical inputs into quantum states
    for i in range(N_QUBITS):
        qml.RY(pnp.pi * inputs[i], wires=i)

    # Apply the parametrized quantum circuit (ansatz)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))

    # Measure the expectation value of the PauliZ operator for each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# ====================================================================
# --- 2. PyTorch Quantum Layer Definition ---
# ====================================================================
class QuantumLayer(nn.Module):
    """A PyTorch layer that encapsulates the quantum circuit."""
    def __init__(self):
        super(QuantumLayer, self).__init__()
        self.qml_circuit = quantum_circuit
        # Define the quantum weights as a trainable PyTorch parameter
        self.weights = nn.Parameter(torch.rand(N_LAYERS_ANSATZ, N_QUBITS, 3, requires_grad=True))

    def forward(self, x):
        """Forward pass of the quantum layer."""
        # x shape: (batch_size, n_qubits)
        batch_out = []
        # Process each item in the batch separately
        for i in range(x.shape[0]):
            q_out_list = self.qml_circuit(x[i], self.weights)
            # The output of qml_circuit is a list of tensors, stack them
            q_out_tensor = torch.stack(q_out_list)
            batch_out.append(q_out_tensor)
        
        # Stack the results for the entire batch
        return torch.stack(batch_out).float()

# ====================================================================
# --- 3. Hybrid Quantum-Classical Model Definition ---
# ====================================================================
class HybridQNN(nn.Module):
    """The complete Hybrid Quantum Neural Network model."""
    def __init__(self):
        super(HybridQNN, self).__init__()
        cfg_train = TRAINING_CONFIG
        cfg_model = MODEL_CONFIG
        
        self.img_size = cfg_train['img_size']
        n_qubits = cfg_model['n_qubits']
        fc_hidden_size = cfg_model['fc_hidden_size']
        num_classes = cfg_model['output_size']

        # Classical pre-processing layers
        self.classical_pre = nn.Sequential(
            nn.Linear(self.img_size * self.img_size, fc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_hidden_size, n_qubits) # Output matches the number of qubits
        )

        # The quantum layer
        self.quantum_layer = QuantumLayer()

        # Classical post-processing layer for classification
        self.classical_post = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        """Forward pass of the full hybrid model."""
        # 1. Flatten the input image
        x = x.view(x.size(0), -1)
        # 2. Pass through the classical pre-processing layers
        x = self.classical_pre(x)
        # 3. Pass through the quantum layer
        x = self.quantum_layer(x)
        # 4. Pass through the final classical layer for classification
        x = self.classical_post(x)
        return x