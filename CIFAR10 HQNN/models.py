# File: models.py

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from config import MODEL_CONFIG, TRAINING_CONFIG # Import both configs

# ====================================================================
# --- 1. Quantum Circuit Definition ---
# ====================================================================
N_QUBITS = MODEL_CONFIG['n_qubits']
N_LAYERS_ANSATZ = MODEL_CONFIG['n_layers_ansatz']

dev = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """The quantum circuit that will be embedded in the PyTorch model."""
    for i in range(N_QUBITS):
        qml.RY(pnp.pi * inputs[i], wires=i)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# ====================================================================
# --- 2. PyTorch Quantum Layer Definition ---
# ====================================================================
class QuantumLayer(nn.Module):
    """A PyTorch layer that encapsulates the quantum circuit."""
    def __init__(self):
        super(QuantumLayer, self).__init__()
        self.qml_circuit = quantum_circuit
        self.weights = nn.Parameter(torch.rand(N_LAYERS_ANSATZ, N_QUBITS, 3, requires_grad=True))

    def forward(self, x):
        """Forward pass of the quantum layer."""
        batch_out = []
        for i in range(x.shape[0]):
            q_out_list = self.qml_circuit(x[i], self.weights)
            q_out_tensor = torch.stack(q_out_list)
            batch_out.append(q_out_tensor)
        return torch.stack(batch_out).float()

# ====================================================================
# --- 3. Hybrid Quantum-Classical Model Definition ---
# ====================================================================
class HybridQNN(nn.Module):
    """The complete Hybrid Quantum Neural Network model."""
    def __init__(self):
        super(HybridQNN, self).__init__()
        # Get all required configs
        cfg_train = TRAINING_CONFIG
        cfg_model = MODEL_CONFIG
        
        self.img_size = cfg_train['img_size']
        self.num_color_channels = cfg_train['num_color_channels']
        n_qubits = cfg_model['n_qubits']
        fc_hidden_size = cfg_model['fc_hidden_size']
        num_classes = cfg_model['output_size']

        # --- KEY CHANGE HERE ---
        # The input features for the first linear layer must account for
        # image size and the number of color channels.
        input_features = self.img_size * self.img_size * self.num_color_channels

        self.classical_pre = nn.Sequential(
            nn.Linear(input_features, fc_hidden_size), # Updated input size
            nn.ReLU(),
            nn.Linear(fc_hidden_size, n_qubits)
        )

        self.quantum_layer = QuantumLayer()
        self.classical_post = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        """Forward pass of the full hybrid model."""
        x = x.view(x.size(0), -1) # Flatten the image
        x = self.classical_pre(x)
        x = self.quantum_layer(x)
        x = self.classical_post(x)
        return x