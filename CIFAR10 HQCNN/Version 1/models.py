# File: models.py

import torch
import torch.nn as nn
import torch.nn.functional as F # For functional API if needed, e.g. MaxPool
import pennylane as qml
from pennylane import numpy as pnp
from config import MODEL_CONFIG, TRAINING_CONFIG

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
# --- 3. Hybrid Quantum-Classical Model Definition (HQCNN) ---
# ====================================================================
class HybridQNN(nn.Module):
    """The complete Hybrid Quantum-Convolutional Neural Network model."""
    def __init__(self):
        super(HybridQNN, self).__init__()
        
        cfg_train = TRAINING_CONFIG
        cfg_model = MODEL_CONFIG
        
        self.img_size = cfg_train['img_size']
        self.num_color_channels = cfg_train['num_color_channels']
        n_qubits = cfg_model['n_qubits']
        num_classes = cfg_model['output_size']

        cnn_channels = cfg_model['cnn_channels']
        kernel_size = cfg_model['cnn_kernel_size']
        pool_size = cfg_model['cnn_pool_size']
        cnn_linear_features = cfg_model['cnn_linear_features']

        # --- NEW: Classical Pre-processing with CNN Block ---
        self.classical_pre_cnn = nn.Sequential(
            # First Conv Layer
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(pool_size), # Image size: 32 -> 16

            # Second Conv Layer
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(pool_size) # Image size: 16 -> 8
        )

        # --- Dynamically calculate the output features from CNN block ---
        # This helps ensure the linear layer always matches the CNN's output
        with torch.no_grad(): # Don't track gradients for this dummy calculation
            dummy_input = torch.zeros(1, self.num_color_channels, self.img_size, self.img_size)
            dummy_output = self.classical_pre_cnn(dummy_input)
            cnn_output_features = dummy_output.view(1, -1).shape[1]

        # Linear layers after CNN to compress features for the quantum layer
        self.classical_pre_linear = nn.Sequential(
            nn.Linear(cnn_output_features, cnn_linear_features), # Larger hidden layer for rich features
            nn.ReLU(),
            nn.Linear(cnn_linear_features, n_qubits) # Compress to N_QUBITS
        )

        self.quantum_layer = QuantumLayer()
        self.classical_post = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        """Forward pass of the full hybrid model."""
        # Process through CNN block first
        x = self.classical_pre_cnn(x)
        
        # Flatten the output of the CNN block
        x = x.view(x.size(0), -1) 
        
        # Process through linear layers to compress for quantum input
        x = self.classical_pre_linear(x)

        # Quantum layer
        x = self.quantum_layer(x)
        
        # Classical post-processing
        x = self.classical_post(x)
        return x