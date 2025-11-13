# File: models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.num_color_channels = cfg_train['num_color_channels'] # This will be 1
        n_qubits = cfg_model['n_qubits']
        num_classes = cfg_model['output_size']

        cnn_channels = cfg_model['cnn_channels']
        kernel_size = cfg_model['cnn_kernel_size']
        pool_size = cfg_model['cnn_pool_size']
        cnn_linear_features = cfg_model['cnn_linear_features']

        # --- NEW: Classical Pre-processing with CNN Block ---
        self.classical_pre_cnn = nn.Sequential(
            # Block 1
            ### MODIFIED ### (Read in_channels from config, which is 1 now)
            nn.Conv2d(in_channels=self.num_color_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # --- Dynamically calculate the output features from CNN block ---
        # (No change needed here. It will automatically use
        # self.num_color_channels=1 and self.img_size=28)
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.num_color_channels, self.img_size, self.img_size)
            dummy_output = self.classical_pre_cnn(dummy_input)
            cnn_output_features = dummy_output.view(1, -1).shape[1]
            
            # For MNIST 28x28 -> Pool(14x14) -> Pool(7x7)
            # Output size will be 64 * 7 * 7

        # Linear layers after CNN to compress features for the quantum layer
        self.classical_pre_linear = nn.Sequential(
            nn.Linear(cnn_output_features, cnn_linear_features), 
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