# models.py
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from config import MODEL_CONFIG, Q_CONFIG

# --- 1. Quantum Circuit Definition ---
N_QUBITS = Q_CONFIG['n_qubits']
N_LAYERS = Q_CONFIG['n_layers_ansatz']

dev = qml.device("default.qubit", wires=N_QUBITS) # Use default simulator

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """The quantum circuit that acts as a feature map."""
    # Encoding: Use inputs to rotate qubits
    # We map 2 inputs to 2 qubits
    for i in range(N_QUBITS):
        qml.RY(pnp.pi * inputs[i], wires=i)

    # Processing (Ansatz): Apply learnable layers
    # Using StronglyEntanglingLayers as an example
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))

    # Measurement: Return expectation value of PauliZ for each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# --- 2. PyTorch Quantum Layer Wrapper ---
class QuantumLayer(nn.Module):
    """Wraps the QNode as a PyTorch layer."""
    def __init__(self):
        super(QuantumLayer, self).__init__()
        
        # Determine shape of the quantum weights
        # (layers, n_qubits, 3 params per gate)
        weights_shape = (N_LAYERS, N_QUBITS, 3)
        
        # Create nn.Parameter for quantum weights
        self.q_weights = nn.Parameter(torch.rand(weights_shape, requires_grad=True))
        self.qml_circuit = quantum_circuit

    def forward(self, x):
        """Forward pass: execute the circuit for each sample in the batch."""
        batch_out = []
        # Loop over batch
        for i in range(x.shape[0]):
            # Run the QNode
            q_out_list = self.qml_circuit(x[i], self.q_weights)
            # Stack results (e.g., [y1, y2])
            q_out_tensor = torch.stack(q_out_list)
            batch_out.append(q_out_tensor)
            
        # Stack all batch results
        return torch.stack(batch_out).float()

# --- 3. Classical MLP (Model A) - NO CHANGES ---
# This class is identical to the one from the previous step.
# It will act as our post-processing network.
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_size = MODEL_CONFIG['input_size']
        output_size = MODEL_CONFIG['output_size']
        hidden_layers = MODEL_CONFIG['hidden_layers']
        activation_fn = MODEL_CONFIG['activation_fn']
        
        layers = []
        current_input_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(activation_fn())
            current_input_size = hidden_size
            
        layers.append(nn.Linear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # NO FLATTENING
        return self.network(x)

# --- 4. The Full Hybrid Model (NEW) ---
class HybridModel(nn.Module):
    """Combines the QuantumLayer and the classical MLP (Model A)."""
    def __init__(self):
        super(HybridModel, self).__init__()
        self.quantum_layer = QuantumLayer()
        self.classical_mlp = MLP() # This is Model A

    def forward(self, x):
        """
        Architecture: Input -> QLayer -> Model A
        """
        # x shape: (Batch, 2)
        x = self.quantum_layer(x)
        # x shape is now (Batch, 2) (new features from QLayer)
        x = self.classical_mlp(x)
        # x shape is now (Batch, 2) (final logits)
        return x