# models.py
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
# Import both configs
from config import MODEL_CONFIG, Q_CONFIG 

# --- 1. Quantum Circuit Definition ---
N_QUBITS = Q_CONFIG['n_qubits'] # Will be 2
N_LAYERS = Q_CONFIG['n_layers_ansatz']

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    The quantum circuit. 
    It receives the 2 features from Model A.
    """
    # Encoding: Map the 2 inputs to 2 qubits
    for i in range(N_QUBITS):
        qml.RY(pnp.pi * inputs[i], wires=i)

    # Processing (Ansatz): Apply learnable layers
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))

    # Measurement: Return 2 expectation values (our 2 logits)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# --- 2. PyTorch Quantum Layer Wrapper ---
class QuantumLayer(nn.Module):
    """Wraps the QNode as a PyTorch layer."""
    def __init__(self):
        super(QuantumLayer, self).__init__()
        # Shape is (layers, n_qubits, 3 params per gate)
        weights_shape = (N_LAYERS, N_QUBITS, 3)
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

# --- 3. Classical MLP (Model A) ---
# This will automatically read MODEL_CONFIG and build:
# Linear(2, 8) -> ReLU -> Linear(8, 4) -> ReLU -> Linear(4, 2)
class MLP(nn.Module):
    """Flexible MLP model. Reads architecture from MODEL_CONFIG."""
    def __init__(self):
        super(MLP, self).__init__()
        
        # 1. Get configurations
        input_size = MODEL_CONFIG['input_size']
        output_size = MODEL_CONFIG['output_size']
        hidden_layers = MODEL_CONFIG['hidden_layers']
        activation_fn = MODEL_CONFIG['activation_fn']
        
        # 2. Build the layers
        layers = []
        current_input_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(activation_fn())
            current_input_size = hidden_size
            
        # Add the final output layer
        layers.append(nn.Linear(current_input_size, output_size))
        
        # Combine all parts
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor (Batch_Size, 2).
        """
        # NO FLATTENING NEEDED
        return self.network(x)

# --- 4. The Full Hybrid Model ---
class HybridModel(nn.Module):
    """Combines Model A (classical) and the QLayer."""
    def __init__(self):
        super(HybridModel, self).__init__()
        # This is Model A, which is now the pre-processor
        self.classical_mlp = MLP() 
        # This is the QLayer, which is now the classifier
        self.quantum_layer = QuantumLayer()

    def forward(self, x):
        """
        Architecture: Input -> Model A -> QLayer -> Output
        """
        # x shape: (Batch, 2)
        # Pass through classical pre-processor (Model A)
        x = self.classical_mlp(x)
        # x shape is (Batch, 2) (new features)
        
        # Pass through quantum classifier
        x = self.quantum_layer(x)
        # x shape is (Batch, 2) (final logits)
        return x