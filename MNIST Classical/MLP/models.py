# models.py

import torch
import torch.nn as nn
from config import MODEL_CONFIG

class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) model for MNIST classification.
    
    The architecture is built dynamically based on the MODEL_CONFIG dictionary.
    This makes the model flexible and easy to modify for hyperparameter search.
    """
    def __init__(self):
        super(MLP, self).__init__()
        
        # --- 1. Get configurations ---
        input_size = MODEL_CONFIG['input_size']
        output_size = MODEL_CONFIG['output_size']
        hidden_layers = MODEL_CONFIG['hidden_layers']
        activation_fn = MODEL_CONFIG['activation_fn']
        use_dropout = MODEL_CONFIG['use_dropout']
        dropout_rate = MODEL_CONFIG['dropout_rate']
        
        # --- 2. Build the Sequential Layers ---
        layers = []
        current_input_size = input_size
        
        # Add Hidden Layers dynamically
        for output_size_layer in hidden_layers:
            # 2a. Linear Layer: Input_Size -> Output_Size
            layers.append(nn.Linear(current_input_size, output_size_layer))
            
            # 2b. Activation Function
            layers.append(activation_fn())
            
            # 2c. Dropout Layer (if configured)
            if use_dropout:
                layers.append(nn.Dropout(p=dropout_rate))
                
            # Update input size for the next layer
            current_input_size = output_size_layer
            
        # Add the final Output Layer (no activation or dropout after this for CrossEntropyLoss)
        # It maps from the last hidden layer size to the number of classes.
        layers.append(nn.Linear(current_input_size, output_size))
        
        # Combine all parts into a single sequence
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor (Batch_Size, 1, H, W).
            
        Returns:
            torch.Tensor: Output logits (Batch_Size, Num_Classes).
        """
        # --- 1. Flatten the input image ---
        # Reshapes the input tensor from (B, 1, 28, 28) to (B, 784)
        x = x.view(x.size(0), -1)
        
        # --- 2. Pass through the network ---
        x = self.network(x)
        
        return x

# Note: The original HybridQNN and QuantumLayer definitions are removed
# to focus purely on the classical MLP model for this study.