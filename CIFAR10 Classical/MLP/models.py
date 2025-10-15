# models.py

import torch
import torch.nn as nn
from config import MODEL_CONFIG

class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) model for CIFAR-10 classification.
    
    The architecture is built dynamically based on the MODEL_CONFIG dictionary.
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
            layers.append(nn.Linear(current_input_size, output_size_layer))
            layers.append(activation_fn())
            
            if use_dropout:
                layers.append(nn.Dropout(p=dropout_rate))
                
            current_input_size = output_size_layer
            
        # Add the final Output Layer
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor (Batch_Size, C, H, W).
            
        Returns:
            torch.Tensor: Output logits (Batch_Size, Num_Classes).
        """
        # --- 1. Flatten the input image (B, 3, 32, 32) -> (B, 3072) ---
        x = x.view(x.size(0), -1)
        
        # --- 2. Pass through the network ---
        x = self.network(x)
        
        return x