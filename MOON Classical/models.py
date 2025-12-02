# models.py
import torch
import torch.nn as nn
from config import MODEL_CONFIG

class MLP(nn.Module):
    """
    Flexible MLP model. Reads architecture from MODEL_CONFIG.
    """
    def __init__(self):
        super(MLP, self).__init__()
        
        # 1. Get configurations
        input_size = MODEL_CONFIG['input_size']
        output_size = MODEL_CONFIG['output_size']
        hidden_layers = MODEL_CONFIG['hidden_layers']
        activation_fn = MODEL_CONFIG['activation_fn']
        use_dropout = MODEL_CONFIG.get('use_dropout', False) # Use .get() for safety
        dropout_rate = MODEL_CONFIG.get('dropout_rate', 0.5)
        
        # 2. Build the layers
        layers = []
        current_input_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(activation_fn())
            if use_dropout:
                layers.append(nn.Dropout(p=dropout_rate))
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
        ### CRITICAL CHANGE ###
        # NO FLATTENING NEEDED! Our data is already (Batch_Size, 2)
        # x = x.view(x.size(0), -1)  <--- REMOVE THIS LINE
        
        return self.network(x)