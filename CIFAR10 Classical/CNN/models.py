# models.py

import torch
import torch.nn as nn
from config import MODEL_CONFIG, IMG_HEIGHT, IMG_WIDTH

class SimpleCNN(nn.Module):
    """
    A VGG-style Convolutional Neural Network (CNN) for CIFAR-10 classification,
    including Batch Normalization for regularization and stability.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # --- 1. Get configurations ---
        cfg = MODEL_CONFIG
        input_channels = cfg['input_channels']
        output_size = cfg['output_size']
        fc_hidden_size = cfg['fc_hidden_size']
        activation_fn = cfg['activation_fn']
        dropout_rate = cfg['dropout_rate']
        
        current_channels = input_channels
        
        # --- 2. Convolutional Layers (Dynamic VGG-style Blocks) ---
        self.conv_sequence = nn.Sequential()
        
        current_size = IMG_HEIGHT # Start at 32
        
        for i, layer_cfg in enumerate(cfg['conv_layers']):
            out_channels = layer_cfg['out_channels']
            kernel_size = layer_cfg['kernel_size']
            padding = layer_cfg['padding']
            pool_after = layer_cfg.get('pool_after', False) 
        
            self.conv_sequence.add_module(
                f'conv_{i}', 
                nn.Conv2d(current_channels, out_channels, kernel_size, padding=padding)
            )
            self.conv_sequence.add_module(f'bn_{i}', nn.BatchNorm2d(out_channels))
            self.conv_sequence.add_module(f'relu_{i}', activation_fn())
            if pool_after:
                self.conv_sequence.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=2, stride=2))
                current_size //= 2 
                
            current_channels = out_channels
            
        # --- 3. Calculate input size for the Fully Connected (FC) layers ---
        final_h = current_size
        final_w = current_size
        
        self.fc_input_size = final_h * final_w * current_channels 
        
        # --- 4. Fully Connected (FC) Layers ---
        self.fc_sequence = nn.Sequential(
            # ADJUSTMENT: Add BatchNorm1d before the first linear layer (after flattening)
            nn.BatchNorm1d(self.fc_input_size), 
            nn.Linear(self.fc_input_size, fc_hidden_size), activation_fn(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_hidden_size, output_size) 
        )

    def forward(self, x):
        """
        Forward pass of the CNN.
        """
        # 1. Convolutional Sequence
        x = self.conv_sequence(x)
        
        # 2. Flatten the feature maps 
        x = x.view(-1, self.fc_input_size)
        
        # 3. Fully Connected Sequence (includes BatchNorm1d)
        x = self.fc_sequence(x)
        
        return x