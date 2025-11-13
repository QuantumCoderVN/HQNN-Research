# models.py

import torch
import torch.nn as nn
from config import MODEL_CONFIG, TRAINING_CONFIG # Import the new CNN config

class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) model for MNIST classification.
    
    The architecture is built dynamically based on the MODEL_CONFIG dictionary.
    """
    def __init__(self):
        super(CNN, self).__init__()
        
        # --- 1. Get configurations ---
        input_channels = MODEL_CONFIG['input_channels']
        output_size = MODEL_CONFIG['output_size']
        conv_layers_config = MODEL_CONFIG['conv_layers']
        pool_kernel_size = MODEL_CONFIG['pool_kernel_size']
        fc_layers_config = MODEL_CONFIG['fc_layers']
        activation_fn = MODEL_CONFIG['activation_fn']
        use_dropout = MODEL_CONFIG['use_dropout']
        dropout_rate = MODEL_CONFIG['dropout_rate']

        # --- 2. Build the Convolutional Layers ---
        conv_layers = []
        current_channels = input_channels
        
        for (out_channels, kernel, stride, padding) in conv_layers_config:
            # Add Convolutional Layer
            conv_layers.append(nn.Conv2d(current_channels, out_channels, 
                                         kernel_size=kernel, stride=stride, padding=padding))
            # Add Activation
            conv_layers.append(activation_fn())
            # Add Pooling
            conv_layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_kernel_size))
            
            current_channels = out_channels # Update channels for next layer
            
        self.conv_part = nn.Sequential(*conv_layers)

        # --- 3. Automatically determine the input size for the FC layers ---
        # This is a robust trick: create a dummy input and pass it through
        # the conv_part to see what the output shape is.
        dummy_input = torch.randn(1, input_channels, 
                                  TRAINING_CONFIG['img_size'], TRAINING_CONFIG['img_size'])
        with torch.no_grad():
            dummy_output = self.conv_part(dummy_input)
            
        # The output size will be (1, num_channels, H, W). We need to flatten this.
        self.fc_input_size = dummy_output.view(1, -1).size(1)
        
        # --- 4. Build the Fully-Connected (Classifier) Layers ---
        fc_layers = []
        current_input_size = self.fc_input_size
        
        for hidden_size in fc_layers_config:
            fc_layers.append(nn.Linear(current_input_size, hidden_size))
            fc_layers.append(activation_fn())
            if use_dropout:
                fc_layers.append(nn.Dropout(p=dropout_rate))
            current_input_size = hidden_size
            
        # Add the final output layer
        fc_layers.append(nn.Linear(current_input_size, output_size))
        
        self.fc_part = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Forward pass of the CNN.
        
        Args:
            x (torch.Tensor): Input tensor (Batch_Size, 1, 28, 28).
            
        Returns:
            torch.Tensor: Output logits (Batch_Size, Num_Classes).
        """
        # --- 1. Pass through convolutional layers ---
        # x starts as (B, 1, 28, 28)
        x = self.conv_part(x)
        # x is now (B, 32, 7, 7) after 2 conv/pool layers
        
        # --- 2. Flatten the output for the FC layers ---
        x = x.view(x.size(0), -1) # Flattens to (B, 32*7*7) = (B, 1568)
        
        # --- 3. Pass through the classifier ---
        x = self.fc_part(x)
        
        return x

# Note: The MLP class has been replaced by the CNN class.
# You can keep the MLP class in a separate file or commented out if you wish.