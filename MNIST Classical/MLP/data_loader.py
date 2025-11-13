# data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Import configuration from config.py
from config import TRAINING_CONFIG, DATA_DIR, MODEL_CONFIG

def get_mnist_data_loaders():
    """
    Downloads and preprocesses the MNIST dataset, then creates DataLoader objects.
    
    The function now uses standard MNIST size (28x28) and parameters from config.py.

    Returns:
        tuple: (train_loader, test_loader)
    """
    
    # Extract configurations
    BATCH_SIZE = TRAINING_CONFIG['batch_size']
    IMG_SIZE = TRAINING_CONFIG['img_size']
    N_TRAIN_SAMPLES = TRAINING_CONFIG.get('n_train_samples')
    N_TEST_SAMPLES = TRAINING_CONFIG.get('n_test_samples')

    # --- 1. Define Preprocessing Transformations ---
    transform = transforms.Compose([
        # We ensure the image is resized to the target size (which is 28x28 in our config)
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # Convert the image (PIL Image or numpy array) to a PyTorch Tensor
        transforms.ToTensor(),
        # Normalize the tensor: (x - mean) / std. 
        # These values (0.1307 and 0.3081) are the standard mean and std dev for MNIST
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # --- 2. Load Datasets ---
    # The 'download=True' argument ensures MNIST is downloaded to the DATA_DIR if not present.
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    # --- 3. Optional Sample Limiting (For quick testing/debugging) ---
    if N_TRAIN_SAMPLES is not None and N_TRAIN_SAMPLES < len(train_dataset):
        # Limit the number of training samples for faster demonstration or debugging
        train_dataset.data = train_dataset.data[:N_TRAIN_SAMPLES]
        train_dataset.targets = train_dataset.targets[:N_TRAIN_SAMPLES]

    if N_TEST_SAMPLES is not None and N_TEST_SAMPLES < len(test_dataset):
        # Limit the number of testing samples
        test_dataset.data = test_dataset.data[:N_TEST_SAMPLES]
        test_dataset.targets = test_dataset.targets[:N_TEST_SAMPLES]

    # --- 4. Create DataLoaders ---
    # DataLoader handles batching, shuffling, and multi-process data loading.
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 5. Logging and Verification ---
    print("\n--- Data Loading Details ---")
    print(f"Dataset Path: {DATA_DIR}")
    print(f"Image dimensions: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Batch Size: {BATCH_SIZE}")

    return train_loader, test_loader