# File: data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import TRAINING_CONFIG, DATA_DIR

def get_mnist_data_loaders():
    """
    Downloads and preprocesses the MNIST dataset, then creates DataLoader objects.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Extract configurations
    BATCH_SIZE = TRAINING_CONFIG['batch_size']
    N_TRAIN_SAMPLES = TRAINING_CONFIG['n_train_samples']
    N_TEST_SAMPLES = TRAINING_CONFIG['n_test_samples']
    IMG_SIZE = TRAINING_CONFIG['img_size']

    # --- 1. Define Preprocessing Transformations ---
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
    ])

    # --- 2. Load Datasets ---
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    # --- 3. Limit number of samples for faster training (for demonstration) ---
    train_dataset.data = train_dataset.data[:N_TRAIN_SAMPLES]
    train_dataset.targets = train_dataset.targets[:N_TRAIN_SAMPLES]
    test_dataset.data = test_dataset.data[:N_TEST_SAMPLES]
    test_dataset.targets = test_dataset.targets[:N_TEST_SAMPLES]

    # --- 4. Create DataLoaders ---
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 5. Logging and Verification ---
    print("\n--- Data Loading Details ---")
    print(f"Dataset: MNIST")
    print(f"Image dimensions: {IMG_SIZE}x{IMG_SIZE}x1")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    return train_loader, test_loader