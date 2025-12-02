# data_loader.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import DATA_CONFIG, TRAINING_CONFIG

def get_moon_data_loaders():
    """
    Generates the Moon dataset and prepares PyTorch DataLoaders.
    """
    # 1. Get parameters from config
    N_SAMPLES = DATA_CONFIG['n_samples']
    NOISE = DATA_CONFIG['noise']
    TEST_SIZE = DATA_CONFIG['test_size']
    RANDOM_STATE = DATA_CONFIG['random_state']
    BATCH_SIZE = TRAINING_CONFIG['batch_size']

    # 2. Generate dataset
    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=RANDOM_STATE)

    # 3. Scale the data (Standard practice)
    # This helps the model train faster and more stably
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 4. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 5. Convert NumPy arrays to PyTorch Tensors
    # Note: Data must be FloatTensor, Labels must be LongTensor for CrossEntropyLoss
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # 6. Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # 7. Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"--- Data Loaded ---")
    print(f"Total samples: {N_SAMPLES}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader, X_test_tensor, y_test_tensor