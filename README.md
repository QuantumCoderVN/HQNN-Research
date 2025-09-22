Hybrid Quantum Neural Networks (HQNN) – Research Project
 
This repository contains experiments on Hybrid Quantum Neural Networks (HQNNs) applied to MNIST and CIFAR-10 datasets.
The project aims to evaluate the effectiveness of hybrid models (classical + quantum layers) compared to purely classical baselines (MLP, CNN).

📌 Objectives

- Implement and evaluate classical baselines:
- MLP on MNIST
- CNN on CIFAR-10
- Design and test Hybrid Quantum Neural Networks (HQNNs):
- HQNN Model 1: Simple VQC after flatten
- HQNN Model 2: Deeper VQC / alternative encoding
- HQNN Model 3: CNN feature extractor + VQC classifier
- Explore Hybrid Quantum Convolutional Neural Network (HQCNN) on CIFAR-10.
- Compare results, analyze strengths/limitations, and discuss the potential of quantum machine learning.

⚙️ Installation

1. Clone the repository
git clone https://github.com/QuantumCoderVN/HQNN-Research.git
cd HQNN-Research

2. Setup Python virtual environment
(Windows)
py -m venv venv
venv\Scripts\activate

(Linux/Mac)
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

🚀 How to Run

Run Classical Baseline (MNIST – MLP)
python src/main.py --dataset mnist --model mlp

Run Classical Baseline (CIFAR-10 – CNN)
python src/main.py --dataset cifar10 --model cnn

Run Hybrid QNN (MNIST – HQNN Model 1)
python src/main.py --dataset mnist --model hqnn1

Run Hybrid QNN (CIFAR-10 – HQNN Model 3)
python src/main.py --dataset cifar10 --model hqnn3

All results (logs, plots, checkpoints) will be saved automatically in:
results/<DATASET>/<MODEL>/<RUN_ID>/


📂 Repository Structure

HQNN-Research/
├── src/
│ ├── config.py → Global configs & hyperparameters
│ ├── data_loader.py → Dataset loaders (MNIST, CIFAR-10)
│ ├── models.py → Classical & hybrid models
│ ├── utils.py → Logging, plotting, evaluation
│ └── main.py → Training script
├── results/ → Outputs (auto-generated per run)
├── requirements.txt → Dependencies
├── .gitignore → Ignore datasets, models, caches
└── README.md → Project documentation
 -----------------------------------------------------------
📊 Example Results
-----------------------------------------------------------
 | Model            | Dataset | Accuracy | Notes                          |
 |------------------|---------|----------|--------------------------------|
 | MLP              | MNIST   | ~97%     | Flattened input, simple arch. |
 | CNN              | MNIST   | ~99%     | Learns spatial features        |
 | CNN              | CIFAR10 | ~75-80%  | Baseline for complex dataset   |
 | HQNN Model 1     | MNIST   | ~85%     | Small VQC, limited encoding    |
 | HQNN Model 3     | MNIST   | ~97%     | CNN features + VQC classifier  |
 | HQNN Model 3     | CIFAR10 | ~60%     | Struggles due to encoding + barren plateau |
 -----------------------------------------------------------
📖 References

- Schuld, M., Killoran, N. (2019). Quantum Machine Learning in Feature Hilbert Spaces.
- Mitarai, K., et al. (2018). Quantum Circuit Learning.
- Farhi, E., Neven, H. (2018). Classification with Quantum Neural Networks on Near Term Processors.

-----------------------------------------------------------
✍️ Author

- Research project by [DINH NHU DUC]
- Contact: [dinhnhuduc28092004@gmail.com]
