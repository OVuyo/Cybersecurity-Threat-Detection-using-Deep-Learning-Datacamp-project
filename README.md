# Cybersecurity Threat Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A deep learning-based system for detecting cybersecurity threats in network traffic using PyTorch. This project implements a neural network model trained on the BETH dataset to classify network events as benign or malicious.

## 🎯 Project Overview

Cyber threats pose significant risks to organizations worldwide, taking forms such as malware, phishing, and denial-of-service (DOS) attacks. This project develops a deep learning model capable of analyzing network logs and identifying patterns indicative of malicious activity.

### Key Features

- Binary classification of network events (benign vs. malicious)
- Deep neural network implementation using PyTorch
- Trained on the BETH dataset (simulated real-world network logs)
- Achieves **94.6% accuracy** on test data
- Scalable data preprocessing pipeline

## 📊 Dataset

The project uses the **BETH dataset**, which simulates real-world network logs with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| `processId` | Unique identifier for the process generating the event | int64 |
| `threadId` | ID for the thread spawning the log | int64 |
| `parentProcessId` | Label for the parent process | int64 |
| `userId` | ID of user spawning the log | int64 |
| `mountNamespace` | Mounting restrictions for the process | int64 |
| `argsNum` | Number of arguments passed to the event | int64 |
| `returnValue` | Value returned from the event log | int64 |
| `sus_label` | Binary label (1 = suspicious, 0 = benign) | int64 |

## 🏗️ Model Architecture

The project implements a feedforward neural network with the following architecture:

```
Input Layer (7 features)
    ↓
Fully Connected Layer (64 neurons) + ReLU
    ↓
Fully Connected Layer (2 neurons)
    ↓
Softmax (Binary Classification)
```

### Model Specifications

- **Input Size**: 7 features
- **Hidden Layer**: 64 neurons with ReLU activation
- **Output Layer**: 2 classes (benign/malicious)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (learning rate = 0.001)
- **Batch Size**: 32
- **Epochs**: 10

## 📈 Performance

| Dataset | Accuracy |
|---------|----------|
| Training | 99.97% |
| Validation | 100.00% |
| **Test** | **94.60%** |

The model demonstrates strong generalization with high test accuracy, successfully identifying malicious network events.

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cybersecurity-threat-detection.git
cd cybersecurity-threat-detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Training the Model

```python
import pandas as pd
from model import CyberSecurityNN
from utils import prepare_data, train_model

# Load preprocessed data
train_df = pd.read_csv('data/labelled_train.csv')
test_df = pd.read_csv('data/labelled_test.csv')
val_df = pd.read_csv('data/labelled_validation.csv')

# Prepare data
X_train, y_train = prepare_data(train_df)
X_test, y_test = prepare_data(test_df)
X_val, y_val = prepare_data(val_df)

# Train model
model = train_model(X_train, y_train, X_val, y_val)
```

### Making Predictions

```python
import torch

# Load trained model
model = torch.load('models/cybersecurity_model.pth')
model.eval()

# Make predictions on new data
with torch.no_grad():
    predictions = model(X_new)
    _, predicted_labels = torch.max(predictions, 1)
```

## 📁 Project Structure

```
cybersecurity-threat-detection/
│
├── data/                          # Dataset files (not included in repo)
│   ├── labelled_train.csv
│   ├── labelled_test.csv
│   └── labelled_validation.csv
│
├── notebooks/                     # Jupyter notebooks
│   └── notebook.ipynb            # Main project notebook
│
├── models/                        # Saved model checkpoints
│   └── .gitkeep
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── model.py                  # Neural network architecture
│   ├── train.py                  # Training pipeline
│   └── utils.py                  # Utility functions
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 🔧 Technologies Used

- **PyTorch**: Deep learning framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Data preprocessing (StandardScaler)
- **TorchMetrics**: Model evaluation metrics
- **NumPy**: Numerical computing

## 📝 Methodology

1. **Data Preprocessing**
   - Separation of features and labels
   - Feature scaling using StandardScaler
   - Conversion to PyTorch tensors

2. **Model Development**
   - Feedforward neural network architecture
   - Binary classification with CrossEntropy loss
   - Adam optimizer with learning rate 0.001

3. **Training**
   - Batch training with DataLoader
   - Validation after each epoch
   - 10 epochs with batch size of 32

4. **Evaluation**
   - Performance measured on train, validation, and test sets
   - Accuracy metric using TorchMetrics

## 🔮 Future Improvements

- [ ] Implement more complex architectures (LSTM, CNN)
- [ ] Add feature importance analysis
- [ ] Develop real-time threat detection pipeline
- [ ] Implement model interpretability (SHAP, LIME)
- [ ] Add confusion matrix and ROC curve visualizations
- [ ] Experiment with different optimization techniques
- [ ] Deploy model as REST API service

## 👤 Author

**Ovuyo Mphile**

- University of the Witwatersrand (BSc Economic Science)
- Dual Major: Economics & Computational & Applied Mathematics
- Digital Marketing @ Tax-X Advisory
- Specializations: Quantitative Finance, Algorithmic Trading, Machine Learning

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DataCamp** for the project framework and dataset
- **BETH Dataset** creators for providing realistic cybersecurity simulation data
- PyTorch community for excellent documentation and resources

## 📧 Contact

For questions, suggestions, or collaboration opportunities, feel free to reach out!

---

⭐ If you find this project useful, please consider giving it a star!
