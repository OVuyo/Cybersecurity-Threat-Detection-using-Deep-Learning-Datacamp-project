# Detecting Cybersecurity Threats using Deep Learning

A deep learning project for detecting cybersecurity threats in network traffic using PyTorch and neural networks.

> **Note**: This project was completed as part of DataCamp's Premium Projects curriculum.

## 📋 Project Overview

This project implements a binary classification neural network to identify malicious network events from the BETH dataset. The model distinguishes between benign (normal) and suspicious (malicious) network activities by analyzing system log features.

**Final Test Accuracy: 94.60%**

## 📊 Dataset

The BETH dataset contains simulated real-world network logs with the following features:

| Feature | Description |
|---------|-------------|
| `processId` | Unique identifier for the process |
| `threadId` | ID for the thread spawning the log |
| `parentProcessId` | Parent process identifier |
| `userId` | User ID spawning the log |
| `mountNamespace` | Process mounting restrictions |
| `argsNum` | Number of arguments passed |
| `returnValue` | Value returned from event |
| `sus_label` | Target: 0 = Benign, 1 = Suspicious |

## 🏗️ Model Architecture

- **Input Layer**: 7 features
- **Hidden Layer**: 64 neurons with ReLU activation
- **Output Layer**: 2 classes (binary classification)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (learning rate = 0.001)
- **Training**: 10 epochs with batch size of 32

## 📁 Project Structure

```
cybersecurity-threat-detection/
│
├── data/                                    # Dataset files (not included)
│   ├── labelled_train.csv
│   ├── labelled_test.csv
│   └── labelled_validation.csv
│
├── Detecting_Cybersecurity_Threats.ipynb   # Main project notebook
└── README.md                                # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy torch torchmetrics scikit-learn matplotlib seaborn jupyter
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### Running the Notebook

1. **Clone or download this repository**
2. **Add the dataset**: Place your CSV files in the `data/` folder
3. **Open the notebook**: 
   ```bash
   jupyter notebook Detecting_Cybersecurity_Threats.ipynb
   ```
4. **Run all cells** to train and evaluate the model

## 📈 Results

| Dataset | Accuracy |
|---------|----------|
| Training | 99.97% |
| Validation | 100.00% |
| **Test** | **94.60%** |

The model successfully detects cybersecurity threats with high accuracy on unseen test data.

## 🔍 What's in the Notebook

1. **Data Exploration**: Loading and understanding the dataset
2. **Data Preprocessing**: Scaling features and creating PyTorch tensors
3. **Model Building**: Defining a feedforward neural network
4. **Training**: 10 epochs with validation monitoring
5. **Evaluation**: Performance metrics and confusion matrix
6. **Analysis**: Sample predictions and detailed classification report

## 💡 Key Learnings

- Deep learning can effectively identify patterns in network security data
- Proper data preprocessing (scaling) is crucial for neural network performance
- Binary classification with CrossEntropyLoss works well for threat detection
- The model generalizes well from training to test data

## 🔮 Future Improvements

- Experiment with deeper architectures (more hidden layers)
- Try dropout layers to reduce overfitting
- Test different optimizers and learning rates
- Implement early stopping based on validation loss
- Add more advanced evaluation metrics (ROC-AUC, precision-recall curves)

## 🛠️ Technologies

- **Python 3.8+**
- **PyTorch**: Deep learning framework
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Preprocessing and metrics
- **Matplotlib & Seaborn**: Visualization
- **TorchMetrics**: Model evaluation

## 👤 Author

**Ovuyo Mphile**
- University of the Witwatersrand
- BSc Economic Science (Economics & Computational Mathematics)

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- DataCamp for the project framework and dataset
- BETH dataset creators for realistic cybersecurity simulation data
- PyTorch community for excellent documentation

---

⭐ If you find this project helpful, please consider giving it a star!
