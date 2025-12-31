# Federated Learning Simulation Platform

A **research-grade simulation platform** for evaluating decentralized machine learning workflows using Federated Averaging (FedAvg).

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)

---

## 🎯 Project Overview

This platform enables **reproducible, academically defensible experiments** comparing centralized and federated learning approaches.

**Key Features:**
- ✅ Privacy-preserving ML experimentation
- ✅ Federated Averaging (FedAvg) implementation
- ✅ Deterministic and reproducible execution
- ✅ Comprehensive performance metrics and visualization
- ✅ Non-IID client data partitioning
- ✅ Research-ready analysis tools

**Use Cases:**
- Benchmarking federated vs centralized learning
- Evaluating convergence behavior under data heterogeneity
- Analyzing privacy-performance trade-offs
- Academic research and publication

---

## 📊 Experiment Design

### Centralized Baseline
- Traditional supervised learning on complete dataset
- Upper-bound performance reference
- Full data access and batch training

### Federated Learning
- Simulates multiple independent clients with private data
- FedAvg aggregation algorithm
- Configurable client participation
- Round-based distributed training

### Comparison Metrics
- Test set accuracy and loss
- Convergence speed and stability
- Communication efficiency
- Generalization performance

---

## 🏗️ Architecture

```
Federate/
├── config.py              # Global experiment configuration
├── dataset.py             # Synthetic dataset generation & partitioning
├── model.py               # Shared neural network architecture
├── centralized.py         # Centralized training pipeline
├── federated.py           # Federated learning pipeline (TFF)
├── visualization.py       # Results visualization and comparison
├── experiment.ipynb       # Main experiment notebook
├── requirements.txt       # Python dependencies
├── experiments/
│   ├── logs/             # Training logs
│   ├── results/          # Experiment results (JSON)
│   └── figures/          # Generated visualizations
└── README.md
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Centralized configuration management, reproducibility settings |
| `dataset.py` | Synthetic data generation, non-IID client partitioning |
| `model.py` | Neural network architecture (shared by both paradigms) |
| `centralized.py` | Baseline centralized training and evaluation |
| `federated.py` | FedAvg simulation with TensorFlow Federated |
| `visualization.py` | Performance comparison charts and metrics |
| `experiment.ipynb` | Complete experiment workflow and analysis |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/MercuryConnor/Federated-Learning-Simulation-Platform.git
cd Federated-Learning-Simulation-Platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments

#### Option A: Jupyter Notebook (Recommended)
```bash
jupyter notebook experiment.ipynb
```
Execute all cells to run the complete experiment.

#### Option B: Python Scripts
```bash
# Run centralized baseline
python centralized.py

# Run federated learning
python federated.py

# Generate visualizations
python visualization.py
```

### 3. View Results

Results are saved in:
- `experiments/results/` - JSON files with metrics
- `experiments/figures/` - Visualization PNG files
- `experiments/logs/` - Training logs

---

## ⚙️ Configuration

Modify [config.py](config.py) to adjust experiment parameters:

```python
class ExperimentConfig:
    # Reproducibility
    RANDOM_SEED = 42
    
    # Dataset
    DATASET_SIZE = 10000
    NUM_FEATURES = 20
    NUM_CLASSES = 2
    
    # Federated Learning
    NUM_CLIENTS = 10          # Number of simulated clients
    NUM_ROUNDS = 50           # Federated training rounds
    CLIENT_FRACTION = 0.3     # Client participation rate
    LOCAL_EPOCHS = 5          # Local training epochs
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    CENTRALIZED_EPOCHS = 100
    
    # Model Architecture
    HIDDEN_UNITS = [64, 32]
    DROPOUT_RATE = 0.3
```

**Key Parameters:**

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `NUM_CLIENTS` | Number of federated clients | More clients = more heterogeneity |
| `NUM_ROUNDS` | Federated training rounds | More rounds = better convergence |
| `CLIENT_FRACTION` | Clients selected per round | Higher = faster but more communication |
| `LOCAL_EPOCHS` | Local training iterations | Higher = more local learning |

---

## 📈 Results and Visualization

The platform automatically generates:

### 1. Training Convergence Curves
- Loss progression (centralized vs federated)
- Accuracy progression across epochs/rounds
- Validation/test metrics overlay

### 2. Final Performance Comparison
- Bar charts comparing test accuracy and loss
- Quantified performance gap analysis

### 3. Client Participation Heatmap
- Visual representation of client sampling
- Participation frequency per round

### 4. Performance Summary Table
- Detailed metric comparison
- Statistical analysis of differences

**Example Output:**
```
======================================================================
PERFORMANCE SUMMARY: CENTRALIZED VS FEDERATED
======================================================================

Metric                    Centralized      Federated    Difference
----------------------------------------------------------------------
Test Accuracy                  0.8750         0.8420       +0.0330
Test Loss                      0.3245         0.3892       -0.0647
======================================================================

INTERPRETATION:
  - Accuracy Gap: 3.30% (Centralized better)
  - Loss Gap: 0.0647 (Federated better)
======================================================================
```

---

## 🔬 Research Workflow

### 1. Hypothesis Formation
Define research questions:
- How does data heterogeneity affect convergence?
- What client participation rate optimizes performance?
- What is the privacy-accuracy trade-off?

### 2. Experiment Configuration
Modify [config.py](config.py) with experimental parameters.

### 3. Execution
Run experiments via [experiment.ipynb](experiment.ipynb) or Python scripts.

### 4. Analysis
- Review convergence curves
- Analyze performance gaps
- Interpret client behavior

### 5. Iteration
- Adjust hyperparameters
- Test different scenarios
- Validate reproducibility

---

## 🧪 Extending the Platform

### Add New Aggregation Strategies
Modify [federated.py](federated.py) to implement:
- FedProx (proximal term regularization)
- FedAdam (adaptive optimization)
- Custom aggregation logic

### Implement Differential Privacy
Add noise mechanisms in aggregation:
```python
def private_aggregation(client_weights, epsilon):
    # Add Gaussian noise for differential privacy
    noise = np.random.normal(0, sensitivity/epsilon, weights.shape)
    return np.mean(client_weights, axis=0) + noise
```

### Custom Dataset Strategies
Extend [dataset.py](dataset.py) with:
- Real-world data loaders
- Different non-IID distributions
- Imbalanced client scenarios

### Advanced Metrics
Add to [visualization.py](visualization.py):
- Per-client performance analysis
- Communication cost estimation
- Fairness metrics

---

## 📚 Technical Details

### Dataset Generation
- Synthetic binary classification using `sklearn.make_classification`
- Deterministic generation with fixed seeds
- Non-IID Dirichlet distribution for client partitioning

### Model Architecture
- Feedforward neural network
- Configurable hidden layers with ReLU activation
- Dropout regularization
- Binary cross-entropy loss

### Federated Learning Implementation
- TensorFlow Federated (TFF) framework
- Federated Averaging (McMahan et al., 2017)
- Client sampling per round
- Model weight aggregation

### Reproducibility Guarantees
- Fixed random seeds across all components
- Deterministic TensorFlow operations
- Versioned dependencies
- Consistent data splits

---

## 🎓 Research Context

This platform is designed for:

**Academic Research:**
- ML conference submissions
- Journal publications
- Thesis experiments

**Interview Preparation:**
- Demonstrating federated learning understanding
- Explaining implementation choices
- Discussing research methodology

**Industrial Applications:**
- Prototyping federated systems
- Benchmarking algorithms
- Privacy-preserving ML evaluation

### Key Research Questions

1. **Convergence:** How does federated learning converge compared to centralized training?
2. **Heterogeneity:** What is the impact of non-IID data distribution?
3. **Communication:** How many rounds are needed for comparable performance?
4. **Privacy-Performance Trade-off:** What accuracy do we sacrifice for privacy?
5. **Client Dynamics:** How does client participation affect stability?

---

## 📖 References

- McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Li et al. (2020). "Federated Optimization in Heterogeneous Networks"
- Kairouz et al. (2021). "Advances and Open Problems in Federated Learning"

---

## 🤝 Contributing

This is a research platform. Contributions welcome:
- New aggregation algorithms
- Improved visualization
- Additional metrics
- Documentation improvements

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🔗 Repository

**GitHub:** https://github.com/MercuryConnor/Federated-Learning-Simulation-Platform

---

## 📧 Contact

For questions about implementation or research collaboration, open an issue on GitHub.

---

## 🏆 Citation

If you use this platform in your research, please cite:

```bibtex
@software{federated_learning_simulation_platform,
  author = {Mercury Connor},
  title = {Federated Learning Simulation Platform},
  year = {2025},
  url = {https://github.com/MercuryConnor/Federated-Learning-Simulation-Platform}
}
```

---

**Built for rigorous experimentation. Designed for reproducible research.**
