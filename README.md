# Federated Learning Simulation Platform

A research-grade federated learning experimentation platform for simulating decentralized learning across multiple client nodes and benchmarking performance against centralized training.

This project is designed for:
- privacy-preserving learning research
- reproducible ML experimentation
- centralized vs federated performance comparison
- experiment logging, evaluation, and visualization

The platform emphasizes:
- modular experiment design
- deterministic execution
- academic-style benchmarking
- explainable and transparent evaluation

---

## Overview
This platform simulates a federated learning environment in which:
- multiple synthetic client datasets are generated
- local models are trained independently on each client
- client updates are aggregated using Federated Averaging (FedAvg)
- a global model is updated round-by-round

A centralized training pipeline using the same dataset and model is included to serve as a performance baseline.

Both pipelines:
- share the same model architecture
- share the same dataset generator
- use structured logging and saved metrics
- produce convergence comparison plots

---

## Architecture
Federated Learning Simulation Platform
```

 dataset.py               synthetic dataset generation
 model.py                 shared model architecture
 centralized.py           centralized training pipeline
 federated.py             federated simulation / FedAvg training
 visualization.py         experiment comparison and charts

 experiments/
    results/             JSON logs and saved models
    figures/             accuracy and loss comparison plots

 Dockerfile
 docker-compose.yml
 RESULTS.md
 README.md
```

The system follows reproducible ML experiment structure: modular, parameterized, and comparable across runs.

---

## Experiment Workflow
### Centralized Training
- Train a model on the full dataset
- Evaluate on test split
- Save metrics and weights
- Log JSON results
- Used as the upper-bound performance reference

### Federated Training Simulation
- Generate deterministic synthetic dataset
- Partition into multiple client datasets
- Run per-client local training
- Aggregate weights using FedAvg
- Update global model each round
- Evaluate federated global model
- Save metrics and logs

This simulates private client data silos, decentralized compute, and global aggregation. No real networking or devices are used—this is a controlled offline simulator.

---

## Data Generation and Client Partitioning
- Synthetic, deterministic, reproducible dataset
- Client data is disjoint, locally isolated, and never shared
- Configurable client count and dataset size for realistic federated heterogeneity

---

## Parameter Configuration
Global experiment parameters are defined in [config.py](config.py), including number of clients, rounds, batch size, local epochs, learning rate, dataset size, and client participation fraction. All experiments use fixed random seeds to ensure reproducibility.

---

## Running the Platform (Docker)
Build and start container:
```
docker compose up -d
```

Run centralized training:
```
docker compose exec federate-tff python -c "from centralized import run_centralized_experiment; run_centralized_experiment()"
```

Run federated simulation:
```
docker compose exec federate-tff python -c "from federated import run_federated_experiment; run_federated_experiment()"
```

Generate comparison charts:
```
docker compose exec federate-tff python -c "from visualization import generate_comparison_plots; generate_comparison_plots()"
```

Artifacts are written to [experiments/results](experiments/results) and [experiments/figures](experiments/figures).

Environment used for validated runs: Docker (Linux), TensorFlow 2.14.1, TensorFlow Federated 0.73.0.

---

## Results and Performance Comparison
Documented in [RESULTS.md](RESULTS.md).

**Centralized Training (Baseline)**
- Test Accuracy: ~0.9693
- Loss: 0.1166
- Artifacts: [experiments/results/centralized_results_20260101_143631.json](experiments/results/centralized_results_20260101_143631.json), [experiments/results/centralized_model_20260101_143631.keras](experiments/results/centralized_model_20260101_143631.keras)

**Federated Learning (50 rounds)**
- Test Accuracy: ~0.9153
- Loss: 0.2429
- Rounds Completed: 50 / 50
- Artifact: [experiments/results/federated_results_20260101_143714.json](experiments/results/federated_results_20260101_143714.json)

---

## Convergence and Performance Charts
Generated from saved logs:
- Accuracy comparison: [experiments/figures/federated_vs_centralized_accuracy.png](experiments/figures/federated_vs_centralized_accuracy.png)
- Loss comparison: [experiments/figures/federated_vs_centralized_loss.png](experiments/figures/federated_vs_centralized_loss.png)

These plots illustrate that centralized training converges faster and higher; federated learning converges gradually via FedAvg while maintaining competitive accuracy—reflecting expected FL trade-offs where centralized maximizes accuracy and federated prioritizes data locality.

---

## Reproducibility
- Deterministic dataset generation and fixed random seeds
- Structured JSON experiment logs under [experiments/results](experiments/results)
- Shared architecture across pipelines
- Docker-validated runtime to avoid dependency drift

---

## Extensibility for Further Research
- Non-IID client partitioning
- Variable participation sampling
- Additional aggregation strategies
- Per-round metric export and analysis
- Federated robustness experiments

These can be added without modifying the core pipeline design.

---

## Why This Matters
This project demonstrates applied federated learning engineering, decentralized learning system behavior, experiment-driven ML evaluation, and research-grade workflow. Suitable for AI research portfolios, ML benchmarking, federated learning concept demonstration, and academic-style study environments.

---

## Conclusion
This platform provides a clean federated training simulator, a centralized baseline reference, structured logs and experiment outputs, convergence visualization, reproducible execution, and an extensible research foundation. It enables meaningful study of privacy-preserving distributed learning, performance vs convergence trade-offs, and federated aggregation behavior.
