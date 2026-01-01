# Experiment Results — Federated Learning Simulation Platform

This document summarizes centralized and federated training executed inside the Docker research environment.

---

## 1. Experiment Configuration
- Deterministic synthetic dataset; shared model architecture; identical metrics.
- Federated setup: 10 clients simulated; 50 rounds; client fraction 0.3; 5 local epochs; batch size 32; FedAvg aggregation.
- Environment: Docker (Linux), TensorFlow 2.14.1, TensorFlow Federated 0.73.0.

---

## 2. Centralized Training Results
- Test accuracy: **0.9693** (loss 0.1166).
- Artifacts: model weights and metrics JSON at [experiments/results/centralized_results_20260101_143631.json](experiments/results/centralized_results_20260101_143631.json) and [experiments/results/centralized_model_20260101_143631.keras](experiments/results/centralized_model_20260101_143631.keras).
- Serves as the upper-bound baseline with full data visibility.

---

## 3. Federated Training Results
- Test accuracy: **0.9153** (loss 0.2429).
- Rounds completed: **50/50** with client fraction 0.3.
- Metrics log: [experiments/results/federated_results_20260101_143714.json](experiments/results/federated_results_20260101_143714.json).
- Demonstrates stable convergence under decentralized constraints.

---

## 4. Performance Comparison
Generated from saved logs:
- Accuracy chart: [experiments/figures/federated_vs_centralized_accuracy.png](experiments/figures/federated_vs_centralized_accuracy.png)
- Loss chart: [experiments/figures/federated_vs_centralized_loss.png](experiments/figures/federated_vs_centralized_loss.png)

Interpretation:
- Centralized converges faster and achieves higher final accuracy.
- Federated converges more slowly due to client partitioning and round-based aggregation but attains strong generalization.

---

## 5. Reproducibility Notes
- Fixed random seeds; deterministic data generation and partitions.
- Shared architecture and hyperparameters across pipelines.
- Machine-readable logs stored under [experiments/results](experiments/results).
- Executed inside Docker to avoid dependency drift.

---

## 6. Optional Extensions
- Explore non-IID client splits, varied participation rates, or communication-efficient optimizers.
- Add cross-round analytical plots via `generate_all_visualizations()` if broader comparisons are needed.

---

## 7. Conclusion
The platform correctly simulates federated learning, supports reproducible experiments, and provides interpretable metrics and convergence curves for centralized vs federated benchmarking.
