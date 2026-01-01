# Architecture Documentation

## System Design

### Overview
The Federated Learning Simulation Platform is designed as a modular research system for comparing centralized and federated learning paradigms. The architecture prioritizes reproducibility, extensibility, and scientific rigor.

### Design Principles

1. **Separation of Concerns**
   - Each module handles a single responsibility
   - Clear interfaces between components
   - Independent testing and validation

2. **Reproducibility First**
   - Fixed random seeds throughout
   - Deterministic operations
   - Version-locked dependencies

3. **Research-Grade Quality**
   - Comprehensive logging
   - Detailed metrics tracking
   - Publication-ready visualizations

---

## Component Architecture

### 1. Configuration Layer (`config.py`)

**Purpose:** Centralized parameter management and reproducibility control

**Key Features:**
- Global experiment parameters
- Random seed initialization
- Configuration export for provenance

**Design Rationale:**
- Single source of truth for all experiments
- Easy parameter sweeping for research
- Ensures consistency across modules

---

### 2. Data Layer (`dataset.py`)

**Purpose:** Synthetic dataset generation and federated partitioning

**Components:**

#### Dataset Generation
- Uses `sklearn.make_classification` for controlled complexity
- Deterministic generation with configurable properties
- Binary classification for interpretability

#### Client Partitioning
- Non-IID distribution via Dirichlet sampling
- Disjoint client datasets (no data leakage)
- Configurable heterogeneity level

**Design Rationale:**
- Synthetic data avoids privacy concerns
- Controlled heterogeneity simulates real-world scenarios
- Deterministic partitioning ensures reproducibility

**Data Flow:**
```
generate_synthetic_dataset()
    ↓
partition_data_for_clients()
    ↓
[Client 0], [Client 1], ..., [Client N]
```

---

### 3. Model Layer (`model.py`)

**Purpose:** Shared neural network architecture

**Architecture:**
- Input layer (NUM_FEATURES dimensions)
- Hidden layers with ReLU activation
- Dropout for regularization
- Sigmoid output for binary classification

**Design Rationale:**
- Simple architecture for interpretability
- Shared across both paradigms for fair comparison
- TensorFlow Federated compatible

**Model Creation Flow:**
```
create_model()
    ↓
compile_model()
    ↓
Ready for Training
```

---

### 4. Training Layers

#### 4.1 Centralized Training (`centralized.py`)

**Purpose:** Baseline centralized learning implementation

**Training Pipeline:**
1. Load complete dataset
2. Create train/validation/test splits
3. Train with standard supervised learning
4. Evaluate on test set
5. Save results and model

**Key Features:**
- Standard Keras training loop
- Validation monitoring
- Comprehensive metric logging

**Design Rationale:**
- Provides upper-bound performance reference
- Uses identical model architecture as federated
- Serves as gold standard for comparison

---

#### 4.2 Federated Training (`federated.py`)

**Purpose:** Federated learning simulation with FedAvg

**Training Pipeline:**
1. Partition data to clients
2. Initialize global model
3. For each round:
   - Select client subset
   - Clients train locally
   - Aggregate model updates
   - Update global model
4. Evaluate on test set
5. Save results

**Key Components:**

**Client Simulation:**
- Each client has private dataset
- Local training with SGD
- Only model updates shared

**Aggregation:**
- Federated Averaging (FedAvg)
- Weighted by client dataset size
- Server-side optimization

**TensorFlow Federated Integration:**
```python
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=...,
    server_optimizer_fn=...
)
```

**Design Rationale:**
- TFF provides production-grade federated primitives
- Client sampling simulates realistic participation
- Round-based training enables convergence analysis

---

### 5. Visualization Layer (`visualization.py`)

**Purpose:** Results analysis and performance comparison

**Visualization Types:**

1. **Convergence Curves**
   - Training and validation metrics over time
   - Side-by-side centralized vs federated
   - Loss and accuracy progression

2. **Performance Comparison**
   - Bar charts of final test metrics
   - Quantitative gap analysis
   - Statistical significance indicators

3. **Client Participation Heatmap**
   - Visual representation of client sampling
   - Participation frequency analysis
   - Fairness evaluation

4. **Summary Tables**
   - Detailed metric breakdown
   - Performance interpretation
   - Actionable insights

**Design Rationale:**
- Publication-ready quality
- Interpretable for research analysis
- Supports hypothesis validation

---

### 6. Experiment Layer (`experiment.ipynb`)

**Purpose:** End-to-end experiment orchestration

**Notebook Structure:**

1. **Setup:** Configuration loading and environment setup
2. **Data:** Dataset generation and analysis
3. **Model:** Architecture definition and inspection
4. **Centralized:** Baseline training and evaluation
5. **Federated:** Distributed training and evaluation
6. **Comparison:** Results visualization and analysis
7. **Insights:** Research interpretation and next steps

**Design Rationale:**
- Narrative structure for research documentation
- Reproducible experiment execution
- Interactive analysis and exploration

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Configuration Layer                      │
│                        (config.py)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Data Layer                             │
│  generate_synthetic_dataset() → partition_data_for_clients() │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
    ┌───────────────────────┐   ┌──────────────────────┐
    │  Centralized Pipeline │   │  Federated Pipeline  │
    │   (centralized.py)    │   │    (federated.py)    │
    └───────────────────────┘   └──────────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
                ┌──────────────────────────┐
                │   Visualization Layer     │
                │   (visualization.py)      │
                └──────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Results & Plots │
                    └──────────────────┘
```

---

## Federated Learning Algorithm

### Federated Averaging (FedAvg)

**Server Algorithm:**
```
Initialize global model w_0

For round t = 1 to T:
    1. Select subset S_t of clients (fraction = CLIENT_FRACTION)
    2. Send global model w_t to selected clients
    3. For each client k in S_t:
        - Train locally: w_k^{t+1} ← ClientUpdate(k, w_t)
    4. Aggregate: w_{t+1} ← Σ (n_k/n) * w_k^{t+1}
    5. Evaluate w_{t+1} on test set
```

**Client Algorithm:**
```
ClientUpdate(k, w):
    B ← split data into batches of size BATCH_SIZE
    
    For local_epoch = 1 to LOCAL_EPOCHS:
        For batch b in B:
            w ← w - η∇ℓ(w; b)  # SGD update
    
    Return w
```

**Key Properties:**
- Communication efficiency (rounds << epochs)
- Privacy preservation (data stays local)
- Handles non-IID data
- Weighted aggregation by dataset size

---

## Experiment Workflow

### Phase 1: Data Preparation
1. Generate synthetic dataset with fixed seed
2. Analyze class distribution
3. Partition for clients (non-IID)
4. Create centralized splits
5. Validate data integrity

### Phase 2: Centralized Baseline
1. Train on complete dataset
2. Monitor validation performance
3. Evaluate on test set
4. Save model and metrics
5. Record training curves

### Phase 3: Federated Training
1. Initialize global model
2. For each round:
   - Sample clients
   - Distribute global model
   - Collect local updates
   - Aggregate with FedAvg
   - Evaluate global model
3. Save round metrics
4. Record convergence behavior

### Phase 4: Analysis
1. Compare final test performance
2. Analyze convergence speed
3. Visualize training dynamics
4. Generate summary statistics
5. Interpret results

---

## Reproducibility Guarantees

### Deterministic Operations

1. **Random Seeds:**
   - NumPy seed set globally
   - TensorFlow seed set globally
   - Per-operation seeds when needed

2. **Dataset Generation:**
   - Fixed seed for `make_classification`
   - Deterministic client partitioning
   - Reproducible train/test splits

3. **Model Initialization:**
   - Fixed TensorFlow seed
   - Deterministic weight initialization

4. **Training:**
   - Fixed batch shuffling seeds
   - Deterministic gradient computation
   - Reproducible client sampling

### Validation
- Run experiments multiple times
- Verify identical metrics
- Compare saved results

---

## Performance Considerations

### Computational Complexity

**Centralized Training:**
- Time: O(EPOCHS × DATASET_SIZE / BATCH_SIZE)
- Space: O(DATASET_SIZE × NUM_FEATURES)

**Federated Training:**
- Time: O(ROUNDS × CLIENTS × LOCAL_EPOCHS × CLIENT_DATA_SIZE / BATCH_SIZE)
- Space: O(MAX_CLIENT_DATA_SIZE × NUM_FEATURES)

### Optimization Opportunities

1. **Parallel Client Training:**
   - TFF handles client parallelization
   - Can utilize multiple cores

2. **Efficient Aggregation:**
   - Weighted averaging is O(NUM_CLIENTS)
   - Minimal computational overhead

3. **Memory Management:**
   - Clients process data independently
   - No full dataset in memory during federated

---

## Extension Points

### 1. New Aggregation Algorithms

Implement in `federated.py`:
```python
def custom_aggregation(client_weights, client_sizes):
    # Your aggregation logic
    return aggregated_weights
```

### 2. Different Model Architectures

Modify `model.py`:
```python
def create_custom_model():
    # Your model architecture
    return model
```

### 3. Advanced Client Sampling

Extend `federated.py`:
```python
def importance_based_sampling(clients, importance_scores):
    # Your sampling strategy
    return selected_clients
```

### 4. Privacy Mechanisms

Add to `federated.py`:
```python
def add_differential_privacy(updates, epsilon, delta):
    # Add noise for privacy
    return private_updates
```

---

## Testing Strategy

### Unit Tests
- Dataset generation determinism
- Model architecture correctness
- Aggregation algorithm validity

### Integration Tests
- End-to-end centralized pipeline
- End-to-end federated pipeline
- Results comparison

### Reproducibility Tests
- Multiple runs produce identical results
- Saved/loaded models match
- Metric consistency validation

---

## Deployment Considerations

### This is a Simulation Platform

**What it IS:**
- Research experimentation tool
- Algorithm validation platform
- Performance benchmarking system

**What it is NOT:**
- Production federated system
- Real device deployment
- Networked distributed training

### For Production Deployment:
- Add networking layer
- Implement security protocols
- Handle device failures
- Add asynchronous communication
- Implement model compression

---

## Research Methodology

### Experimental Rigor

1. **Hypothesis Formation:**
   - Define clear research questions
   - Specify expected outcomes

2. **Controlled Variables:**
   - Fix all parameters except experimental variable
   - Use same random seeds for comparison

3. **Multiple Runs:**
   - Verify reproducibility
   - Compute confidence intervals

4. **Statistical Analysis:**
   - Compare distributions
   - Test for significance
   - Report effect sizes

### Documentation Requirements

1. **Configuration:**
   - Record all parameters
   - Note any modifications
   - Version control experiments

2. **Results:**
   - Save raw metrics
   - Archive visualizations
   - Document interpretations

3. **Provenance:**
   - Track experiment lineage
   - Record software versions
   - Maintain reproducibility info

---

## Conclusion

This architecture prioritizes:
- **Modularity:** Easy to extend and modify
- **Reproducibility:** Deterministic and versioned
- **Research Quality:** Publication-ready outputs
- **Interpretability:** Clear design and documentation

The platform serves as a foundation for rigorous federated learning research and experimentation.
