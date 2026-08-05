# CortiX: Neuro-Inspired Adaptive Firewall & Intrusion Detection System
---

**CortiX** is a brain-inspired, low-latency, adaptive firewall prototype. It leverages unsupervised Spiking Neural Networks (SNN) utilizing Hebbian plasticity and Spike-Timing Dependent Plasticity (STDP) alongside a parallel LSTM-CNN classifier to detect and isolate zero-day attacks and insider threat signatures in real-time.

---

## 🧠 System Architecture

```
[Network Interface / Packet Capture]
          ↓
[Module 1: Packet Preprocessor & Spike Encoder]
    16 flow features → Gaussian population coding → 512 binary spikes
          ↓
[Module 2: Deep Hebbian SNN Anomaly Engine]    ←→  [Module 6: Attacker Attribution Engine]
    5 independent Hebbian modules with:                  AbuseIPDB + VirusTotal + Shodan
    • STDP trace-based weight updates                    GeoIP resolution & reverse DNS
    • Oja normalisation                                  SMTP alerting
    • k-Winner-Take-All (10% sparsity)
    • Independent MAD-based z-score scoring
    • Majority voting consensus
    • Metaplasticity (adaptive learning rate)
    • Periodic synaptic consolidation
          ↓
[Module 3: LSTM-CNN Supervised Classifier]
    Temporal sequence model (10 flows) → 9-class output
    Trained on CICIDS2017 (DoS, DDoS, PortScan, BruteForce,
    WebAttack, Infiltration, Botnet, ZeroDay, BENIGN)
          ↓
[Module 4: Deep RL Containment Agent]
    Double-Dueling DQN in Gymnasium environment
    Actions: ALLOW / RATE_LIMIT / TEMP_BLOCK / QUARANTINE / HARD_BLOCK / HONEYPOT_REDIRECT
    Reward shaping: +10 correct block, -20 false positive, +15 honeypot capture
          ↓
[Module 5: Ransomware Honeypot Trap]
    inotify + Docker isolated filesystem watcher
          ↓
[Module 7: React Dashboard + FastAPI Backend + Admin Alerting]
    Live WebSocket event stream with auto-reconnect
    Recharts visualisations + real-time metrics polling
```

---

## 🚀 Key Features

1. **Unsupervised SNN Core (Module 2)**: Online continuous adaptation without labeled training data via Hebbian/STDP weight updating. Each of 5 modules maintains independent baselines — majority voting reduces false positives.
2. **Tabular Deep Calibration (Module 3)**: Hybrid LSTM-CNN temporal self-attention classifier trained on CICIDS2017 flow sequences.
3. **Optimized Reinforcement Learning Firewall (Module 4)**: Double-dueling DQN containment controller selecting active rate-limits, VLAN quarantines, or ACL blocks.
4. **Proactive File Watcher Trap (Module 5)**: Realtime bulk filesystem activity watchdog mounting isolated Docker honeypots upon suspicious ransomware behavior.
5. **Admin OSINT Attribution & Alerting (Module 6)**: Concurrent async geolocation, abuse lookup, reverse DNS, and SMTP notification dispatch.
6. **Premium Realtime Dashboard (Module 7)**: React & FastAPI websocket stream dashboards detailing active threat profiles, brain performance, and manual controls.

---

## 🛠️ Quick Start Setup

### Step 1: System Dependencies
Install core dependencies on a target Linux environment:
```bash
sudo apt update
sudo apt install -y python3.11 python3-pip docker.io docker-compose \
    mininet tcpdump redis-server postgresql libpcap-dev build-essential
```

### Step 2: Install Packages
```bash
cd CortiX
pip install -r requirements.txt
npm install --prefix cortix-dashboard
```

### Step 3: Run the Daemon Pipeline
```bash
# Start Redis and Postgres backing services
docker-compose up -d

# Start the primary CortiX daemon capturing on eth0
python3 -m cortix.main --interface eth0 --mode live
```

### Step 4: Run the Web Dashboard
```bash
# Start FastAPI backend
uvicorn cortix.api.main:app --host 0.0.0.0 --port 8000 --reload &

# Start React frontend app
npm start --prefix cortix-dashboard
```

---

## 📊 Benchmarking

### Run the Full Evaluation Suite
```bash
# Run SNN benchmark with synthetic test data (no dataset download needed)
python -m evaluation.benchmark --output evaluation/results

# Results are saved to:
#   evaluation/results/benchmark_results.json   — Raw metrics (JSON)
#   evaluation/results/snn_confusion_matrix.png — Confusion matrix heatmap
#   evaluation/results/snn_roc_curve.png        — ROC curve with AUC
#   evaluation/results/snn_precision_recall.png — Precision-Recall curve
#   evaluation/results/latency_profile.png      — p50/p99 latency comparison
```

### Train on Real Data (CICIDS2017)
```bash
# 1. Download the CICIDS2017 dataset (~1.3 GB)
bash data/download_cicids2017.sh

# 2. Merge and map labels to CortiX's 9-class taxonomy
python data/merge_cicids2017.py

# 3. Train the LSTM-CNN classifier
python -m cortix.classifier.train --dataset data/cicids2017_merged.csv

# 4. Train the DQN containment agent
python -m cortix.containment.train_agent --timesteps 500000
```

### Acceptance Criteria
| Metric | Target | Description |
|--------|--------|-------------|
| Hot-path p50 | ≤ 9 ms | Median per-event processing latency |
| FPR | ≤ 0.3% | False Positive Rate |
| Detection Rate | ≥ 70% | True Positive Rate (recall) |
| F1 Score | ≥ 0.70 | Harmonic mean of precision and recall |

---

## 🧪 Testing

### Run Unit Tests
```bash
# Run all tests (SNN core, classifier, RL, integration)
python -m pytest tests/ -v

# Run only SNN core tests (no PyTorch required)
python -m pytest tests/test_snn_core.py -v

# Run with coverage
python -m pytest tests/ -v --cov=cortix --cov-report=term-missing
```

### Test Coverage
- **48 unit tests** covering STDP, Oja normalisation, kWTA, HebbianModule, AnomalyScorer, MetaplasticityController, HebbianEnsemble, SpikeEncoder, PrototypeConsolidator
- **30+ RL/classifier tests** covering reward shaping, heuristic fallback, buffer management
- **5 integration tests** covering full pipeline end-to-end, latency validation, consolidation

---

## 🧪 Active Simulation Scenarios

All tests can be safely replayed inside isolated SDN lab network workspaces:
```bash
# Port Scan Simulation
sudo python3 lab/attack_scenarios/port_scan.py --target 10.0.0.2 --rate fast

# SYN Flood DoS Simulation
sudo python3 lab/attack_scenarios/dos_attack.py --target 10.0.0.2 --duration 30

# Safe Ransomware Watcher Simulation
sudo python3 lab/attack_scenarios/ransomware_sim.py --target-dir /lab/decoy_files/
```

---

## 📁 Project Structure

```
CortiX/
├── cortix/                      # Core Python package
│   ├── main.py                  # Pipeline orchestrator
│   ├── config.py                # Centralised configuration
│   ├── database.py              # SQLAlchemy models
│   ├── redis_bus.py             # Redis pub/sub message bus
│   ├── preprocessor/            # Module 1: Feature extraction + spike encoding
│   ├── snn/                     # Module 2: Hebbian SNN ensemble
│   │   ├── hebbian_module.py    #   Single Hebbian module with STDP
│   │   ├── stdp.py              #   STDP learning rules + Oja + kWTA
│   │   ├── ensemble.py          #   M-module ensemble with majority voting
│   │   ├── scorer.py            #   MAD-based z-score anomaly scorer
│   │   ├── metaplasticity.py    #   Adaptive learning rate controller
│   │   └── consolidator.py      #   Periodic synaptic pruning & recovery
│   ├── classifier/              # Module 3: LSTM-CNN classifier
│   ├── containment/             # Module 4: DQN containment agent
│   ├── honeypot/                # Module 5: Ransomware honeypot
│   ├── attribution/             # Module 6: OSINT attacker profiling
│   └── api/                     # Module 7: FastAPI backend + WebSocket
├── cortix-dashboard/            # React frontend (Vite)
├── evaluation/                  # Benchmark framework
├── tests/                       # Pytest test suite
├── data/                        # Dataset download + merge scripts
└── lab/                         # SDN attack simulation scenarios
```

---

## 📜 Neuroscience-to-IDS Mapping

| Brain Mechanism | CortiX Implementation | IDS Benefit |
|----------------|----------------------|-------------|
| Hebbian Learning ("neurons that fire together wire together") | Weight strengthening for co-occurring spike patterns | Learns normal traffic baselines without labels |
| STDP (Spike-Timing Dependent Plasticity) | Causal timing-based weight updates | Captures temporal attack signatures |
| Metaplasticity | Adaptive learning rate based on activation variance | Prevents catastrophic forgetting under distribution shift |
| k-Winner-Take-All | 10% sparsity in hidden layer | Sparse, efficient, distinct feature detectors |
| Oja's Rule | Weight normalisation after STDP | Prevents runaway weight growth |
| Synaptic Consolidation | Periodic pruning + dead neuron recovery | Memory management, prevents resource exhaustion |
| Ensemble Consensus | 5 independent modules with majority voting | Reduces false positives |
| MAD-based Z-scoring | Robust outlier detection per context | Resistant to outlier contamination |
