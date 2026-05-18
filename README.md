# CortiX: Neuro-Inspired Adaptive Firewall & Intrusion Detection System
---

**CortiX** is a brain-inspired, low-latency, adaptive firewall prototype. It leverages unsupervised Spiking Neural Networks (SNN) utilizing Hebbian plasticity and Spike-Timing Dependent Plasticity (STDP) alongside a parallel LSTM-CNN classifier to detect and isolate zero-day attacks and insider threat signatures in real-time.

---

## 🧠 System Architecture

```
[Network Interface / Packet Capture]
          ↓
[Module 1: Packet Preprocessor & Spike Encoder]
          ↓
[Module 2: Deep Hebbian SNN Anomaly Engine]    ←→  [Module 6: Attacker Attribution Engine]
          ↓
[Module 3: LSTM-CNN Supervised Classifier]
          ↓
[Module 4: Deep RL Containment Agent]
          ↓
[Module 5: Ransomware Honeypot Trap]
          ↓
[Module 7: React Dashboard + FastAPI Backend + Admin Alerting]
```

---

## 🚀 Key Features

1. **Unsupervised SNN Core (Module 2)**: Online continuous adaptation without labeled training data via Hebbian/STDP weight updating.
2. **Tabular Deep Calibration (Module 3)**: Hybrid LSTM-CNN temporal self-attention classifier trained on CICIDS2017 flow sequences.
3. **Optimized Reinforcement Learning Firewall (Module 4)**: Double-dueling DQN containment controller selecting active rate-limits, VLAN quarantines, or ACL blocks.
4. **Proactive File Watcher Trap (Module 5)**: Realtime bulk filesystem activity watchdog mounting isolated Docker honeypots upon suspicious ransomware behavior.
5. **Admin OSINT attribution & Alerting (Module 6)**: Concurrent async geolocation, abuse lookup, reverse DNS, and SMTP notification dispatch.
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
cd cortix
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
