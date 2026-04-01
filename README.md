# 🧬 ToxiSense: Precision Drug Toxicity Engine
### Powered by Team Code Cuisine 🧑‍🍳🧪

**ToxiSense** is an enterprise-grade research platform for predicting molecular toxicity using a **Stacked Meta-Learner Ensemble.** It transforms raw chemical SMILES into deep-insight toxicological risk assessments across 12 critical biological endpoints.

---

## 🚀 Key Features

### 🧠 High-Altitude Model Stacking
ToxiSense doesn't just average results; it uses an intelligent **Meta-Learner Architecture**:
- **Tier 1 (Base Learners)**: Optimized **XGBoost**, **LightGBM**, and **Random Forest** models, hyperparameter-tuned with **Optuna**.
- **Tier 2 (Meta-Learner)**: A **Logistic Regression** bridge that intelligently weighs the base predictions for maximum accuracy.
- **Benchmarks**: Achieved a final Mean **Stacked AUC of 0.7880** on the Tox21 dataset.

### 🔬 Research-Grade Observatory (UI/UX)
Built for pharmaceutical professionals and hackathon judges:
- **Glassmorphism Design**: A premium, obsidian-dark research interface.
- **Interpretability**: Interactive SHAP feature importance charts and molecular structural insights.
- **3D Visualization**: Real-time 3D rendering of drug compounds.

---

## 📂 Project Structure

```text
├── backend/                # FastAPI High-Concurrency Server
│   ├── api.py              # Main API Logic & Feature Extraction
│   └── explorer.py         # 3D Molecular Analysis Tools
├── frontend/               # React + TailwindCSS v4 Dashboard
├── models/                 # Pre-trained Stacked ML Binaries
├── data/                   # Tox21 Curated Dataset (CSV/GZ)
├── train_fast.py           # Optimized Tuning Script (~30 min)
├── train_upgrade.py        # Deep-Dive Ensemble Training Script (2h+)
├── rescue_training.py      # Meta-Learner Generation & Repair Script
└── requirements.txt        # Production Dependencies
```

---

## 🛠 Installation & Setup

### 1. Requirements
Ensure you have **Python 3.10+** and **Node.js 18+** installed.
```bash
pip install -r requirements.txt
cd frontend && npm install
```

### 2. Launching the Backend
```bash
python3 -m uvicorn backend.api:app --reload --port 8000
```

### 3. Launching the Frontend
```bash
cd frontend
npm run dev
```

---

## 🧪 Training & Upgrading
We’ve included professional scripts to re-train the brain on your own hardware:
-   **`python3 train_fast.py`**: Fast optimization with 12 Optuna trials per task.
-   **`python3 train_upgrade.py`**: Full-scale enterprise training with 50 trials.
-   **`python3 rescue_training.py`**: The "Repair Hero"—used to finish stacking logic without full re-training.

---

## 🧑‍💻 Developed By
**Team Code Cuisine**
- *Architecting the future of computational toxicology.*

---

## ⚖️ License
MIT License - Open for hackathon review and scientific research.
