---
title: AutoMLOps
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

<h1>🤖 AutoMLOps</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=4F46E5&center=true&vCenter=true&width=700&lines=ML+Experiment+Tracking+%26+Pipeline+Platform;Visual+DAG+Orchestration+with+Apache+Airflow;50%2B+Algorithms+%C2%B7+AutoML+%C2%B7+Model+Registry" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-4f46e5?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Airflow](https://img.shields.io/badge/Apache_Airflow-2.10-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white)](https://airflow.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

**🤖 AutoMLOps** — A full-stack ML experiment tracking and pipeline orchestration platform. Train 50+ algorithms, run visual DAG pipelines powered by Apache Airflow, and manage models — all in one Docker container deployed to HuggingFace Spaces.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Pages](#-pages)
- [Pipelines](#-pipelines)
- [ML Algorithms](#-ml-algorithms)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>🎨 <b>Pipeline Studio</b></td>
    <td>Interactive full-screen DAG canvas with clickable nodes, slide-in config panel, and live execution terminal</td>
  </tr>
  <tr>
    <td>✈️ <b>Real Apache Airflow</b></td>
    <td>Pipelines execute as genuine Airflow DAGs with XCom, TaskInstance tracking, and DagRun polling</td>
  </tr>
  <tr>
    <td>🤖 <b>AutoML Engine</b></td>
    <td>Automated hyperparameter search across all algorithm categories for classification and regression tasks</td>
  </tr>
  <tr>
    <td>📈 <b>MLflow Tracking</b></td>
    <td>Every training run logs parameters, metrics, and model artifacts to a persistent SQLite-backed MLflow store</td>
  </tr>
  <tr>
    <td>📦 <b>Model Registry</b></td>
    <td>Register, version, and transition models through Staging → Production → Archived lifecycle stages</td>
  </tr>
  <tr>
    <td>🌙 <b>Theme Toggle</b></td>
    <td>Dark and light mode with instant CSS variable switching and localStorage persistence</td>
  </tr>
  <tr>
    <td>🐳 <b>Single-Container Deployment</b></td>
    <td>Flask + Airflow Scheduler + SQLite in one Docker image — no external services required</td>
  </tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           AutoMLOps                                 │
│                                                                     │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐  │
│  │  Datasets   │───▶│   MLOps Engine   │───▶│   Flask API       │  │
│  │  (sklearn + │    │  (sklearn /      │    │   Backend         │  │
│  │   custom)   │    │   XGBoost /      │    └────────┬──────────┘  │
│  └─────────────┘    │   LightGBM /     │             │             │
│                     │   MLP)           │    ┌────────▼──────────┐  │
│                     └──────────────────┘    │  Pipeline Studio  │  │
│                                             │  AutoML Page      │  │
│  ┌─────────────┐    ┌──────────────────┐    │  Model Registry   │  │
│  │  MLflow DB  │◀───│ Airflow          │    └───────────────────┘  │
│  │  (SQLite)   │    │ Scheduler        │                           │
│  └─────────────┘    │ (DAGs / XCom /   │                           │
│                     │  TaskInstance)   │                           │
│                     └──────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker (for containerised deployment)
- Git

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/AutoMLOps.git
cd AutoMLOps

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install app dependencies
pip install -r requirements.txt

# 4. Install Apache Airflow with official constraints
AIRFLOW_VERSION=2.10.4
pip install "apache-airflow==${AIRFLOW_VERSION}" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.11.txt"

# 5. Initialise Airflow metadata DB
export AIRFLOW_HOME=$(pwd)/airflow_home
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
airflow db migrate

# 6. Start the Airflow scheduler (background)
airflow scheduler &

# 7. Run the Flask application
python app.py
```

Open your browser at `http://localhost:7860` 🎉

---

## 🐳 Docker Deployment

```bash
# Build and run
docker build -t automlops .
docker run -p 7860:7860 automlops

# Or deploy directly to HuggingFace Spaces
# Push to a Space with sdk: docker in README frontmatter
```

The Dockerfile builds a single image containing:
- Flask application (served via Gunicorn on port 7860)
- Apache Airflow Scheduler (started by `start.sh`)
- All Python dependencies with Airflow constraint-file pinning
- Pre-initialised Airflow SQLite metadata DB (`airflow db migrate`)

---

## 📄 Pages

| Page | Route | Description | Status |
|------|-------|-------------|--------|
| 🎨 Pipeline Studio | `/` | Interactive DAG canvas — click nodes to configure and execute pipelines | ✅ Live |
| 🤖 AutoML | `/automl` | Automated algorithm search across 50+ models for any dataset | ✅ Live |
| 📦 Model Registry | `/models` | Browse registered models, versions, and lifecycle stages | ✅ Live |

---

## 🔗 Pipelines

Three production-quality pipelines are pre-built and immediately executable from the Pipeline Studio:

**Training Pipeline** (`training_pipeline`)
```
Load Data → Validate → Preprocess → Feature Engineering → Train Model → Evaluate → Report → Register → Deploy to Staging
```

**Retraining Pipeline** (`retraining_pipeline`)
```
Drift Detection → Fetch New Data → Merge Datasets → Retrain Champion → A/B Test → Promote to Production
```

**Data Processing Pipeline** (`data_pipeline`)
```
Ingest Raw Data → Clean Data → Encode Features → Scale Features → Save to Feature Store
```

Each pipeline node is clickable in the UI — configurable nodes (dataset picker, algorithm picker) show a purple indicator dot. Execution logs stream live in the built-in terminal panel.

---

## 🧠 ML Algorithms

```python
# AutoMLOps Algorithm Registry — 50+ algorithms across 2 tasks
ALGORITHMS = {
    "classification": {
        "Linear Models":          ["Logistic Regression", "Logistic Regression (L1)", "Ridge Classifier",
                                   "SGD Classifier", "Passive Aggressive", "Linear Discriminant Analysis"],
        "Tree-Based":             ["Decision Tree", "Random Forest", "Extra Trees",
                                   "Quadratic Discriminant Analysis"],
        "Ensemble / Boosting":    ["Gradient Boosting", "AdaBoost", "Bagging Classifier",
                                   "XGBoost", "LightGBM"],
        "Support Vector Machines":["SVC (RBF Kernel)", "SVC (Polynomial)", "SVC (Linear)", "LinearSVC"],
        "Probabilistic":          ["Gaussian Naive Bayes", "Bernoulli Naive Bayes", "Complement Naive Bayes"],
        "Instance-Based (KNN)":   ["KNN (k=3)", "KNN (k=5)", "KNN (k=9)"],
        "Neural Networks":        ["MLP (Small)", "MLP (Medium)", "MLP (Deep)"],
    },
    "regression": {
        "Linear Models":          ["Linear Regression", "Ridge Regression", "Lasso",
                                   "ElasticNet", "Bayesian Ridge", "Huber Regressor"],
        "Tree-Based":             ["Decision Tree Regressor", "Random Forest Regressor",
                                   "Extra Trees Regressor"],
        "Ensemble / Boosting":    ["Gradient Boosting Regressor", "AdaBoost Regressor",
                                   "Bagging Regressor", "XGBoost Regressor", "LightGBM Regressor"],
        "Support Vector Machines":["SVR (RBF)", "SVR (Linear)"],
        "Instance-Based (KNN)":   ["KNN Regressor (k=3)", "KNN Regressor (k=5)"],
        "Neural Networks":        ["MLP Regressor (Small)", "MLP Regressor (Medium)"],
    }
}
```

**Built-in Datasets:**
| Dataset | Task | Samples | Features |
|---------|------|---------|---------|
| Iris Flowers | Classification | 150 | 4 |
| Wine Quality | Classification | 178 | 13 |
| Breast Cancer | Classification | 569 | 30 |
| Diabetes Progression | Regression | 442 | 10 |
| California Housing | Regression | 20,640 | 8 |

---

## 📁 Project Structure

```
AutoMLOps/
│
├── 📂 mlops/
│   ├── algorithms.py        # 50+ algorithm registry (classification + regression)
│   ├── datasets.py          # Dataset loaders (sklearn built-ins + California Housing)
│   ├── trainer.py           # Training & AutoML job management
│   └── airflow_runner.py    # Apache Airflow DAG trigger & watcher
│
├── 📂 pipelines/
│   ├── dag_engine.py        # Built-in DAG execution engine (fallback)
│   └── pipeline_defs.py     # Training / Retraining / Data pipeline definitions
│
├── 📂 dags/                 # Apache Airflow DAG files (parsed by scheduler)
│
├── 📂 templates/
│   ├── base.html            # Base layout: sidebar + topnav + theme toggle
│   ├── pipeline.html        # Pipeline Studio (home page — interactive DAG canvas)
│   ├── automl.html          # AutoML experiment launcher
│   └── models.html          # Model Registry browser
│
├── 📂 static/
│   ├── css/style.css        # Global styles + dark/light theme CSS variables
│   └── js/app.js            # Shared JS (toasts, theme switching)
│
├── 📄 app.py                # Flask application entry point + all API routes
├── 📄 Dockerfile            # Single-container image (Flask + Airflow)
├── 📄 start.sh              # Startup: Airflow scheduler → Gunicorn Flask
├── 📄 requirements.txt      # Python dependencies
└── 📄 README.md
```

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## Disclaimer

<span style="color:red">This project is developed strictly for educational and research purposes and does not constitute professional advice of any kind. All datasets used are either synthetically generated or publicly available — no real user data is stored. This software is provided "as is" without warranty of any kind; use at your own risk.</span>

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:4f46e5&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/AutoMLOps?style=social)](https://github.com/mnoorchenar/AutoMLOps)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/AutoMLOps?style=social)](https://github.com/mnoorchenar/AutoMLOps/fork)

<sub>The name "AutoMLOps" is used purely for academic and research purposes. Any similarity to existing company names, products, or trademarks is entirely coincidental and unintentional. This project has no affiliation with any commercial entity.</sub>

</div>
