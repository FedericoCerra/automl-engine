---
title: AutoML API Engine
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# 🚀 AutoML Master Engine & API

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Async-green)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An **AutoML engine** powered by an asynchronous FastAPI backend.  
Automatically detects ML task types, performs intelligent hyperparameter tuning with Optuna, and is fully containerized for deployment on **Hugging Face Spaces, AWS, or any Docker environment**.

---

# Features

## Intelligent Task Auto-Detection
Automatically analyzes your target column to determine:
- **Classification**
- **Regression**

No manual configuration required.

---

## Fully Asynchronous Architecture
Built with **FastAPI background workers**, allowing:
- Large dataset uploads
- Long training jobs
- No HTTP timeout issues
- Non-blocking API behavior

---

## Live Training Progress
Custom Optuna callbacks stream:
- Trial number
- Optimization progress
- Evaluation metrics
- Final model accuracy

Accessible via the `/status/{job_id}` endpoint.

---

## Dual Execution Modes

###  Web API Mode
For production deployment and remote access.

### 2Command Line Interface (CLI)
For fast local experimentation and debugging.

---

# Installation

```bash
git clone https://github.com/yourusername/automl-api-engine.git
cd automl-api-engine
pip install -r requirements.txt
```

---

# Running the Web API

Start the development server:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 7860
```

Open interactive documentation:

```
http://localhost:7860/docs
```

---

# 🔌 API Endpoints

## `POST /train`
Upload:
- `.csv` dataset  
- Target column  

Returns:
- Unique `job_id`

---

## `GET /status/{job_id}`
Returns:
- Current Optuna trial
- Training progress
- Final accuracy / metric

---

## `GET /download/{job_id}`
Download:
- Serialized `.pkl` pipeline

---

## `POST /predict`
Upload:
- New dataset
- Existing `job_id`

Returns:
- JSON array of predictions

---

# Running via CLI

## Basic Training

```bash
python main.py --data datasets/titanic/train.csv --target Survived --trials 20 --task classification
```



---

# Docker Deployment

## Build Image

```bash
docker build -t automl-engine .
```

## Run Container

```bash
docker run -p 7860:7860 automl-engine
```

API will be available at:

```
http://localhost:7860/docs
```

---

# Tech Stack

- FastAPI
- Optuna
- Scikit-Learn
- Pandas
- Docker
- Uvicorn

---

# License

MIT License

---

# Author

Federico Cerra
GitHub: https://github.com/FedericoCerra
