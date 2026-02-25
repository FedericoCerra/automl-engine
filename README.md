---
title: AutoML API Engine
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# 🚀 AutoML Master Engine & Streamlit Dashboard

[![FastAPI](https://img.shields.io/badge/FastAPI-Engine-green)](https://huggingface.co/spaces/fedede/automl-engine)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)](https://huggingface.co/spaces/fedede/automl-frontend)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-orange)](https://optuna.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](https://opensource.org/licenses/MIT)

A full-stack **AutoML platform** featuring an asynchronous FastAPI backend engine and an interactive Streamlit frontend.  
Automatically detects ML task types, performs intelligent hyperparameter tuning with Optuna, and is fully containerized for deployment on **Hugging Face Spaces, AWS, or any Docker environment**.

---

# ☁️ Live Cloud Demos

Don't want to run it locally? Try the live cloud versions hosted on Hugging Face Spaces:

* **🎨 Streamlit Dashboard (Frontend):** [Test the UI Live Here](https://huggingface.co/spaces/fedede/automl-frontend)
* **⚙️ FastAPI Engine (Backend Docs):** [View the API Swagger UI Here](https://huggingface.co/spaces/fedede/automl-engine)

---

# Project Structure (Monorepo)

The project is divided into two decoupled microservices:
- **`/backend`**: The FastAPI engine, Optuna workers, and ML logic.
- **`/frontend`**: The Streamlit user interface.

---

# Features

## Interactive No-Code Dashboard 
A fully integrated Streamlit UI that allows you to:
- Drag-and-drop CSV datasets
- Select target columns visually
- Monitor live Optuna training charts
- Download the final `.pkl` model with one click

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

Accessible via the `/status/{job_id}` endpoint or viewed live on the Streamlit UI.

---

## Execution Modes

### 1. Streamlit UI
For visual, no-code training and easy file downloads.

### 2. Web API Mode
For production deployment, remote access, and programmatic integration.

### 3. Command Line Interface (CLI)
For fast local experimentation and debugging.

---

# Installation

Clone the repository:
```bash
git clone [https://github.com/yourusername/automl-api-engine.git](https://github.com/yourusername/automl-api-engine.git)
cd automl-api-engine
```
### Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```
### Install Frontend Dependencies
```bash
cd ../frontend
pip install -r requirements.txt
```
---

# 🤖 CI/CD Pipeline (Automated Deployments)

This project features a zero-touch Continuous Deployment pipeline powered by **GitHub Actions**. 

Whenever new code is pushed to the `main` branch, the `.github/workflows/deploy.yml` workflow automatically triggers. Because this project uses a monorepo architecture, the pipeline uses a custom inline Python script leveraging the `huggingface_hub` library to separate and deploy the two microservices independently.

Here is exactly what happens under the hood during a `git push`:
1. **Environment Setup:** GitHub Actions spins up an Ubuntu runner and configures Python 3.10.
2. **Backend Deployment:** A native Python command (`HfApi().upload_folder()`) uploads only the contents of the `/backend` folder directly to the Hugging Face Docker Space, bypassing any complex Git manipulation or CLI PATH issues.
3. **Frontend Deployment:** The same Python method uploads the `/frontend` folder directly to the Hugging Face Streamlit Space.

### How to Deploy Updates
You do not need to manually upload files or build Docker images to push to production. Just commit your code:
```bash
git add .
git commit -m "..."
git push origin main
```
---
# 💻 Running Locally

Because the frontend and backend are decoupled, they run in separate terminals.

### 1. Start the Backend API
Open a terminal in the `backend/` folder:
```bash
uvicorn api:app --host 0.0.0.0 --port 7860 --reload
```
*Interactive API documentation will be at: `http://localhost:7860/docs`*

### 2. Start the Frontend UI
Open a second terminal in the `frontend/` folder:
```bash
streamlit run app.py
```
*The dashboard will automatically open in your browser.*

---

# 🔌 API Endpoints (Backend)

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
*(Run from inside the `backend/` folder)*

python main.py --data /datapath --target targetCol --trials n --task classification/regression

---

# Docker Deployment (Backend)

## Build Image
*(Run from inside the `backend/` folder)*

docker build -t automl-engine .

## Run Container

docker run -p 7860:7860 automl-engine

API will be available at:

http://localhost:7860/docs

---

# Tech Stack

- **Backend:** FastAPI, Optuna, Scikit-Learn, Pandas, Docker, Uvicorn
- **Frontend:** Streamlit, Requests, Plotly

---

# License

MIT License

---

# Author

Federico Cerra  
GitHub: [https://github.com/FedericoCerra](https://github.com/FedericoCerra)
