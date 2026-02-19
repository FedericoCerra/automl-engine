import os
import uuid
import pandas as pd
import joblib
from typing import List
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.model_selector import AutoModelSelector

app = FastAPI(title="AutoML Master API", description="Drag, Drop, and Train ML Models")

os.makedirs("api_uploads", exist_ok=True)
os.makedirs("api_models", exist_ok=True)
job_database = {}

# ==========================================
# 🛡️ PYDANTIC SCHEMAS (The Contract)
# ==========================================
class TrainingTicketResponse(BaseModel):
    message: str
    job_id: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str

class PredictionResponse(BaseModel):
    job_id: str
    total_predictions: int
    predictions: List[str | int | float] # Can handle text classes or numerical regression

# ==========================================
# ⚙️ BACKGROUND WORKER
# ==========================================
def run_training_task(job_id: str, file_path: str, target: str, trials: int, task: str):
    try:
        job_database[job_id] = "Loading data..."
        df = pd.read_csv(file_path)
        
        if target not in df.columns:
            job_database[job_id] = f"Error: Target '{target}' not found in CSV."
            return

        X = df.drop(columns=[target])
        y = df[target]

        automl = AutoModelSelector(n_trials=trials, task=task, scoring='auto')
        automl.fit(X, y)
        
        model_path = f"api_models/{job_id}_model.pkl"
        joblib.dump(automl, model_path)
        
        os.remove(file_path)
        job_database[job_id] = "COMPLETED"
    
    except Exception as e:
        job_database[job_id] = f"FAILED: {str(e)}"

# ==========================================
# 🚀 ENDPOINTS
# ==========================================

# Notice response_model=TrainingTicketResponse
@app.post("/train", response_model=TrainingTicketResponse)
async def start_training(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target: str = Form(...),
    trials: int = Form(20),
    task: str = Form("auto")
):
    job_id = str(uuid.uuid4())
    file_path = f"api_uploads/{job_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    job_database[job_id] = "PENDING - Waiting in queue"
    background_tasks.add_task(run_training_task, job_id, file_path, target, trials, task)    
    return TrainingTicketResponse(
        message="Ticket generated! Training started in background.", 
        job_id=job_id
    )

@app.get("/status/{job_id}", response_model=JobStatusResponse)
def check_status(job_id: str):
    status = job_database.get(job_id, "Job ID not found.")
    return JobStatusResponse(job_id=job_id, status=status)

@app.get("/download/{job_id}")
def download_model(job_id: str):
    model_path = f"api_models/{job_id}_model.pkl"
    if os.path.exists(model_path):
        return FileResponse(path=model_path, filename=f"trained_model_{job_id}.pkl")
    
    # Using HTTPException is cleaner than returning a fake error dict
    raise HTTPException(status_code=404, detail="Model not found or still training.")

@app.post("/predict", response_model=PredictionResponse)
async def make_predictions(
    job_id: str = Form(...),
    file: UploadFile = File(...)
):
    model_path = f"api_models/{job_id}_model.pkl"
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found. Please provide a valid Job ID.")
        
    automl = joblib.load(model_path)
    df_test = pd.read_csv(file.file)
    predictions = automl.predict(df_test)
    
    return PredictionResponse(
        job_id=job_id,
        total_predictions=len(predictions),
        predictions=predictions.tolist()
    )