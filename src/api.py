import os
import uuid
import pandas as pd
import joblib
from typing import List
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
import sys
import io

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from src.model_selector import AutoModelSelector

app = FastAPI(title="AutoML Master API", description="Drag, Drop, and Train ML Models")

@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")
os.makedirs("api_uploads", exist_ok=True)
os.makedirs("api_models", exist_ok=True)

job_database = {}
results_database = {}

# PYDANTIC SCHEMAS 
class TrainingTicketResponse(BaseModel):
    message: str
    job_id: str
      


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    score: float | None = Field(default=None, example=0.85)
    metric: str | None = Field(default=None, example="accuracy")  

class PredictionResponse(BaseModel):
    job_id: str
    total_predictions: int
    predictions: List[str | int | float] # Can handle text classes or numerical regression

# BACKGROUND WORKER
def run_training_task(job_id: str, file_path: str, target: str, trials: int, task: str):
    try:
        job_database[job_id] = "Loading data..."
        df = pd.read_csv(file_path)
        
        if target not in df.columns:
            job_database[job_id] = f"Error: Target '{target}' not found in CSV."
            return

        X = df.drop(columns=[target])
        y = df[target]
        
        job_database[job_id] = "Training..."        
        automl = AutoModelSelector(n_trials=trials, task=task, scoring='auto')
        
        def update_api_status(current_trial, total_trials):
            percentage = int((current_trial / total_trials) * 100)
            job_database[job_id] = f"Training... Trial {current_trial}/{total_trials} ({percentage}%)"
        
        automl.fit(X, y, progress_callback=update_api_status)
        
        model_path = f"api_models/{job_id}_model.pkl"
        joblib.dump(automl, model_path)
        
        os.remove(file_path)
        
        results_database[job_id] = {
            "score": abs(automl.best_score), # abs() removes the negative sign from MSE
            "metric": automl.scoring
        }
        
        job_database[job_id] = "COMPLETED"
    
    except Exception as e:
        job_database[job_id] = f"FAILED: {str(e)}"

# ENDPOINTS
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
    
    # Check if there are final results for this job
    results = results_database.get(job_id, {})
    
    return JobStatusResponse(
        job_id=job_id, 
        status=status,
        score=results.get("score"),
        metric=results.get("metric")
    )

@app.get("/download/{job_id}")
def download_model(job_id: str):
    model_path = f"api_models/{job_id}_model.pkl"
    if os.path.exists(model_path):
        return FileResponse(path=model_path, filename=f"trained_model_{job_id}.pkl")
    
    # Using HTTPException is cleaner than returning a fake error dict
    raise HTTPException(status_code=404, detail="Model not found or still training.")


@app.post("/predict")
async def predict(job_id: str = Form(...), file: UploadFile = File(...)):

    model_path = f"api_models/{job_id}_model.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found for this Job ID.")
    
    automl = joblib.load(model_path)
    
    df = pd.read_csv(file.file)
    
    # Run predictions
    predictions = automl.predict(df)
    
    # Smart column detection for Kaggle/Standard formats
    id_col = None
    for candidate in ["Id", "id", "index"]:
        if candidate in df.columns:
            id_col = candidate
            break
    
    if id_col:
        output_df = pd.DataFrame({id_col: df[id_col], "Target": predictions})
    else:
        output_df = pd.DataFrame({
        "id": range(1, len(predictions) + 1),
        "Prediction": predictions
    })

    # Convert to CSV in memory using streams for speed(uses RAM instead of hard disk)
    stream = io.StringIO()
    output_df.to_csv(stream, index=False)
    
    # Return as a downloadable file
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = f"attachment; filename=predictions_{job_id}.csv"
    
    return response