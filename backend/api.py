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
from datetime import datetime
import shutil
current_dir = os.path.dirname(os.path.abspath(__file__))
from model_selector import AutoModelSelector

app = FastAPI(title="AutoML Master API", description="Drag, Drop, and Train ML Models")

@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Force the folders to be built strictly inside that directory
UPLOAD_DIR = os.path.join(BASE_DIR, "api_uploads")
MODEL_DIR = os.path.join(BASE_DIR, "api_models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

job_database = {}
results_database = {}
times_database = {}
meta_database = {}

# PYDANTIC SCHEMAS 
class TrainingTicketResponse(BaseModel):
    message: str
    job_id: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    score: float | None = Field(default=None, example=0.85)
    metric: str | None = Field(default=None, example="accuracy")  
    feature_importance: dict | None = None
    start_time: str | None = None
    end_time: str | None = None
    filename: str | None = None
    target: str | None = None
    shap_enabled: bool = False

class PredictionResponse(BaseModel):
    job_id: str
    total_predictions: int
    predictions: List[str | int | float] # Can handle text classes or numerical regression

# BACKGROUND WORKER
def run_training_task(job_id: str, file_path: str, target: str, trials: int, task: str, shap_enabled: bool):
    times_database[job_id] = {"start": datetime.now().strftime("%H:%M:%S"), "end": None}
    try:
        # Stop if job was deleted
        if job_id not in job_database: return

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
            # Stop if job was deleted
            if job_id not in job_database: return
            percentage = int((current_trial / total_trials) * 100)
            job_database[job_id] = f"Training... Trial {current_trial}/{total_trials} ({percentage}%)"
        
        automl.fit(X, y, progress_callback=update_api_status)
        
        # Extract Feature Importance (if supported by the model)
        feat_importance = None
        try:
            if hasattr(automl.best_pipeline, "named_steps"):
                pipeline = automl.best_pipeline
                model_step = pipeline.named_steps["model"]
                
                # Check if model supports feature importance (e.g., Trees)
                if hasattr(model_step, "feature_importances_"):
                    importances = model_step.feature_importances_
                    
                    # Get feature names from pipeline transformers
                    preprocessor = pipeline.named_steps["preprocessor"]
                    feature_eng = pipeline.named_steps["feature_eng"]
                    
                    feature_names = preprocessor.get_feature_names_out()
                    feature_names = feature_eng.get_feature_names_out(feature_names)
                    
                    if len(feature_names) == len(importances):
                        feat_importance = {name: float(imp) for name, imp in zip(feature_names, importances)}
        except Exception as e:
            print(f"Feature importance extraction warning: {e}")

        model_path = f"api_models/{job_id}_model.pkl"
        joblib.dump(automl, model_path)
        
        if shap_enabled:
            job_database[job_id] = "Calculating SHAP values..."
            shap_path = f"api_models/{job_id}_shap.png"
            try:
                automl.explain(X, y, shap_path)
            except Exception as e:
                print(f"SHAP calculation failed: {e}")

        os.remove(file_path)
        
        # Stop if job was deleted
        if job_id not in job_database: return

        results_database[job_id] = {
            "score": abs(automl.best_score), # abs() removes the negative sign from MSE
            "metric": automl.scoring,
            "feature_importance": feat_importance
        }
        
        job_database[job_id] = "COMPLETED"
        times_database[job_id]["end"] = datetime.now().strftime("%H:%M:%S") # Log end time
        
    except Exception as e:
        job_database[job_id] = f"FAILED: {str(e)}"
        times_database[job_id]["end"] = datetime.now().strftime("%H:%M:%S")
        
# ENDPOINTS
@app.post("/train", response_model=TrainingTicketResponse)
async def start_training(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target: str = Form(...),
    trials: int = Form(20),
    task: str = Form("auto"),
    shap: bool = Form(False),
    uid: str = Form(...)
):
    # Generate the unique ID
    job_id = str(uuid.uuid4())[:8]
    
    # Safely join the absolute path with the filename
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    job_database[job_id] = "PENDING - Waiting in queue"
    times_database[job_id] = {"start": datetime.now().strftime("%H:%M:%S"), "end": None}
    meta_database[job_id] = {"filename": file.filename, "target": target, "shap": shap, "uid": uid}
    background_tasks.add_task(run_training_task, job_id, file_path, target, trials, task, shap)    
    return TrainingTicketResponse(
        message="Ticket generated! Training started in background.", 
        job_id=job_id
    )

@app.get("/jobs/{uid}")
def get_jobs_by_user(uid: str):
    response = {}
    # Iterate over meta_database to find jobs for this user
    for job_id, meta in meta_database.items():
        if meta.get("uid") == uid:
            status = job_database.get(job_id, "Unknown")
            results = results_database.get(job_id, {})
            times = times_database.get(job_id, {})
            
            response[job_id] = {
                "job_id": job_id, 
                "status": status,
                "score": results.get("score"),
                "metric": results.get("metric"),
                "feature_importance": results.get("feature_importance"),
                "start_time": times.get("start"), 
                "end_time": times.get("end"),
                "filename": meta.get("filename"),
                "target": meta.get("target"),
                "shap_enabled": meta.get("shap", False)
            }
    return response

@app.get("/status/{job_id}", response_model=JobStatusResponse)
def check_status(job_id: str):
    status = job_database.get(job_id, "Job ID not found.")
    
    # Check if there are final results for this job
    results = results_database.get(job_id, {})
    
    times = times_database.get(job_id, {})
    meta = meta_database.get(job_id, {})
    
    return JobStatusResponse(
        job_id=job_id, 
        status=status,
        score=results.get("score"),
        metric=results.get("metric"),
        feature_importance=results.get("feature_importance"),
        start_time=times.get("start"), 
        end_time=times.get("end"),
        filename=meta.get("filename"),
        target=meta.get("target"),
        shap_enabled=meta.get("shap", False)
    )

@app.get("/download/{job_id}")
def download_model(job_id: str):
    model_path = f"api_models/{job_id}_model.pkl"
    if os.path.exists(model_path):
        return FileResponse(path=model_path, filename=f"trained_model_{job_id}.pkl")
    
    # Using HTTPException is cleaner than returning a fake error dict
    raise HTTPException(status_code=404, detail="Model not found or still training.")

@app.get("/shap/{job_id}")
def get_shap_plot(job_id: str):
    plot_path = f"api_models/{job_id}_shap.png"
    if os.path.exists(plot_path):
        return FileResponse(path=plot_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="SHAP plot not found.")

@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    # Remove from databases
    job_database.pop(job_id, None)
    results_database.pop(job_id, None)
    times_database.pop(job_id, None)
    meta_database.pop(job_id, None)
    
    # Remove model file
    model_path = f"api_models/{job_id}_model.pkl"
    if os.path.exists(model_path):
        os.remove(model_path)
    
    shap_path = f"api_models/{job_id}_shap.png"
    if os.path.exists(shap_path):
        os.remove(shap_path)
        
    return {"message": "Job deleted"}

@app.post("/predict")
async def predict(
    job_id: str = Form(None), 
    file: UploadFile = File(...),
    model_file: UploadFile = File(None)
):
    # 1. Determine source of the model
    if model_file:
        try:
            automl = joblib.load(model_file.file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid model file: {str(e)}")
    elif job_id:
        model_path = f"api_models/{job_id}_model.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found for this Job ID.")
        automl = joblib.load(model_path)
    else:
        raise HTTPException(status_code=400, detail="Please provide a Job ID or upload a Model file.")
    
    df = pd.read_csv(file.file)
    
    # Run predictions
    predictions = automl.predict(df)
    
    # 👇 NEW: Extract the memorized target name (Fallback to "Prediction" just in case)
    target_col_name = getattr(automl, "target_name", "Prediction")
    
    # Smart column detection for Kaggle/Standard formats
    id_col = None
    for candidate in ["Id", "id", "index", "PassengerId"]:
        if candidate in df.columns:
            id_col = candidate
            break
    
    if id_col:
        # Use the dynamic target_col_name here
        output_df = pd.DataFrame({id_col: df[id_col].values, target_col_name: predictions})
    else:
        # Use the dynamic target_col_name here too
        output_df = pd.DataFrame({
            "id": range(1, len(predictions) + 1),
            target_col_name: predictions
        })

    # Convert to CSV in memory
    stream = io.StringIO()
    output_df.to_csv(stream, index=False)
    
    # Return as a downloadable file
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = f"attachment; filename=predictions_{job_id}.csv"
    
    return response