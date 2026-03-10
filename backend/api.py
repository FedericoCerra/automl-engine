import os
import uuid
import logging
import pandas as pd
import joblib
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
import io
from datetime import datetime
import shutil

from model_selector import AutoModelSelector

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("automl-api")

# --- CONSTANTS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "api_uploads")
MODEL_DIR = os.path.join(BASE_DIR, "api_models")
MAX_FILE_SIZE_MB = 100
MAX_TRIALS = 100
MIN_TRIALS = 1
ALLOWED_TASKS = {"auto", "classification", "regression"}
ALLOWED_SCORING = {
    "auto", "accuracy", "f1", "f1_macro", "roc_auc", "precision", "recall",
    "r2", "neg_mean_squared_error"
}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI(title="AutoML Master API", description="Drag, Drop, and Train ML Models")

@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")

# --- IN-MEMORY DATABASES ---
job_database: Dict[str, str] = {}
results_database: Dict[str, Dict] = {}
times_database: Dict[str, Dict] = {}
meta_database: Dict[str, Dict] = {}

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
    best_params: dict | None = None
    dataset_stats: dict | None = None
    task_type: str | None = None
    sample_data: dict | None = None
    baseline_score: float | None = None
    all_metrics: dict | None = None

class PredictionResponse(BaseModel):
    job_id: str
    total_predictions: int
    predictions: List[str | int | float]

class DatasetValidation(BaseModel):
    columns: List[str]
    dtypes: Dict[str, str]
    shape: List[int]
    missing_pct: Dict[str, float]
    numeric_columns: List[str]
    categorical_columns: List[str]
    suggested_target: str | None = None

# --- HELPERS ---
def _model_path(job_id: str) -> str:
    return os.path.join(MODEL_DIR, f"{job_id}_model.pkl")

def _shap_path(job_id: str) -> str:
    return os.path.join(MODEL_DIR, f"{job_id}_shap.png")

def _build_job_response(job_id: str) -> dict:
    """Build a unified job status response dict from all databases."""
    status = job_database.get(job_id, "Job ID not found.")
    results = results_database.get(job_id, {})
    times = times_database.get(job_id, {})
    meta = meta_database.get(job_id, {})
    return {
        "job_id": job_id,
        "status": status,
        "score": results.get("score"),
        "metric": results.get("metric"),
        "feature_importance": results.get("feature_importance"),
        "start_time": times.get("start"),
        "end_time": times.get("end"),
        "filename": meta.get("filename"),
        "target": meta.get("target"),
        "shap_enabled": meta.get("shap", False),
        "best_params": results.get("best_params"),
        "dataset_stats": results.get("dataset_stats"),
        "task_type": results.get("task_type"),
        "sample_data": results.get("sample_data"),
        "baseline_score": results.get("baseline_score"),
        "all_metrics": results.get("all_metrics"),
    }

def _sanitize_filename(filename: str) -> str:
    """Strip path components to prevent directory traversal."""
    return os.path.basename(filename).replace("..", "")

# --- BACKGROUND WORKER ---
def run_training_task(job_id: str, file_path: str, target: str, trials: int, task: str, scoring: str, shap_enabled: bool):
    times_database[job_id] = {"start": datetime.now().strftime("%H:%M:%S"), "end": None}
    try:
        if job_id not in job_database:
            return

        job_database[job_id] = "Loading data..."
        df = pd.read_csv(file_path)

        rows, cols = df.shape
        dataset_stats = {"rows": rows, "columns": cols}

        # Validate dataset
        if df.empty:
            job_database[job_id] = "FAILED: Uploaded CSV is empty."
            return
        if rows < 5:
            job_database[job_id] = "FAILED: Dataset too small (minimum 5 rows required for cross-validation)."
            return

        sample_data = None
        if not df.empty:
            sample_data = df.head(1).replace({float('nan'): None}).to_dict(orient='records')[0]

        if target not in df.columns:
            job_database[job_id] = f"Error: Target '{target}' not found in CSV."
            return

        X = df.drop(columns=[target])
        y = df[target]

        # Validate features
        if X.empty or X.shape[1] == 0:
            job_database[job_id] = "FAILED: No feature columns found after removing target."
            return
        if y.nunique() < 2:
            job_database[job_id] = "FAILED: Target column has only one unique value — nothing to learn."
            return

        # Warn about high missing data
        missing_pct = (X.isnull().sum() / len(X) * 100)
        high_missing = missing_pct[missing_pct > 90].index.tolist()
        if high_missing:
            logger.warning(f"Job {job_id}: Columns with >90% missing data: {high_missing}")

        job_database[job_id] = "Training..."
        automl = AutoModelSelector(n_trials=trials, task=task, scoring=scoring)

        def update_api_status(current_trial, total_trials):
            if job_id not in job_database:
                return
            percentage = int((current_trial / total_trials) * 100)
            job_database[job_id] = f"Training... Trial {current_trial}/{total_trials} ({percentage}%)"

        automl.fit(X, y, progress_callback=update_api_status)

        # Extract feature importance
        feat_importance = None
        try:
            if hasattr(automl.best_pipeline, "named_steps"):
                pipeline = automl.best_pipeline
                model_step = pipeline.named_steps["model"]

                if hasattr(model_step, "feature_importances_"):
                    importances = model_step.feature_importances_
                    preprocessor = pipeline.named_steps["preprocessor"]
                    feature_eng = pipeline.named_steps["feature_eng"]
                    feature_names = preprocessor.get_feature_names_out()
                    feature_names = feature_eng.get_feature_names_out(feature_names)

                    if len(feature_names) == len(importances):
                        feat_importance = {name: float(imp) for name, imp in zip(feature_names, importances)}
        except (AttributeError, ValueError, IndexError) as e:
            logger.warning(f"Job {job_id}: Feature importance extraction failed: {e}")

        model_path = _model_path(job_id)
        joblib.dump(automl, model_path)

        if shap_enabled:
            job_database[job_id] = "Calculating SHAP values..."
            shap_out = _shap_path(job_id)
            try:
                automl.explain(X, y, shap_out)
            except (ValueError, TypeError, RuntimeError) as e:
                logger.warning(f"Job {job_id}: SHAP calculation failed: {e}")

        os.remove(file_path)

        if job_id not in job_database:
            return

        results_database[job_id] = {
            "score": abs(automl.best_score),
            "metric": automl.scoring,
            "feature_importance": feat_importance,
            "best_params": automl.study.best_params if automl.study else None,
            "dataset_stats": dataset_stats,
            "task_type": automl.task,
            "sample_data": sample_data,
            "baseline_score": getattr(automl, "baseline_score", None),
            "all_metrics": getattr(automl, "all_metrics", None),
        }

        job_database[job_id] = "COMPLETED"
        times_database[job_id]["end"] = datetime.now().strftime("%H:%M:%S")
        logger.info(f"Job {job_id} completed. Score: {abs(automl.best_score):.4f}")

    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "... (Error truncated. Check configuration.)"
        job_database[job_id] = f"FAILED: {error_msg}"
        times_database[job_id]["end"] = datetime.now().strftime("%H:%M:%S")
        logger.error(f"Job {job_id} failed: {error_msg}")
        
# --- ENDPOINTS ---

@app.get("/health")
def health_check() -> dict:
    """Health check for load balancers and monitoring."""
    active_jobs = sum(1 for s in job_database.values() if "Training" in s or "PENDING" in s)
    return {"status": "ok", "active_jobs": active_jobs, "total_jobs": len(job_database)}

@app.post("/validate-csv")
async def validate_csv(file: UploadFile = File(...)) -> DatasetValidation:
    """Validate a CSV file and return column info, types, and missing data stats."""
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    missing = (df.isnull().sum() / len(df) * 100).round(2).to_dict()

    # Suggest last column as target
    suggested = df.columns[-1] if len(df.columns) > 1 else None

    return DatasetValidation(
        columns=df.columns.tolist(),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        shape=[df.shape[0], df.shape[1]],
        missing_pct=missing,
        numeric_columns=num_cols,
        categorical_columns=cat_cols,
        suggested_target=suggested,
    )

@app.post("/train", response_model=TrainingTicketResponse)
async def start_training(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target: str = Form(...),
    trials: int = Form(20),
    task: str = Form("auto"),
    scoring: str = Form("auto"),
    shap: bool = Form(False),
    uid: str = Form(...)
) -> TrainingTicketResponse:
    # Input validation
    if task not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task '{task}'. Allowed: {ALLOWED_TASKS}")
    if scoring not in ALLOWED_SCORING:
        raise HTTPException(status_code=400, detail=f"Invalid scoring '{scoring}'. Allowed: {ALLOWED_SCORING}")
    trials = max(MIN_TRIALS, min(trials, MAX_TRIALS))

    # Sanitize filename to prevent path traversal
    safe_filename = _sanitize_filename(file.filename)
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    # File size check (read into memory, check size)
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File too large ({size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB.")

    job_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe_filename}")

    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    job_database[job_id] = "PENDING - Waiting in queue"
    times_database[job_id] = {"start": datetime.now().strftime("%H:%M:%S"), "end": None}
    meta_database[job_id] = {"filename": safe_filename, "target": target, "shap": shap, "uid": uid}
    background_tasks.add_task(run_training_task, job_id, file_path, target, trials, task, scoring, shap)

    logger.info(f"Job {job_id} queued for user {uid} (file={safe_filename}, target={target}, trials={trials})")
    return TrainingTicketResponse(
        message="Ticket generated! Training started in background.",
        job_id=job_id
    )

@app.get("/jobs/{uid}")
def get_jobs_by_user(
    uid: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """Get all jobs for a user, with pagination."""
    user_jobs = {
        job_id: _build_job_response(job_id)
        for job_id, meta in meta_database.items()
        if meta.get("uid") == uid
    }
    # Apply pagination (dict preserves insertion order in Python 3.7+)
    keys = list(user_jobs.keys())[offset:offset + limit]
    return {k: user_jobs[k] for k in keys}

@app.get("/status/{job_id}", response_model=JobStatusResponse)
def check_status(job_id: str) -> JobStatusResponse:
    if job_id not in job_database:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    data = _build_job_response(job_id)
    return JobStatusResponse(**data)

@app.get("/download/{job_id}")
def download_model(job_id: str):
    path = _model_path(job_id)
    if os.path.exists(path):
        return FileResponse(path=path, filename=f"trained_model_{job_id}.pkl")
    raise HTTPException(status_code=404, detail="Model not found or still training.")

@app.get("/shap/{job_id}")
def get_shap_plot(job_id: str):
    path = _shap_path(job_id)
    if os.path.exists(path):
        return FileResponse(path=path, media_type="image/png")
    raise HTTPException(status_code=404, detail="SHAP plot not found.")

@app.delete("/job/{job_id}")
def delete_job(job_id: str) -> dict:
    job_database.pop(job_id, None)
    results_database.pop(job_id, None)
    times_database.pop(job_id, None)
    meta_database.pop(job_id, None)

    for path in [_model_path(job_id), _shap_path(job_id)]:
        if os.path.exists(path):
            os.remove(path)

    logger.info(f"Job {job_id} deleted.")
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
        path = _model_path(job_id)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Model not found for this Job ID.")
        automl = joblib.load(path)
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