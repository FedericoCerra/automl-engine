import streamlit as st
import pandas as pd
import requests
import re
import io
import json
import os
import uuid

# --- CONFIGURATION ---
API_URL = "https://fedede-automl-engine.hf.space"

st.set_page_config(page_title="AutoML Master", layout="centered")

# --- SESSION MANAGEMENT ---
if "uid" in st.query_params:
    user_id = st.query_params["uid"]
else:
    user_id = str(uuid.uuid4())[:8]
    st.query_params["uid"] = user_id

jobs_file = f"jobs_{user_id}.json"

if "jobs" not in st.session_state:
    if os.path.exists(jobs_file):
        try:
            with open(jobs_file, "r") as f:
                st.session_state.jobs = json.load(f)
        except:
            st.session_state.jobs = []
    else:
        st.session_state.jobs = []

st.title("AutoML Master Engine")
st.markdown("Drag, drop, and train Machine Learning models instantly.")

# --- SIDEBAR ---
with st.sidebar:
    st.write(f"**User Session:** `{user_id}`")
    if st.button("Clear History", type="primary"):
        if os.path.exists(jobs_file):
            os.remove(jobs_file)
        st.session_state.jobs = []
        st.rerun()

tab_train, tab_predict = st.tabs(["Train Model", "Predict"])

# ==========================================
# DASHBOARD FRAGMENT
# ==========================================
@st.fragment(run_every=2)
def render_dashboard():
    st.subheader("Your Training History")
    
    if not st.session_state.jobs:
        st.info("No models training yet. Upload a CSV above to begin.")
        return
        
    for job in reversed(st.session_state.jobs):
        try:
            status_res = requests.get(f"{API_URL}/status/{job}").json()
        except:
            continue 
            
        status_text = status_res.get("status", "Unknown")
        score = status_res.get("score")
        start = status_res.get("start_time", "N/A")
        end = status_res.get("end_time", "N/A")
        filename = status_res.get("filename", "Unknown File")
        target = status_res.get("target", "Unknown Target")
        
        is_active = "COMPLETED" not in status_text and "FAILED" not in status_text
        
        with st.expander(f"Model: {job} ({filename}) | Status: {status_text}", expanded=is_active):
            
            # 1. Hide the "Finished" text if the engine is still running
            if is_active or end in [None, "N/A"]:
                st.caption(f"**ID:** `{job}` | **Started:** {start}")
            else:
                st.caption(f"**ID:** `{job}` | **Started:** {start} | **Finished:** {end}")
            
            with st.expander("Training Data Details"):
                st.write(f"**File Name:** `{filename}`")
                st.write(f"**Target Column:** `{target}`")
            
            if is_active:
                match = re.search(r'\((\d+)%\)', status_text)
                if match:
                    pct = int(match.group(1))
                    # 2. Add the percentage text directly above the progress bar
                    st.write(f"**Progress:** {pct}%")
                    st.progress(pct)
                else:
                    st.write("**Progress:** Starting...")
                    st.progress(0)
            else:
                if score:
                    st.write(f"**Final Score:** {score:.4f} ({status_res.get('metric')})")
                
                if status_text == "COMPLETED":
                    dl_res = requests.get(f"{API_URL}/download/{job}")
                    if dl_res.status_code == 200:
                        st.download_button(
                            label="Download Model (.pkl)",
                            data=dl_res.content,
                            file_name=f"model_{job[:8]}.pkl",
                            key=f"dl_{job}"
                        )
# ==========================================
# SECTION 1: TRAIN
# ==========================================
with tab_train:
    st.header("1. Train a New Model")
    
    train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"], accept_multiple_files=False, key="train_uploader")
    
    if train_file is not None:
        df_train = pd.read_csv(train_file)
        
        with st.expander("⚙️ Engine Settings", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                last_col_index = len(df_train.columns) - 1
                target_col = st.selectbox("Target Column to Predict:", df_train.columns, index=last_col_index)
            with col2:
                trials = st.number_input("Number of Trials:", min_value=1, max_value=100, value=20)
            
        if st.button("Start Training", type="primary"):
            train_file.seek(0)
            response = requests.post(
                f"{API_URL}/train",
                data={"target": target_col, "trials": trials, "task": "auto"},
                files={"file": (train_file.name, train_file, "text/csv")}
            )
            
            if response.status_code == 200:
                job_id = response.json()["job_id"]
                st.session_state.jobs.append(job_id)
                # Save job list to user-specific file
                with open(jobs_file, "w") as f:
                    json.dump(st.session_state.jobs, f)
            else:
                st.error(f"Error: {response.text}")

    st.divider()
    # Call the fragment function to render the dashboard
    render_dashboard()
# ==========================================
# SECTION 2: PREDICT
# ==========================================
with tab_predict:
    st.header("2. Generate Predictions")
    
    input_method = st.radio("Model Source:", ["Select from History", "Upload Model File (.pkl)"], horizontal=True)
    
    predict_job_id = None
    uploaded_model = None
    
    if input_method == "Select from History":
        job_options = list(reversed(st.session_state.jobs))
        if job_options:
            predict_job_id = st.selectbox(
                "Select a trained Model:", 
                job_options, 
                format_func=lambda x: f"Model ID: {x[:8]}"
            )
        else:
            st.info("No recent models found. Train one first or upload a .pkl file.")
    else:
        uploaded_model = st.file_uploader("Drag & Drop your .pkl model here", type=["pkl"])
    
    predict_file = st.file_uploader("Upload New Data for Prediction (CSV)", type=["csv"], accept_multiple_files=False, key="predict_uploader")
    
    if predict_file is not None:
        df_predict = pd.read_csv(predict_file)
        
        # 1. Toggleable Test Data Preview
        with st.expander("Show Test Data Preview"):
            st.dataframe(df_predict.head())
        
        # Create a unique key for session state based on input method
        if uploaded_model:
            pred_state_key = f"prediction_upload_{uploaded_model.name}"
        else:
            pred_state_key = f"prediction_{predict_job_id}"
        
        if st.button("Generate Predictions", type="primary"):
            if not predict_job_id and not uploaded_model:
                st.warning("Please select a model or upload a .pkl file.")
            else:
                with st.spinner("Waking up model and predicting..."):
                    predict_file.seek(0)
                    
                    files = {"file": (predict_file.name, predict_file, "text/csv")}
                    data = {}
                    
                    if uploaded_model:
                        files["model_file"] = (uploaded_model.name, uploaded_model, "application/octet-stream")
                    else:
                        data["job_id"] = predict_job_id

                    response = requests.post(
                        f"{API_URL}/predict",
                        data=data,
                        files=files
                    )
                    
                    if response.status_code == 200:
                        st.session_state[pred_state_key] = response.content
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Failed to generate predictions.')}")
        
        # 2. Check memory for predictions
        if pred_state_key in st.session_state:
            
            # Read the raw API bytes back into a Pandas DataFrame for the UI
            csv_bytes = st.session_state[pred_state_key]
            df_results = pd.read_csv(io.BytesIO(csv_bytes))
            
            # 3. Toggleable Prediction Preview (Starts open by default)
            with st.expander("Show Predictions Preview", expanded=True):
                st.dataframe(df_results.head())
            
            # 4. The Permanent Download Button
            st.download_button(
                label="Download Predictions (CSV)",
                data=csv_bytes,
                file_name=f"predictions.csv",
                mime="text/csv"
            )