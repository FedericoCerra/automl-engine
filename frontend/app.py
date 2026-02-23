import streamlit as st
import pandas as pd
import requests
import re
import io

# --- CONFIGURATION ---
API_URL = "https://fedede-automl-engine.hf.space"
st.set_page_config(page_title="AutoML Master", layout="centered")

if "jobs" not in st.session_state:
    st.session_state.jobs = []

st.title("AutoML Master Engine")
st.markdown("Drag, drop, and train Machine Learning models instantly.")

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
        
        is_active = "COMPLETED" not in status_text and "FAILED" not in status_text
        
        with st.expander(f"Model ID: {job[:8]} | Status: {status_text}", expanded=is_active):
            
            # 1. Hide the "Finished" text if the engine is still running
            if is_active or end in [None, "N/A"]:
                st.caption(f"**ID:** `{job}` | **Started:** {start}")
            else:
                st.caption(f"**ID:** `{job}` | **Started:** {start} | **Finished:** {end}")
            
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
        
        st.markdown("**Engine Settings:**")
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
    
    input_method = st.radio("Select your model source:", ["Use a recent model", "Enter an old Job ID"], horizontal=True)
    
    predict_job_id = None
    
    if input_method == "Use a recent model":
        job_options = list(reversed(st.session_state.jobs))
        if job_options:
            predict_job_id = st.selectbox(
                "Select a trained Model:", 
                job_options, 
                format_func=lambda x: f"Model ID: {x[:8]}"
            )
        else:
            st.info("No recent models found in this session. Train one first, or enter an old ID.")
    else:
        predict_job_id = st.text_input("Paste your full Job ID here:")
    
    predict_file = st.file_uploader("Upload New Data for Prediction (CSV)", type=["csv"], accept_multiple_files=False, key="predict_uploader")
    
    if predict_file is not None:
        df_predict = pd.read_csv(predict_file)
        
        # 1. Toggleable Test Data Preview
        with st.expander("Show Test Data Preview"):
            st.dataframe(df_predict.head())
        
        pred_state_key = f"prediction_{predict_job_id}"
        
        if st.button("Generate Predictions", type="primary"):
            if not predict_job_id:
                st.warning("Please select or enter a Job ID first.")
            else:
                with st.spinner("Waking up model and predicting..."):
                    predict_file.seek(0)
                    response = requests.post(
                        f"{API_URL}/predict",
                        data={"job_id": predict_job_id},
                        files={"file": (predict_file.name, predict_file, "text/csv")}
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
                file_name=f"predictions_{predict_job_id}.csv",
                mime="text/csv"
            )