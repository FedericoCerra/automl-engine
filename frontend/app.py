import streamlit as st
import time
import pandas as pd
import requests
import re
import io
import json
import os
import uuid
import plotly.express as px

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

if "last_statuses" not in st.session_state:
    st.session_state.last_statuses = {}

if "engine_expanded" not in st.session_state:
    st.session_state.engine_expanded = True

st.title("AutoML Master Engine")
st.markdown("Drag, drop, and train Machine Learning models instantly.")

tab_train, tab_predict = st.tabs(["Train Model", "Predict"])

# ==========================================
# DASHBOARD FRAGMENT
# ==========================================
@st.fragment(run_every=6)
def render_dashboard():
    st.subheader("Your Training History")
    
    # 1. Fetch all jobs for this user in one single request
    try:
        jobs_data = requests.get(f"{API_URL}/jobs/{user_id}").json()
        # Sync local session state with backend truth
        st.session_state.jobs = list(jobs_data.keys())
    except:
        jobs_data = {}

    if not jobs_data:
        st.info("No models training yet. Upload a CSV above to begin.")
        return
        
    for job in reversed(st.session_state.jobs):
        status_res = jobs_data.get(job)
        if not status_res:
            continue
            
        status_text = status_res.get("status", "Unknown")
        
        # Check for completion transition to trigger visual effects
        prev_status = st.session_state.last_statuses.get(job)
        just_finished = (status_text == "COMPLETED" and prev_status != "COMPLETED" and prev_status is not None)
        st.session_state.last_statuses[job] = status_text

        score = status_res.get("score")
        start = status_res.get("start_time", "N/A")
        end = status_res.get("end_time", "N/A")
        filename = status_res.get("filename", "Unknown File")
        target = status_res.get("target", "Unknown Target")
        shap_enabled = status_res.get("shap_enabled", False)
        best_params = status_res.get("best_params")
        dataset_stats = status_res.get("dataset_stats")
        task_type = status_res.get("task_type")
        
        is_active = "COMPLETED" not in status_text and "FAILED" not in status_text
        
        # Check if this is a newly created job (within last 5 seconds)
        is_new = False
        if "new_job_id" in st.session_state and st.session_state.new_job_id == job:
            if time.time() - st.session_state.get("new_job_time", 0) < 5:
                is_new = True

        # Determine Icon and Label
        if is_new:
            icon = "🟢"
        elif status_text == "COMPLETED":
            icon = "✅"
        elif "FAILED" in status_text:
            icon = "❌"
        else:
            icon = "⏳"
            
        # Static label prevents re-expansion on status update
        label = f"{icon} Model: {job} ({filename})"
        
        # Expand if: New, Running, or Just Finished
        should_expand = is_new or is_active or just_finished

        # Layout: Expander for details | Delete Button
        col_exp, col_del = st.columns([0.9, 0.1])
        
        with col_exp:
            with st.expander(label, expanded=should_expand):
                
                if "FAILED" in status_text:
                    st.error(f"**Training Failed**\n\n{status_text.replace('FAILED:', '').strip()}", icon="🚨")
                else:
                    st.write(f"**Status:** {status_text}")
                
                # 1. Hide the "Finished" text if the engine is still running
                if is_active or end in [None, "N/A"]:
                    st.caption(f"**ID:** `{job}` | **Started:** {start}")
                else:
                    st.caption(f"**ID:** `{job}` | **Started:** {start} | **Finished:** {end}")
                
                with st.expander("Model & Data Details"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Target:** `{target}`")
                        if task_type:
                            st.write(f"**Task:** `{task_type.title()}`")
                    with c2:
                        if dataset_stats:
                            st.write(f"**Rows:** {dataset_stats.get('rows')}")
                            st.write(f"**Cols:** {dataset_stats.get('columns')}")
                    
                    if best_params:
                        st.divider()
                        st.write("**Best Hyperparameters:**")
                        st.json(best_params, expanded=False)
                
                if is_active:
                    match = re.search(r'\((\d+)%\)', status_text)
                    if match:
                        pct = int(match.group(1))
                        # 2. Add the percentage text directly above the progress bar
                        st.write(f"**Progress:** {pct}%")
                        st.progress(pct)
                    elif "Calculating SHAP values" in status_text:
                        st.write("**Status:** Calculating SHAP values...")
                        st.progress(90)
                    else:
                        st.write("**Progress:** Starting...")
                        st.progress(0)
                else:
                    if score:
                        st.write(f"**Final Score:** {score:.4f} ({status_res.get('metric')})")
                    
                    # Feature Importance Visualization
                    feat_imp = status_res.get("feature_importance")
                    if feat_imp:
                        with st.expander("📊 Feature Importance", expanded=False):
                            df_imp = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"])
                            df_imp = df_imp.sort_values(by="Importance", ascending=False).head(10)
                            
                            fig = px.bar(df_imp, x="Importance", y="Feature", orientation='h', 
                                         title="Top 10 Important Features", color="Importance")
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if status_text == "COMPLETED":
                        st.link_button(
                            label="Download Model (.pkl)",
                            url=f"{API_URL}/download/{job}"
                        )
                        
                        # Check for SHAP plot
                        if shap_enabled:
                            shap_url = f"{API_URL}/shap/{job}"
                            # We use a simple check by trying to load it or just providing the button
                            if st.checkbox("Show Feature Importance (SHAP)", key=f"shap_btn_{job}"):
                                st.image(shap_url, caption="Feature Importance", use_container_width=True)
        with col_del:
            if st.button("❌", key=f"del_{job}", help="Delete Model"):
                st.session_state.jobs.remove(job)
                with open(jobs_file, "w") as f:
                    json.dump(st.session_state.jobs, f)
                requests.delete(f"{API_URL}/job/{job}")
                st.rerun()
# ==========================================
# SECTION 1: TRAIN
# ==========================================
with tab_train:
    st.header("1. Train a New Model")
    
    train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"], accept_multiple_files=False, key="train_uploader")
    
    if train_file is not None:
        df_train = pd.read_csv(train_file)
        
        with st.expander("⚙️ Training Configuration", expanded=st.session_state.engine_expanded):
            t1, t2 = st.tabs(["General Settings", "Advanced Options"])
            
            with t1:
                col1, col2 = st.columns(2)
                with col1:
                    last_col_index = len(df_train.columns) - 1
                    target_col = st.selectbox("Target Column to Predict:", df_train.columns, index=last_col_index)
                with col2:
                    trials = st.number_input("Number of Trials:", min_value=1, max_value=100, value=20)
                
                shap = st.checkbox("Enable SHAP Feature Importance (Slower)", value=False)
            
            with t2:
                st.caption("Override automatic detection settings.")
                c_adv1, c_adv2 = st.columns(2)
                with c_adv1:
                    task_type = st.selectbox(
                        "Task Type", 
                        ["auto", "classification", "regression"], 
                        help="Force the model to perform a specific task type."
                    )
                
                # Filter metrics based on selected task
                is_metric_disabled = False
                if task_type == "classification":
                    metric_options = ["auto", "accuracy", "f1", "roc_auc", "precision", "recall"]
                elif task_type == "regression":
                    metric_options = ["auto", "r2", "neg_mean_squared_error"]
                else:
                    metric_options = ["auto"]
                    is_metric_disabled = True

                with c_adv2:
                    scoring_metric = st.selectbox(
                        "Optimization Goal (Metric)", 
                        metric_options,
                        disabled=is_metric_disabled,
                        help="The metric the model optimizes for during training."
                    )
            
        if st.button("Start Training", type="primary"):
            train_file.seek(0)
            response = requests.post(
                f"{API_URL}/train",
                data={"target": target_col, "trials": trials, "task": task_type, "scoring": scoring_metric, "shap": shap, "uid": user_id},
                files={"file": (train_file.name, train_file, "text/csv")}
            )
            
            if response.status_code == 200:
                job_id = response.json()["job_id"]
                st.session_state.jobs.append(job_id)
                # Save job list to user-specific file
                with open(jobs_file, "w") as f:
                    json.dump(st.session_state.jobs, f)
                
                st.session_state.engine_expanded = False
                st.session_state.new_job_id = job_id
                st.session_state.new_job_time = time.time()
                
                st.toast("Training started! Scroll down to watch progress.", icon="🚀")
                st.rerun()
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
            
            # Prediction Distribution Visualization
            st.subheader("Prediction Distribution")
            pred_col = df_results.columns[-1] # Assume last column is the prediction
            
            if pd.api.types.is_numeric_dtype(df_results[pred_col]):
                # Regression: Histogram
                fig_pred = px.histogram(df_results, x=pred_col, title=f"Distribution of {pred_col}",
                                        marginal="box", color_discrete_sequence=['#636EFA'])
                st.plotly_chart(fig_pred, use_container_width=True)
            else:
                # Classification: Bar Chart
                counts = df_results[pred_col].value_counts().reset_index()
                counts.columns = [pred_col, "Count"]
                fig_pred = px.bar(counts, x=pred_col, y="Count", title=f"Class Distribution: {pred_col}",
                                  color=pred_col, text="Count")
                st.plotly_chart(fig_pred, use_container_width=True)
            
            # 4. The Permanent Download Button
            st.download_button(
                label="Download Predictions (CSV)",
                data=csv_bytes,
                file_name=f"predictions.csv",
                mime="text/csv"
            )