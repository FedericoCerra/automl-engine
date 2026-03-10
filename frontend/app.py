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
REQUEST_TIMEOUT = 30  # seconds
JOBS_PER_PAGE = 10

CLASSIFICATION_METRICS = ["auto", "accuracy", "f1", "roc_auc", "precision", "recall"]
REGRESSION_METRICS = ["auto", "r2", "neg_mean_squared_error"]
TASK_OPTIONS = ["auto", "classification", "regression"]
EXAMPLE_DATASETS = ["Titanic (Classification)", "House Prices (Regression)"]

st.set_page_config(page_title="AutoML Master", layout="centered")

# --- SESSION STATE DEFAULTS ---
_defaults = {
    "jobs": [],
    "last_statuses": {},
    "engine_expanded": True,
    "confirm_delete": None,
    "jobs_page": 0,
}
for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- SESSION MANAGEMENT ---
if "uid" in st.query_params:
    user_id = st.query_params["uid"]
else:
    user_id = str(uuid.uuid4())[:8]
    st.query_params["uid"] = user_id

jobs_file = f"jobs_{user_id}.json"

# Load persisted jobs on first run
if not st.session_state.jobs and os.path.exists(jobs_file):
    try:
        with open(jobs_file, "r") as f:
            st.session_state.jobs = json.load(f)
    except (json.JSONDecodeError, IOError):
        st.session_state.jobs = []

# --- HELPERS ---
def api_get(path: str, **kwargs):
    """GET request with timeout and error handling."""
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return None
    except requests.Timeout:
        return None
    except requests.RequestException:
        return None

def api_post(path: str, **kwargs):
    """POST request with timeout. Returns response object."""
    return requests.post(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)

def save_jobs():
    """Persist job list to disk."""
    with open(jobs_file, "w") as f:
        json.dump(st.session_state.jobs, f)

def parse_api_error(response) -> str:
    """Extract a user-friendly error message from an API response."""
    try:
        detail = response.json().get("detail", "")
        if detail:
            return detail
    except Exception:
        pass
    return f"Server returned status {response.status_code}. Please try again."


# --- LAYOUT ---
st.title("AutoML Master Engine")
st.markdown("Drag, drop, and train Machine Learning models instantly.")

tab_train, tab_predict = st.tabs(["Train Model", "Predict"])


# ==========================================
# DASHBOARD FRAGMENT
# ==========================================
@st.fragment(run_every=6)
def render_dashboard():
    st.subheader("Your Training History")

    jobs_data = api_get(f"/jobs/{user_id}")

    if jobs_data is None:
        st.warning("Could not reach the API. The server may be starting up — please wait a moment.")
        return

    # Sync local state with backend
    st.session_state.jobs = list(jobs_data.keys())
    save_jobs()

    if not jobs_data:
        st.info("No models training yet. Upload a CSV above to begin.")
        return

    all_jobs = list(reversed(st.session_state.jobs))

    # Pagination
    total_pages = max(1, (len(all_jobs) + JOBS_PER_PAGE - 1) // JOBS_PER_PAGE)
    page = st.session_state.jobs_page
    page = min(page, total_pages - 1)
    start = page * JOBS_PER_PAGE
    page_jobs = all_jobs[start:start + JOBS_PER_PAGE]

    for job in page_jobs:
        status_res = jobs_data.get(job)
        if not status_res:
            continue

        status_text = status_res.get("status", "Unknown")

        # Completion transition detection
        prev_status = st.session_state.last_statuses.get(job)
        just_finished = (status_text == "COMPLETED" and prev_status != "COMPLETED" and prev_status is not None)
        st.session_state.last_statuses[job] = status_text

        score = status_res.get("score")
        start_time = status_res.get("start_time", "N/A")
        end_time = status_res.get("end_time", "N/A")
        filename = status_res.get("filename", "Unknown File")
        target = status_res.get("target", "Unknown Target")
        shap_enabled = status_res.get("shap_enabled", False)
        best_params = status_res.get("best_params")
        dataset_stats = status_res.get("dataset_stats")
        task_type = status_res.get("task_type")
        baseline_score = status_res.get("baseline_score")
        all_metrics = status_res.get("all_metrics")

        is_active = "COMPLETED" not in status_text and "FAILED" not in status_text

        # New job highlight
        is_new = False
        if "new_job_id" in st.session_state and st.session_state.new_job_id == job:
            if time.time() - st.session_state.get("new_job_time", 0) < 5:
                is_new = True

        # Icon
        if is_new:
            icon = "🟢"
        elif status_text == "COMPLETED":
            icon = "✅"
        elif "FAILED" in status_text:
            icon = "❌"
        else:
            icon = "⏳"

        label = f"{icon} Model: {job} ({filename})"
        should_expand = is_new or is_active or just_finished

        col_exp, col_del = st.columns([0.85, 0.15])

        with col_exp:
            with st.expander(label, expanded=should_expand):

                if "FAILED" in status_text:
                    st.error(f"**Training Failed**\n\n{status_text.replace('FAILED:', '').strip()}", icon="🚨")
                    st.caption(f"**ID:** `{job}` | **Started:** {start_time} | **Failed:** {end_time}")
                else:
                    st.write(f"**Status:** {status_text}")

                    if is_active or end_time in [None, "N/A"]:
                        st.caption(f"**ID:** `{job}` | **Started:** {start_time}")
                    else:
                        st.caption(f"**ID:** `{job}` | **Started:** {start_time} | **Finished:** {end_time}")

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

                        # Show comprehensive metrics
                        if all_metrics:
                            st.divider()
                            st.write("**Metrics (on training data):**")
                            metric_cols = st.columns(len(all_metrics))
                            for i, (k, v) in enumerate(all_metrics.items()):
                                metric_cols[i].metric(k.upper(), f"{v:.4f}")

                        # Show baseline comparison
                        if baseline_score is not None and score is not None:
                            improvement = score - baseline_score
                            st.write(f"**Baseline Score:** {baseline_score:.4f} | **Improvement:** +{improvement:.4f}")

                        if best_params:
                            st.divider()
                            st.write("**Best Hyperparameters:**")
                            st.json(best_params, expanded=False)

                    if is_active:
                        match = re.search(r'\((\d+)%\)', status_text)
                        if match:
                            pct = int(match.group(1))
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

                        if status_text == "COMPLETED":
                            st.link_button(
                                label="Download Model (.pkl)",
                                url=f"{API_URL}/download/{job}"
                            )

                            if shap_enabled:
                                shap_url = f"{API_URL}/shap/{job}"
                                if st.checkbox("Show Feature Importance (SHAP)", key=f"shap_btn_{job}"):
                                    st.image(shap_url, caption="Feature Importance", use_container_width=True)

        with col_del:
            # Two-step delete: click once to confirm, click again to execute
            if st.session_state.confirm_delete == job:
                if st.button("Sure?", key=f"confirm_del_{job}", type="primary"):
                    st.session_state.jobs.remove(job)
                    save_jobs()
                    try:
                        requests.delete(f"{API_URL}/job/{job}", timeout=REQUEST_TIMEOUT)
                    except requests.RequestException:
                        pass
                    st.session_state.confirm_delete = None
                    st.rerun()
            else:
                if st.button("🗑️", key=f"del_{job}", help="Delete Model"):
                    st.session_state.confirm_delete = job
                    st.rerun()

    # Pagination controls
    if total_pages > 1:
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("Previous", disabled=(page == 0), key="pg_prev"):
                st.session_state.jobs_page = max(0, page - 1)
                st.rerun()
        with col_info:
            st.caption(f"Page {page + 1} of {total_pages}")
        with col_next:
            if st.button("Next", disabled=(page >= total_pages - 1), key="pg_next"):
                st.session_state.jobs_page = min(total_pages - 1, page + 1)
                st.rerun()


# ==========================================
# DATA PROFILING
# ==========================================
def render_data_profile(df: pd.DataFrame):
    """Show a quick data profiling section for the uploaded dataset."""
    with st.expander("📊 Data Profile", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Cells", int(df.isnull().sum().sum()))

        col_types = st.columns(2)
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
        col_types[0].write(f"**Numeric columns ({len(num_cols)}):** {', '.join(num_cols[:10])}")
        col_types[1].write(f"**Categorical columns ({len(cat_cols)}):** {', '.join(cat_cols[:10])}")

        # Missing data per column
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(1)
        has_missing = missing_pct[missing_pct > 0]
        if not has_missing.empty:
            st.write("**Columns with missing values:**")
            missing_df = pd.DataFrame({"Column": has_missing.index, "Missing %": has_missing.values})
            st.dataframe(missing_df, hide_index=True, use_container_width=True)
        else:
            st.write("**No missing values found.**")

        st.write("**Sample Data (first 5 rows):**")
        st.dataframe(df.head(), hide_index=True, use_container_width=True)


# ==========================================
# SECTION 1: TRAIN
# ==========================================
@st.cache_data
def load_example_data(name):
    if "Titanic" in name:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        fname = "titanic.csv"
        df = pd.read_csv(url)
        if 'Survived' in df.columns:
            df['Survived'] = df.pop('Survived')
    else:
        url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
        fname = "housing.csv"
        df = pd.read_csv(url)
        if 'median_house_value' in df.columns:
            df['median_house_value'] = df.pop('median_house_value')
    return df, fname

with tab_train:
    st.header("1. Train a New Model")

    data_source = st.radio("Data Source", ["Upload CSV", "Example Datasets"], horizontal=True)

    train_file = None

    if data_source == "Upload CSV":
        train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"], accept_multiple_files=False, key="train_uploader")
    else:
        example_ds = st.selectbox("Select Example Dataset", EXAMPLE_DATASETS)
        try:
            df_example, fname = load_example_data(example_ds)
            train_file = io.BytesIO()
            df_example.to_csv(train_file, index=False)
            train_file.seek(0)
            train_file.name = fname
            st.info(f"Loaded {fname} ({len(df_example)} rows)")
        except Exception as e:
            st.error(f"Error loading example data: {e}")

    if train_file is not None:
        try:
            df_train = pd.read_csv(train_file)
        except Exception as e:
            st.error(f"Could not read CSV file: {e}")
            st.stop()

        if df_train.empty or len(df_train.columns) < 2:
            st.error("CSV must have at least 2 columns and some data rows.")
            st.stop()

        # Data profiling
        render_data_profile(df_train)

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
                        TASK_OPTIONS,
                        help="Force the model to perform a specific task type."
                    )

                is_metric_disabled = False
                if task_type == "classification":
                    metric_options = CLASSIFICATION_METRICS
                elif task_type == "regression":
                    metric_options = REGRESSION_METRICS
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

        # Target validation
        target_valid = target_col in df_train.columns and df_train[target_col].nunique() >= 2

        if st.button("Start Training", type="primary", disabled=not target_valid):
            train_file.seek(0)
            try:
                response = api_post(
                    "/train",
                    data={"target": target_col, "trials": trials, "task": task_type, "scoring": scoring_metric, "shap": shap, "uid": user_id},
                    files={"file": (train_file.name, train_file, "text/csv")}
                )

                if response.status_code == 200:
                    job_id = response.json()["job_id"]
                    st.session_state.jobs.append(job_id)
                    save_jobs()

                    st.session_state.engine_expanded = False
                    st.session_state.new_job_id = job_id
                    st.session_state.new_job_time = time.time()

                    st.toast("Training started! Scroll down to watch progress.", icon="🚀")
                    st.rerun()
                else:
                    st.error(f"Training failed: {parse_api_error(response)}")
            except requests.ConnectionError:
                st.error("Could not connect to the API server. Please check if the backend is running.")
            except requests.Timeout:
                st.error("Request timed out. The server may be overloaded — please try again.")

        if not target_valid and train_file is not None:
            st.caption("Target column must have at least 2 unique values.")

    st.divider()
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

    # Show Training Data Sample if available
    if input_method == "Select from History" and predict_job_id:
        job_details = api_get(f"/status/{predict_job_id}")
        if job_details:
            sample_data = job_details.get("sample_data")
            if sample_data:
                with st.expander("ℹ️ Training Data Example (Expected Format)", expanded=False):
                    st.caption("Ensure your prediction data matches these columns:")
                    st.dataframe(pd.DataFrame([sample_data]), hide_index=True)

    predict_file = st.file_uploader("Upload New Data for Prediction (CSV)", type=["csv"], accept_multiple_files=False, key="predict_uploader")

    if predict_file is not None:
        try:
            df_predict = pd.read_csv(predict_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        with st.expander("Show Test Data Preview"):
            st.dataframe(df_predict.head())

        if uploaded_model:
            pred_state_key = f"prediction_upload_{uploaded_model.name}"
        else:
            pred_state_key = f"prediction_{predict_job_id}"

        can_predict = bool(predict_job_id or uploaded_model)

        if st.button("Generate Predictions", type="primary", disabled=not can_predict):
            if not can_predict:
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

                    try:
                        response = api_post("/predict", data=data, files=files)

                        if response.status_code == 200:
                            st.session_state[pred_state_key] = response.content
                        else:
                            st.error(f"Prediction failed: {parse_api_error(response)}")
                    except requests.ConnectionError:
                        st.error("Could not connect to the API server.")
                    except requests.Timeout:
                        st.error("Request timed out. The server may be overloaded.")

        if pred_state_key in st.session_state:
            csv_bytes = st.session_state[pred_state_key]
            df_results = pd.read_csv(io.BytesIO(csv_bytes))

            with st.expander("Show Predictions Preview", expanded=True):
                st.dataframe(df_results.head())

            st.subheader("Prediction Distribution")
            pred_col = df_results.columns[-1]

            if pd.api.types.is_numeric_dtype(df_results[pred_col]):
                fig_pred = px.histogram(
                    df_results, x=pred_col, title=f"Distribution of {pred_col}",
                    marginal="box", color_discrete_sequence=['#636EFA'],
                    labels={pred_col: pred_col},
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            else:
                counts = df_results[pred_col].value_counts().reset_index()
                counts.columns = [pred_col, "Count"]
                fig_pred = px.bar(
                    counts, x=pred_col, y="Count", title=f"Class Distribution: {pred_col}",
                    color=pred_col, text="Count",
                    labels={pred_col: pred_col},
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            st.download_button(
                label="Download Predictions (CSV)",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv"
            )
