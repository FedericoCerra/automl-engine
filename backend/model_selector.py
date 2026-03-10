import logging
import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error,
)
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocessing import AutoPreProcessor
from feature_eng import AutoFeatureEngine

logger = logging.getLogger("automl-api")

class AutoModelSelector:
    def __init__(self, n_trials=50, task='auto', scoring='auto'):
        self.n_trials = n_trials
        self.task = task
        self.scoring = scoring

        self.best_pipeline = None
        self.study = None
        self.label_encoder = None
        self.target_name = None
        self.best_score = None
        self.baseline_score = None
        self.all_metrics = None

    def _create_model_instance(self, model_type, params, y=None):
        """Helper to instantiate model objects with specific parameters."""
        random_state = 42
        n_jobs = -1
        
        if self.task == 'regression':
            if model_type == "rf":
                return RandomForestRegressor(
                    n_estimators=params['rf_n_estimators'],
                    max_depth=params['rf_max_depth'],
                    random_state=random_state, n_jobs=n_jobs
                )
            elif model_type == "xgb":
                return xgb.XGBRegressor(
                    n_estimators=params['xgb_n_estimators'],
                    max_depth=params['xgb_max_depth'],
                    learning_rate=params['xgb_lr'],
                    objective="reg:squarederror",
                    random_state=random_state, n_jobs=n_jobs
                )
            elif model_type == "svm":
                return SVR(
                    C=params['svm_C'],
                    kernel="rbf",
                    max_iter=2000
                )

        elif self.task == 'classification':
            if model_type == "rf":
                return RandomForestClassifier(
                    n_estimators=params['rf_n_estimators'],
                    max_depth=params['rf_max_depth'],
                    random_state=random_state, n_jobs=n_jobs
                )
            elif model_type == "xgb":
                # Determine objective based on class count
                xgb_obj = "binary:logistic"
                if y is not None:
                    num_classes = len(np.unique(y))
                    if num_classes > 2:
                        xgb_obj = "multi:softprob"
                
                return xgb.XGBClassifier(
                    n_estimators=params['xgb_n_estimators'],
                    max_depth=params['xgb_max_depth'],
                    learning_rate=params['xgb_lr'],
                    objective=xgb_obj,
                    random_state=random_state, n_jobs=n_jobs
                )
            elif model_type == "svm":
                return SVC(
                    C=params['svm_C'],
                    kernel="rbf",
                    probability=True,
                    max_iter=2000
                )
        raise ValueError(f"Invalid model_type '{model_type}' or task '{self.task}'")

    def _initialize_pipeline(self, params, y=None):
        """
        Constructs the pipeline (Preprocessing + Feature Eng + Model) based on parameters.
        Centralizes logic to avoid duplication between optimization and final training.
        """
        # 1. Preprocessor Configuration
        use_scaler = params.get('use_scaler', True)
        use_pca = params.get('use_pca', False)
        model_type = params.get('model_type')
        
        # Enforce scaling for SVMs or PCA to ensure convergence/performance
        if (use_pca or model_type == 'svm') and not use_scaler:
            use_scaler = True

        preprocessor = AutoPreProcessor(
            num_strategy=params.get('num_strategy', 'median'),
            cat_strategy='constant',
            use_scaler=use_scaler,
            use_poly=params.get('use_poly', False),
            poly_degree=params.get('poly_degree', 2)
        )

        # 2. Feature Engineering Configuration
        feature_eng = AutoFeatureEngine(
            use_log=params.get('use_log', False),
            use_pca=use_pca,
            pca_components=params.get('pca_components', 0.95)
        )

        # 3. Model Configuration
        model = self._create_model_instance(model_type, params, y)
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('feature_eng', feature_eng),
            ('model', model)
        ])

    def _get_cv(self, y):
        """Return a cross-validation splitter appropriate for the dataset size and task."""
        n_samples = len(y)
        n_folds = 5 if n_samples >= 50 else 3
        if self.task == "classification":
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        return KFold(n_splits=n_folds, shuffle=True, random_state=42)

    def _create_objective(self, X, y):
        cv = self._get_cv(y)

        def objective(trial):
            use_poly = trial.suggest_categorical("use_poly", [True, False])
            poly_degree = trial.suggest_int("poly_degree", 1, 2) if use_poly else 2

            # Limit polynomial features to prevent explosion
            if use_poly and poly_degree == 2 and X.select_dtypes(include=["number"]).shape[1] > 50:
                use_poly = False

            use_pca = trial.suggest_categorical("use_pca", [True, False])
            pca_comps = trial.suggest_float("pca_components", 0.80, 0.95) if use_pca else 0.95

            model_type = trial.suggest_categorical("model_type", ["rf", "xgb", "svm"])

            params = {
                'num_strategy': trial.suggest_categorical("num_strategy", ["mean", "median"]),
                'use_scaler': trial.suggest_categorical("use_scaler", [True, False]),
                'use_log': trial.suggest_categorical("use_log", [True, False]),
                'use_poly': use_poly,
                'poly_degree': poly_degree,
                'use_pca': use_pca,
                'pca_components': pca_comps,
                'model_type': model_type,
            }

            if model_type == "rf":
                params['rf_n_estimators'] = trial.suggest_int("rf_n_estimators", 50, 300)
                params['rf_max_depth'] = trial.suggest_int("rf_max_depth", 3, 20)
            elif model_type == "xgb":
                params['xgb_n_estimators'] = trial.suggest_int("xgb_n_estimators", 50, 500)
                params['xgb_max_depth'] = trial.suggest_int("xgb_max_depth", 3, 10)
                params['xgb_lr'] = trial.suggest_float("xgb_lr", 0.01, 0.3, log=True)
            elif model_type == "svm":
                params['svm_C'] = trial.suggest_float("svm_C", 0.1, 10.0, log=True)

            try:
                pipeline = self._initialize_pipeline(params, y)
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring=self.scoring)
                return scores.mean()
            except Exception as e:
                raise optuna.TrialPruned(f"Trial failed: {str(e)}")

        return objective

    def _compute_baseline(self, X, y):
        """Train a dummy model to establish a baseline score."""
        cv = self._get_cv(y)
        try:
            if self.task == "classification":
                dummy = DummyClassifier(strategy="most_frequent")
            else:
                dummy = DummyRegressor(strategy="mean")
            scores = cross_val_score(dummy, X, y, cv=cv, scoring=self.scoring)
            return round(float(abs(scores.mean())), 4)
        except Exception:
            return None

    def _compute_all_metrics(self, X, y):
        """Compute comprehensive metrics using cross-val predictions."""
        try:
            preds = self.best_pipeline.predict(X)
            if self.task == "classification":
                avg = "binary" if len(np.unique(y)) == 2 else "macro"
                return {
                    "accuracy": round(float(accuracy_score(y, preds)), 4),
                    "f1": round(float(f1_score(y, preds, average=avg, zero_division=0)), 4),
                    "precision": round(float(precision_score(y, preds, average=avg, zero_division=0)), 4),
                    "recall": round(float(recall_score(y, preds, average=avg, zero_division=0)), 4),
                }
            else:
                return {
                    "r2": round(float(r2_score(y, preds)), 4),
                    "mae": round(float(mean_absolute_error(y, preds)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 4),
                }
        except Exception as e:
            logger.warning(f"Could not compute all metrics: {e}")
            return None

    def fit(self, X, y, progress_callback=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        self.target_name = getattr(y, 'name', 'target')

        # Auto-select task
        if self.task == 'auto':
            y_series = pd.Series(y)
            is_text = y_series.dtype == 'object' or y_series.dtype.name == 'category'
            unique_values = y_series.nunique()

            if is_text or unique_values < 20:
                self.task = 'classification'
            else:
                self.task = 'regression'

            logger.info(f"Auto-configured for {self.task.upper()}")

        # Encode targets for classification
        if self.task == 'classification':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        # Auto-select scoring metric
        if self.scoring == 'auto':
            self.scoring = 'r2' if self.task == 'regression' else 'accuracy'

        logger.info(f"Starting AutoML ({self.task.upper()}) | Scoring: {self.scoring} | Trials: {self.n_trials}")

        # Baseline model
        self.baseline_score = self._compute_baseline(X, y)
        if self.baseline_score is not None:
            logger.info(f"Baseline score ({self.scoring}): {self.baseline_score}")

        maximize_metrics = ['accuracy', 'f1', 'f1_macro', 'roc_auc', 'r2', 'precision', 'recall']
        direction = "maximize" if "neg" in self.scoring or self.scoring in maximize_metrics else "minimize"

        def optuna_callback(study, trial):
            if progress_callback:
                progress_callback(trial.number + 1, self.n_trials)

        # Seed Optuna for reproducible results
        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(direction=direction, sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study.optimize(
            self._create_objective(X, y),
            n_trials=self.n_trials,
            callbacks=[optuna_callback],
        )

        completed = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) == 0:
            raise ValueError("All training trials failed. The selected Task Type or Optimization Goal might be incompatible with your dataset.")

        self.best_score = self.study.best_value

        logger.info(f"Best trial: {self.study.best_params}")

        # Finalize model on full dataset
        logger.info("Retraining best model on full dataset...")
        self.best_pipeline = self._initialize_pipeline(self.study.best_params, y)
        self.best_pipeline.fit(X, y)

        # Compute comprehensive metrics on training data
        self.all_metrics = self._compute_all_metrics(X, y)

        logger.info("Final Pipeline Retrained and Ready.")

    def predict(self, X):
        predictions = self.best_pipeline.predict(X)
        
        # TRANSLATING PREDICTIONS BACK TO TEXT 
        if self.task == 'classification' and self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
            
        return predictions

    def explain(self, X, y, output_path):
        """
        Generates a SHAP summary plot for the best model.
        Uses KernelExplainer on the full pipeline to ensure feature importance
        maps back to the original input features.
        """
        if self.best_pipeline is None:
            logger.warning("Model not fitted yet — cannot compute SHAP.")
            return

        limit = 50
        X_bg = X.sample(limit, random_state=42) if X.shape[0] > limit else X

        logger.info("Calculating SHAP values (KernelExplainer)...")

        def predict_wrapper(data):
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=X.columns)
                for col in X.columns:
                    if pd.api.types.is_numeric_dtype(X[col]):
                        data[col] = pd.to_numeric(data[col], errors='coerce')
            return self.best_pipeline.predict(data)

        explainer = shap.KernelExplainer(predict_wrapper, X_bg)
        shap_values = explainer.shap_values(X_bg, nsamples=100)

        plt.figure(figsize=(10, 6))

        # Handle binary/multiclass output
        vals = shap_values
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                vals = shap_values[1]  # positive class
            elif len(shap_values) > 2:
                # Multiclass: average absolute SHAP values across classes
                vals = np.mean(np.abs(np.array(shap_values)), axis=0)
            else:
                vals = shap_values[0]

        shap.summary_plot(vals, X_bg, feature_names=X.columns, show=False)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP plot saved to {output_path}")