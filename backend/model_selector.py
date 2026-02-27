import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt

from preprocessing import AutoPreProcessor
from feature_eng import AutoFeatureEngine

class AutoModelSelector:
    def __init__(self, n_trials=50, task='auto', scoring='auto'):
        self.n_trials = n_trials
        self.task = task
        self.scoring = scoring
        
        self.best_pipeline = None
        self.study = None
        self.label_encoder = None # For text targets in classification
        self.target_name = None

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

    def _create_objective(self, X, y):
        def objective(trial):
            # Define Search Space
            use_poly = trial.suggest_categorical("use_poly", [True, False])
            poly_degree = trial.suggest_int("poly_degree", 1, 2) if use_poly else 2
            
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
                'model_type': model_type
            }

            # Model Specific Hyperparameters
            if model_type == "rf":
                params['rf_n_estimators'] = trial.suggest_int("rf_n_estimators", 50, 300)
                params['rf_max_depth'] = trial.suggest_int("rf_max_depth", 3, 20)
            elif model_type == "xgb":
                params['xgb_n_estimators'] = trial.suggest_int("xgb_n_estimators", 50, 500)
                params['xgb_max_depth'] = trial.suggest_int("xgb_max_depth", 3, 10)
                params['xgb_lr'] = trial.suggest_float("xgb_lr", 0.01, 0.3, log=True)
            elif model_type == "svm":
                params['svm_C'] = trial.suggest_float("svm_C", 0.1, 10.0, log=True)

            # Build and Evaluate
            pipeline = self._initialize_pipeline(params, y)
            scores = cross_val_score(pipeline, X, y, cv=3, scoring=self.scoring)
            return scores.mean()

        return objective

    def fit(self, X, y, progress_callback=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        self.target_name = getattr(y, 'name', 'target')
        
        # AUTO-SELECT TASK
        if self.task == 'auto':
            y_series = pd.Series(y)
            is_text = y_series.dtype == 'object' or y_series.dtype.name == 'category'
            unique_values = y_series.nunique()

            if is_text or unique_values < 20:
                self.task = 'classification'
            else:
                self.task = 'regression'
                
            print(f"Auto-configured for {self.task.upper()}")

        # ENCODE TARGETS FOR CLASSIFICATION 
        if self.task == 'classification':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            
        # AUTO-SELECT SCORING METRIC 
        if self.scoring == 'auto':
            if self.task == 'regression':
                self.scoring = 'r2'
            else:
                self.scoring = 'accuracy'
                
        print(f"Starting AutoML ({self.task.upper()}) | Scoring: {self.scoring} | Trials: {self.n_trials}")
        
        maximize_metrics = ['accuracy', 'f1', 'f1_macro', 'roc_auc', 'r2', 'precision', 'recall']
        direction = "maximize" if "neg" in self.scoring or self.scoring in maximize_metrics else "minimize"
        
        def optuna_callback(study, trial):
            if progress_callback:
                # trial.number starts at 0, so we add 1
                progress_callback(trial.number + 1, self.n_trials)
                
        self.study = optuna.create_study(direction=direction)
        self.study.optimize(
            self._create_objective(X, y), 
            n_trials=self.n_trials,
            callbacks=[optuna_callback] 
            )
        self.best_score = self.study.best_value 
        
        print("\n" + "="*30)
        print("BEST TRIAL FOUND:")
        print(self.study.best_params)
        print("="*30 + "\n")

        # Finalize Model
        print("Retraining best model on full dataset...")
        self.best_pipeline = self._initialize_pipeline(self.study.best_params, y)
        
        self.best_pipeline.fit(X, y)
        print("Final Pipeline Retrained and Ready.")

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
            print("Model not fitted yet.")
            return

        # Optimization: Subsample background data to prevent timeouts
        # KernelExplainer is computationally expensive.
        limit = 50
        X_bg = X.sample(limit, random_state=42) if X.shape[0] > limit else X

        print("Calculating SHAP values (KernelExplainer)...")
        
        def predict_wrapper(data):
            # SHAP passes numpy arrays; pipeline expects DataFrame with correct dtypes
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=X.columns)
                for col in X.columns:
                    # Attempt to restore numeric types if they were coerced to object
                    if pd.api.types.is_numeric_dtype(X[col]):
                        data[col] = pd.to_numeric(data[col], errors='coerce')
            
            return self.best_pipeline.predict(data)

        # nsamples=100 provides a good balance between speed and approximation accuracy
        explainer = shap.KernelExplainer(predict_wrapper, X_bg)
        shap_values = explainer.shap_values(X_bg, nsamples=100)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        # Handle binary/multiclass output (shap_values might be a list)
        vals = shap_values
        if isinstance(shap_values, list):
            # For binary classification, shap_values is a list of [class0, class1]
            # We usually plot the positive class (index 1)
            vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        shap.summary_plot(vals, X_bg, feature_names=X.columns, show=False)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()