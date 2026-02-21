import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

from src.preprocessing import AutoPreProcessor
from src.feature_eng import AutoFeatureEngine

class AutoModelSelector:
    def __init__(self, n_trials=50, task='auto', scoring='auto'):
        self.n_trials = n_trials
        self.task = task
        self.scoring = scoring
        
        self.best_pipeline = None
        self.study = None
        self.label_encoder = None # For text targets in classification

    def _create_objective(self, X, y):
        
        def objective(trial):
            # PREPROCESSOR TUNING
            num_strategy = trial.suggest_categorical("num_strategy", ["mean", "median"])
            use_scaler = trial.suggest_categorical("use_scaler", [True, False])
            
            # FEATURE ENGINEERING TUNING 
            use_log = trial.suggest_categorical("use_log", [True, False])
            use_poly = trial.suggest_categorical("use_poly", [True, False])
            degree = trial.suggest_int("poly_degree", 1, 2) if use_poly else 2
            
            use_pca = trial.suggest_categorical("use_pca", [True, False])
            pca_comps = trial.suggest_float("pca_components", 0.80, 0.95) if use_pca else 0.95

            # MODEL TUNING
            model_type = trial.suggest_categorical("model_type", ["rf", "xgb", "svm"])
            
            if (use_pca or model_type == 'svm') and not use_scaler:
                # We can't change the trial params once suggested, 
                # but we can force the preprocessor to use a scaler anyway!
                actual_use_scaler = True 
            else:
                actual_use_scaler = use_scaler
                
            preprocessor = AutoPreProcessor(
                num_strategy=num_strategy, 
                cat_strategy='constant',
                use_scaler=actual_use_scaler,
                use_poly=use_poly,     
                poly_degree=degree
            )
            
            
            feature_eng = AutoFeatureEngine(
                use_log=use_log, 
                use_pca=use_pca, 
                pca_components=pca_comps
            )


            
            if self.task == 'regression':
                if model_type == "rf":
                    model = RandomForestRegressor(
                        n_estimators=trial.suggest_int("rf_n_estimators", 50, 300),
                        max_depth=trial.suggest_int("rf_max_depth", 3, 20),
                        random_state=42, n_jobs=-1 
                    )
                elif model_type == "xgb":
                    model = xgb.XGBRegressor(
                        n_estimators=trial.suggest_int("xgb_n_estimators", 50, 500),
                        max_depth=trial.suggest_int("xgb_max_depth", 3, 10),
                        learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
                        objective="reg:squarederror", random_state=42, n_jobs=-1
                    )
                elif model_type == "svm":
                    model = SVR(
                        C=trial.suggest_float("svm_C", 0.1, 10.0, log=True),
                        kernel="rbf",
                        max_iter=2000
                    )

            elif self.task == 'classification':
                if model_type == "rf":
                    model = RandomForestClassifier(
                        n_estimators=trial.suggest_int("rf_n_estimators", 50, 300),
                        max_depth=trial.suggest_int("rf_max_depth", 3, 20),
                        random_state=42, n_jobs=-1 
                    )
                elif model_type == "xgb":
                    # XGBoost needs to know if it's 2 classes or 3+ classes
                    num_classes = len(np.unique(y))
                    xgb_obj = "binary:logistic" if num_classes == 2 else "multi:softprob"
                    
                    model = xgb.XGBClassifier(
                        n_estimators=trial.suggest_int("xgb_n_estimators", 50, 500),
                        max_depth=trial.suggest_int("xgb_max_depth", 3, 10),
                        learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
                        objective=xgb_obj, random_state=42, n_jobs=-1
                    )
                elif model_type == "svm":
                    model = SVC(
                        C=trial.suggest_float("svm_C", 0.1, 10.0, log=True),
                        kernel="rbf", probability=True, max_iter=2000
                    )
            # EVALUATION
            final_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('feature_eng', feature_eng),
                ('model', model)
            ])

            scores = cross_val_score(final_pipeline, X, y, cv=3, scoring=self.scoring)
            return scores.mean()

        return objective

    def fit(self, X, y, progress_callback=None):

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
                self.scoring = 'neg_mean_squared_error'
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

        self._build_best_pipeline(X, y)

    def _build_best_pipeline(self, X, y):
        p = self.study.best_params
        
        use_scaler = p.get('use_scaler', True)
        model_type = p['model_type']
        use_pca = p.get('use_pca', False)
        
        ## This may be a superfluous check
        if (use_pca or model_type == 'svm') and not use_scaler:
            actual_use_scaler = True
        else:
            actual_use_scaler = use_scaler
        
        preprocessor = AutoPreProcessor(
            num_strategy=p.get('num_strategy', 'median'),
            use_scaler=actual_use_scaler,           
            use_poly=p.get('use_poly', False),     
            poly_degree=p.get('poly_degree', 2)    
        ) 
        
        feature_eng = AutoFeatureEngine(
            use_log=p.get('use_log', False),        
            use_pca=use_pca,
            pca_components=p.get('pca_components', 0.95)
        )

        if self.task == 'regression':
            if model_type == "rf":
                model = RandomForestRegressor(n_estimators=p['rf_n_estimators'], max_depth=p['rf_max_depth'], random_state=42)
            elif model_type == "xgb":
                model = xgb.XGBRegressor(n_estimators=p['xgb_n_estimators'], max_depth=p['xgb_max_depth'], learning_rate=p['xgb_lr'], objective="reg:squarederror", random_state=42)
            elif model_type == "svm":
                model = SVR(C=p['svm_C'], kernel="rbf", max_iter=2000)
                
        elif self.task == 'classification':
            if model_type == "rf":
                model = RandomForestClassifier(n_estimators=p['rf_n_estimators'], max_depth=p['rf_max_depth'], random_state=42)
            elif model_type == "xgb":
                num_classes = len(np.unique(y))
                xgb_obj = "binary:logistic" if num_classes == 2 else "multi:softprob"
                model = xgb.XGBClassifier(n_estimators=p['xgb_n_estimators'], max_depth=p['xgb_max_depth'], learning_rate=p['xgb_lr'], objective=xgb_obj, random_state=42)
            elif model_type == "svm":
                model = SVC(C=p['svm_C'], kernel="rbf", probability=True, max_iter=2000) 

        self.best_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_eng', feature_eng),
            ('model', model)
        ])
        
        self.best_pipeline.fit(X, y)
        print("Final Pipeline Retrained and Ready.")

    def predict(self, X):
        predictions = self.best_pipeline.predict(X)
        
        # TRANSLATING PREDICTIONS BACK TO TEXT 
        if self.task == 'classification' and self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
            
        return predictions