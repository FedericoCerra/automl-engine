import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
import numpy as np

from src.preprocessing import AutoPreProcessor
from src.feature_eng import AutoFeatureEngine

class AutoModelSelector:
    def __init__(self, n_trials=20, scoring='neg_mean_squared_error'):
        """
        Args:
            n_trials (int): How many "contestants" to try.
            scoring (str): The metric to optimize (e.g., 'r2', 'neg_mean_absolute_error').
        """
        self.n_trials = n_trials
        self.scoring = scoring
        self.best_pipeline = None
        self.study = None

    def _create_objective(self, X, y):
       
        def objective(trial):
            
            num_strategy = trial.suggest_categorical('num_strategy', ['mean', 'median', 'most_frequent'])
            use_scaler = trial.suggest_categorical('use_scaler', [True, False])
            
            preprocessor = AutoPreProcessor(
                num_strategy=num_strategy, 
                cat_strategy='constant', 
                use_scaler=use_scaler
                )
            
            use_poly=trial.suggest_categorical('use_poly', [True, False])
            degree=trial.suggest_int('degree', 2, 5) if use_poly else 2
            use_log=trial.suggest_categorical('use_log', [True, False])
            use_pca=trial.suggest_categorical('use_pca', [True, False])
            pca_components=trial.suggest_float('pca_components', 0.75, 0.99) if use_pca else 0.95
            
            
            feature_eng = AutoFeatureEngine(
                use_poly=use_poly, 
                degree=degree,
                use_log=use_log,
                use_pca=use_pca,
                pca_components=pca_components
            
            )

            
            model_type = trial.suggest_categorical('model_type', ['rf', 'xgb'])
            
            if model_type == "rf":
                    model = RandomForestRegressor(
                        n_estimators=trial.suggest_int("rf_n_estimators", 50, 300),
                        max_depth=trial.suggest_int("rf_max_depth", 3, 20),
                        random_state=42,
                        n_jobs=-1 
                    )
            elif model_type == "xgb": 
                    model = xgb.XGBRegressor(
                        n_estimators=trial.suggest_int("xgb_n_estimators", 50, 500),
                        max_depth=trial.suggest_int("xgb_max_depth", 3, 10),
                        learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    
            final_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('feature_eng', feature_eng),
                    ('model', model)
                ])
            
            scores = cross_val_score(final_pipeline, X, y, scoring=self.scoring, cv=5, n_jobs=-1)
            
            return np.mean(scores)
    
        return objective
    
    
    def fit(self, X, y):
        print(f"Starting AutoML Search for {self.n_trials} trials...")
        maximize_metrics = ['r2', 'accuracy', 'precision', 'recall']

        # If the metric starts with 'neg_' OR it's in our maximize list, climb the mountain!
        if "neg" in self.scoring or self.scoring in maximize_metrics:
            direction = "maximize"
        else:
            direction = "minimize" # Otherwise, descend into the valley (e.g., pure errors)

        self.study = optuna.create_study(direction=direction)
        
        
        self.study.optimize(self._create_objective(X, y), n_trials=self.n_trials)

        print("\n" + "="*30)
        print("BEST TRIAL FOUND:")
        print(self.study.best_params)
        print("="*30 + "\n")

        self._build_best_pipeline(X, y)

    def _build_best_pipeline(self, X, y):
        p = self.study.best_params
        
        preprocessor = AutoPreProcessor(
            num_strategy=p.get('num_strategy', 'median'),
            use_scaler=p.get('use_scaler', True)
        ) 
        
        feature_eng = AutoFeatureEngine(
            use_poly=p['use_poly'], 
            degree=p.get('poly_degree', 2), # Use .get() in case it wasn't tuned (because use_poly was False)
            use_log=p['use_log'], 
            use_pca=p['use_pca'],
            pca_components=p.get('pca_components', 0.95)
        )

        if p['model_type'] == "rf":
            model = RandomForestRegressor(
                n_estimators=p['rf_n_estimators'],
                max_depth=p['rf_max_depth'],
                random_state=42
            )
        elif p['model_type'] == "xgb":
            model = xgb.XGBRegressor(
                n_estimators=p['xgb_n_estimators'],
                max_depth=p['xgb_max_depth'],
                learning_rate=p['xgb_lr'],
                objective="reg:squarederror",
                random_state=42
            )

        self.best_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_eng', feature_eng),
            ('model', model)
        ])
        
        self.best_pipeline.fit(X, y)
        print("Final Pipeline Retrained and Ready.")

    def predict(self, X):
        return self.best_pipeline.predict(X)