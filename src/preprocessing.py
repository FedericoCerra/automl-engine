import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted

class AutoPreProcessor(BaseEstimator , TransformerMixin):
    def __init__(self, num_strategy='median', cat_strategy='constant', use_scaler=True):
        """
        Args:
            num_stra
        """
        # 1. (Optimizable by Optuna)
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.use_scaler = use_scaler
        
        # State Attributes
        self.preprocessor_ = None 
        self.num_cols = []
        self.cat_cols = [] 
        
    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(include=['number']).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        
        num_pipeline_steps = [('imputer',SimpleImputer(strategy=self.num_strategy))]
        if self.use_scaler:
            num_pipeline_steps.append(('scaler',StandardScaler()))
        num_pipeline = Pipeline(num_pipeline_steps)
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=self.cat_strategy, fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor_ = ColumnTransformer([
            ('num', num_pipeline, self.num_cols),
            ('cat', cat_pipeline, self.cat_cols)
        ], verbose_feature_names_out=False)
        
        self.preprocessor_.fit(X)
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'preprocessor_')
        return self.preprocessor_.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'preprocessor_')
        return self.preprocessor_.get_feature_names_out()