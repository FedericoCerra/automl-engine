import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import PolynomialFeatures

class AutoPreProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, num_strategy='median', cat_strategy='constant', use_scaler=True, use_poly=False, poly_degree=2):        
        """
        Args:
            num_strategy = the strategy to impute missing numerical values.
            cat_strategy = the strategy to impute missing categorical values.
            use_scaler = if True, scales data(default True)
            use_poly = if True, creates polynomial features for numerical data only.
            poly_degree = the degree of the polynomial features.
        """
        # (Optimizable by Optuna)
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.use_scaler = use_scaler
        self.use_poly = use_poly
        self.poly_degree = poly_degree
        
        # State Attributes
        self.preprocessor_ = None 
        self.num_cols = []
        self.cat_cols = [] 
        
    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(include=['number']).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        
        num_pipeline_steps = [('imputer', SimpleImputer(strategy=self.num_strategy))]
        
        if self.use_poly:
            num_pipeline_steps.append(('poly', PolynomialFeatures(degree=self.poly_degree, include_bias=False)))
            
        if self.use_scaler:
            num_pipeline_steps.append(('scaler', StandardScaler()))
            
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