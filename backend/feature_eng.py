import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

class AutoFeatureEngine(BaseEstimator, TransformerMixin):
    def __init__(self, use_log=True, use_pca=False, pca_components=0.95):
        """
        Args:
            use_log (bool): If True, applies log(1+x) to help with skewed data.
            use_pca (bool): If True, compresses data to remove noise/redundancy.
            pca_components (float): If < 1.0, keep variance (e.g., 95%). If > 1, keep N features.
        """
        self.use_log = use_log
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        self.pipeline = None
    
    def fit(self, X, y=None):
        steps = []
        
        if self.use_log:
            if np.nanmin(X) >= 0:
                steps.append(('log', FunctionTransformer(np.log1p, validate=True)))
            else:
                print("Skipping Log Transform: Data contains negative values.")

        if self.use_pca:
            steps.append(('pca', PCA(n_components=self.pca_components)))

        if not steps:
            steps.append(('identity', FunctionTransformer(func=None, validate=False)))

        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, 'pipeline')
        return self.pipeline.transform(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        check_is_fitted(self, 'pipeline')
        return self.pipeline.get_feature_names_out(input_features)