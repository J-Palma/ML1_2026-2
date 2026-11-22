"""Serializable classifiers used by the reproducible experiment."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class SVDSubspaceClassifier(ClassifierMixin, BaseEstimator):
    """Assign the nearest class-specific affine subspace by residual norm."""
    def __init__(self, k_components=30, temperature=0.25):
        self.k_components = k_components
        self.temperature = temperature

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.means_, self.bases_ = {}, {}
        for cls in self.classes_:
            values = X[np.asarray(y) == cls]
            self.means_[cls] = values.mean(axis=0)
            basis, _, _ = np.linalg.svd((values - self.means_[cls]).T, full_matrices=False)
            self.bases_[cls] = basis[:, :self.k_components]
        return self

    def residuals(self, X):
        output = []
        for cls in self.classes_:
            delta = np.asarray(X) - self.means_[cls]
            basis = self.bases_[cls][:, :self.k_components]
            output.append(np.linalg.norm(delta - (delta @ basis) @ basis.T, axis=1))
        return np.column_stack(output)

    def predict(self, X):
        return self.classes_[self.residuals(X).argmin(axis=1)]

    def predict_proba(self, X):
        distances = self.residuals(X)
        # Normalize each row independently, so a sample's output does not depend
        # on other samples present in the request. These scores are uncalibrated.
        scores = -distances / (distances.mean(axis=1, keepdims=True) + 1e-9) / self.temperature
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)
