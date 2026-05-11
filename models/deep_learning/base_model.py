import torch
import torch.nn as nn
import numpy as np


class BaseDeepLearningModel:
    """Base class for all DL models."""

    def __init__(self, input_dim, model_name='base_dl'):
        self.input_dim = input_dim
        self.model_name = model_name
        self.model = None
        self.is_trained = False

    def build_model(self):
        """Override in subclass."""
        raise NotImplementedError

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train model."""
        raise NotImplementedError

    def predict(self, X):
        """Predict labels."""
        raise NotImplementedError

    def predict_proba(self, X):
        """Predict probabilities."""
        raise NotImplementedError

    def evaluate(self, X, y):
        """Evaluate model."""
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
