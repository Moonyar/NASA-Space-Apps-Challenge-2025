import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from typing import Dict, Any, Tuple, List

class BaseModel(ABC):
    """Base class for all ML models"""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize model with hyperparameters
        
        Args:
            params: Dictionary of hyperparameters
        """
        self.params = params
        self.model = None
        self.feature_cols = None
        self.target_col = None
        self.training_history = {}
        
    @abstractmethod
    def build_model(self):
        """Build and return the sklearn model instance"""
        pass
    
    def preprocess_data(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data before training
        
        Args:
            df: Input dataframe
            feature_cols: List of feature column names
            target_col: Target column name
            
        Returns:
            Tuple of (features, target)
        """
        # Drop rows with missing values in selected columns
        df_clean = df[feature_cols + [target_col]].dropna()
        
        # Separate features and target
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Handle categorical target if needed
        if y.dtype == 'object':
            y = pd.Categorical(y).codes
            
        return X, y
    
    def train(self, df: pd.DataFrame, feature_cols: List[str], target_col: str, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train the model and return results
        
        Args:
            df: Input dataframe
            feature_cols: List of feature column names
            target_col: Target column name
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing training results and metrics
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Preprocess data
        X, y = self.preprocess_data(df, feature_cols, target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Build model
        self.model = self.build_model()
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        results = {
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_cols": feature_cols,
            "target_col": target_col,
            "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
            "classification_report": classification_report(y_test, y_test_pred, output_dict=True),
        }
        
        # Add model-specific metrics
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            results["feature_importances"] = {
                col: float(imp) for col, imp in zip(feature_cols, importances)
            }
        
        self.training_history = results
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters"""
        return self.params