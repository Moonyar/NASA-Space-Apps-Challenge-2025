import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class MLModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def feature_engineering(self, df):
        """
        Apply feature engineering steps to the dataframe
        Customize this based on your specific needs
        """
        df_processed = df.copy()
        
        # Handle missing values
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(
            include=['object']
        ).columns
        
        # Fill numeric columns with median
        for col in numeric_cols:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        for col in categorical_cols:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(
                    df_processed[col]
                )
            else:
                df_processed[col] = self.label_encoders[col].transform(
                    df_processed[col]
                )
        
        return df_processed
    
    def train(self, X, y):
        """Train the model"""
        X_processed = self.feature_engineering(X)
        X_scaled = self.scaler.fit_transform(X_processed)
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        """Make predictions on new data"""
        X_processed = self.feature_engineering(X)
        X_scaled = self.scaler.transform(X_processed)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X, y_true):
        """Evaluate model and return metrics"""
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        return metrics, y_pred
    
    def save(self, filepath):
        """Save the model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load the model from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)