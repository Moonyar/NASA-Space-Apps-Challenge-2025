from typing import Dict, Any
from .base import BaseModel
from .random_forest import RandomForestModel

# Model registry
MODEL_REGISTRY = {
    "RandomForest": RandomForestModel,
    # Add more models here as you implement them
    # "GradientBoosting": GradientBoostingModel,
    # "AdaBoost": AdaBoostModel,
    # "NeuralNet": NeuralNetModel,
    # "LogisticRegression": LogisticRegressionModel,
}

def get_model(model_name: str, params: Dict[str, Any]) -> BaseModel:
    """
    Factory function to get a model instance
    
    Args:
        model_name: Name of the model (e.g., "RandomForest")
        params: Dictionary of hyperparameters
        
    Returns:
        Instance of the requested model
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not implemented. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(params)