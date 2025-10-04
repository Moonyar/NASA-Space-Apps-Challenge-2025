from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel
from typing import Dict, Any

class RandomForestModel(BaseModel):
    """Random Forest Classifier implementation"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        
    def build_model(self):
        """Build and return RandomForestClassifier with configured parameters"""
        
        # Filter out None values and prepare parameters
        model_params = {}
        
        if self.params.get('n_estimators') is not None:
            model_params['n_estimators'] = int(self.params['n_estimators'])
            
        if self.params.get('max_depth') is not None:
            model_params['max_depth'] = int(self.params['max_depth'])
            
        if self.params.get('min_samples_split') is not None:
            model_params['min_samples_split'] = int(self.params['min_samples_split'])
            
        if self.params.get('min_samples_leaf') is not None:
            model_params['min_samples_leaf'] = int(self.params['min_samples_leaf'])
            
        if self.params.get('max_features') is not None:
            max_feat = self.params['max_features']
            # Handle string or float
            if isinstance(max_feat, str) and max_feat not in ['sqrt', 'log2']:
                try:
                    max_feat = float(max_feat)
                except:
                    max_feat = 'sqrt'
            model_params['max_features'] = max_feat
            
        if self.params.get('bootstrap') is not None:
            model_params['bootstrap'] = bool(self.params['bootstrap'])
            
        if self.params.get('class_weight') is not None:
            model_params['class_weight'] = self.params['class_weight']
            
        if self.params.get('random_state') is not None:
            model_params['random_state'] = int(self.params['random_state'])
        
        # Always set n_jobs for parallel processing
        model_params['n_jobs'] = -1
        
        return RandomForestClassifier(**model_params)