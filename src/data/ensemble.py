"""
Ensemble model combining Computer Vision, Time Series, and Graph Neural Network predictions.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from .computer_vision import ActivityDetectionCNN, predict_activity_levels
from .time_series import TimeSeriesLSTM, predict_time_series_risks
from .graph_neural_network import SupplyChainGNN, predict_graph_risks
from torch_geometric.data import Data


class EnsemblePredictor:
    """Ensemble model combining multiple prediction approaches."""
    
    def __init__(
        self,
        cv_weight: float = 0.2,
        ts_weight: float = 0.4,
        gnn_weight: float = 0.4,
        device: str = 'cpu'
    ):
        """
        Initialize ensemble predictor with model weights.
        
        Args:
            cv_weight: Weight for computer vision model
            ts_weight: Weight for time series model  
            gnn_weight: Weight for graph neural network model
            device: Device to run models on
        """
        self.cv_weight = cv_weight
        self.ts_weight = ts_weight
        self.gnn_weight = gnn_weight
        self.device = device
        
        # Model placeholders
        self.cv_model: Optional[ActivityDetectionCNN] = None
        self.ts_model: Optional[TimeSeriesLSTM] = None
        self.gnn_model: Optional[SupplyChainGNN] = None
        self.ts_scaler = None
        self.graph_data: Optional[Data] = None
        
        # Validate weights sum to 1
        total_weight = cv_weight + ts_weight + gnn_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def load_models(
        self,
        cv_model: ActivityDetectionCNN,
        ts_model: TimeSeriesLSTM,
        gnn_model: SupplyChainGNN,
        ts_scaler,
        graph_data: Data
    ):
        """Load trained models into the ensemble."""
        self.cv_model = cv_model.to(self.device)
        self.ts_model = ts_model.to(self.device)
        self.gnn_model = gnn_model.to(self.device)
        self.ts_scaler = ts_scaler
        self.graph_data = graph_data.to(self.device)
    
    def predict_ensemble(
        self,
        company_df: pd.DataFrame,
        time_series_df: pd.DataFrame,
        satellite_images: List[Dict[str, Any]],
        sequence_length: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate ensemble predictions combining all models.
        
        Returns:
            Tuple of (ensemble_predictions, gnn_predictions, ts_predictions, cv_predictions)
        """
        if not all([self.cv_model, self.ts_model, self.gnn_model]):
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Get GNN predictions
        gnn_probs = predict_graph_risks(self.gnn_model, self.graph_data, self.device)
        
        # Get Time Series predictions
        ts_probs_list = predict_time_series_risks(
            self.ts_model, self.ts_scaler, time_series_df, sequence_length, self.device
        )
        ts_probs = torch.FloatTensor(ts_probs_list).to(self.device)
        
        # Get Computer Vision predictions
        cv_probs_list = predict_activity_levels(self.cv_model, satellite_images, self.device)
        
        # Map CV predictions to companies (simplified mapping)
        cv_company_probs = []
        for i in range(len(company_df)):
            # Find images for this company
            company_images = [img for img in satellite_images if img['company_id'] == i]
            if company_images:
                # Average activity levels for company
                avg_activity = np.mean([img['activity_level'] for img in company_images])
                cv_company_probs.append(avg_activity)
            else:
                cv_company_probs.append(0.5)  # Default
        
        cv_probs = torch.FloatTensor(cv_company_probs).to(self.device)
        
        # Ensure all tensors have the same length
        min_len = min(len(gnn_probs), len(ts_probs), len(cv_probs))
        gnn_probs = gnn_probs[:min_len]
        ts_probs = ts_probs[:min_len]
        cv_probs = cv_probs[:min_len]
        
        # Ensemble predictions (weighted average)
        ensemble_probs = (
            self.gnn_weight * gnn_probs + 
            self.ts_weight * ts_probs + 
            self.cv_weight * cv_probs
        )
        
        return ensemble_probs, gnn_probs, ts_probs, cv_probs
    
    def predict_top_risks(
        self,
        company_df: pd.DataFrame,
        time_series_df: pd.DataFrame,
        satellite_images: List[Dict[str, Any]],
        top_k: int = 10
    ) -> pd.DataFrame:
        """Get top K companies at risk."""
        
        ensemble_preds, _, _, _ = self.predict_ensemble(
            company_df, time_series_df, satellite_images
        )
        
        # Create results dataframe
        results_df = company_df.copy()
        results_df['risk_score'] = ensemble_preds.cpu().numpy()
        
        # Return top K highest risk companies
        return results_df.nlargest(top_k, 'risk_score')
    
    def analyze_risk_factors(
        self,
        company_df: pd.DataFrame,
        time_series_df: pd.DataFrame,
        satellite_images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze risk factors across different dimensions."""
        
        ensemble_preds, gnn_preds, ts_preds, cv_preds = self.predict_ensemble(
            company_df, time_series_df, satellite_images
        )
        
        results_df = company_df.copy()
        results_df['ensemble_risk'] = ensemble_preds.cpu().numpy()
        results_df['gnn_risk'] = gnn_preds.cpu().numpy()
        results_df['ts_risk'] = ts_preds.cpu().numpy()
        results_df['cv_risk'] = cv_preds.cpu().numpy()
        
        analysis = {
            'overall_stats': {
                'avg_risk': results_df['ensemble_risk'].mean(),
                'high_risk_count': len(results_df[results_df['ensemble_risk'] > 0.7]),
                'total_companies': len(results_df)
            },
            'regional_risk': results_df.groupby('region')['ensemble_risk'].mean().to_dict(),
            'industry_risk': results_df.groupby('industry')['ensemble_risk'].mean().to_dict(),
            'model_contributions': {
                'gnn_avg': results_df['gnn_risk'].mean(),
                'ts_avg': results_df['ts_risk'].mean(),
                'cv_avg': results_df['cv_risk'].mean()
            },
            'top_risk_companies': results_df.nlargest(5, 'ensemble_risk')[
                ['company_name', 'region', 'industry', 'ensemble_risk']
            ].to_dict('records')
        }
        
        return analysis


def create_ensemble_predictions(
    cv_model: ActivityDetectionCNN,
    ts_model: TimeSeriesLSTM,
    gnn_model: SupplyChainGNN,
    ts_scaler,
    graph_data: Data,
    company_df: pd.DataFrame,
    time_series_df: pd.DataFrame,
    satellite_images: List[Dict[str, Any]],
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standalone function to create ensemble predictions.
    
    This is a simplified version for backward compatibility.
    """
    ensemble = EnsemblePredictor(device=device)
    ensemble.load_models(cv_model, ts_model, gnn_model, ts_scaler, graph_data)
    
    return ensemble.predict_ensemble(company_df, time_series_df, satellite_images)
