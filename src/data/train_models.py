#!/usr/bin/env python3
"""
Main training script for the Supply Chain Risk Prediction System.
Trains all models and saves them for inference.
"""

import os
import sys
import torch
import pickle
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_generator import generate_all_data
from src.models.computer_vision import train_cv_model
from src.models.time_series import train_time_series_model
from src.models.graph_neural_network import train_gnn_model, prepare_graph_data
from src.models.ensemble import create_ensemble_predictions
from src.visualization.network_viz import visualize_supply_network
from src.visualization.risk_analysis import visualize_results, analyze_risk_propagation


def setup_directories():
    """Create necessary directories for data and models."""
    directories = [
        'data/synthetic',
        'models/saved_models',
        'models/checkpoints', 
        'results/figures',
        'results/reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_models_and_data(cv_model, ts_model, gnn_model, ts_scaler, graph_data, 
                        company_df, relationships_df, time_series_df, satellite_images):
    """Save trained models and data."""
    
    # Save models
    torch.save(cv_model.state_dict(), 'models/saved_models/cv_model.pth')
    torch.save(ts_model.state_dict(), 'models/saved_models/ts_model.pth')
    torch.save(gnn_model.state_dict(), 'models/saved_models/gnn_model.pth')
    
    # Save preprocessors and data
    with open('models/saved_models/ts_scaler.pkl', 'wb') as f:
        pickle.dump(ts_scaler, f)
    
    torch.save(graph_data, 'models/saved_models/graph_data.pth')
    
    # Save datasets
    company_df.to_csv('data/synthetic/companies.csv', index=False)
    relationships_df.to_csv('data/synthetic/relationships.csv', index=False)
    time_series_df.to_csv('data/synthetic/time_series.csv', index=False)
    
    with open('data/synthetic/satellite_images.pkl', 'wb') as f:
        pickle.dump(satellite_images, f)
    
    print(" Models and data saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='Train Supply Chain Risk Prediction Models')
    parser.add_argument('--device', default='cpu', help='Device to use for training')
    parser.add_argument('--cv_epochs', type=int, default=20, help='CV model epochs')
    parser.add_argument('--ts_epochs', type=int, default=30, help='Time series epochs')  
    parser.add_argument('--gnn_epochs', type=int, default=100, help='GNN epochs')
    parser.add_argument('--skip_viz', action='store_true', help='Skip visualizations')
    
    args = parser.parse_args()
    
    print(" Starting Supply Chain Risk Prediction Training Pipeline")
    print(f"Using device: {args.device}")
    
    # Setup
    setup_directories()
    device = torch.device(args.device)
    
    # Generate synthetic data
    print("\nGenerating synthetic supply chain data...")
    company_df, relationships_df, time_series_df, satellite_images = generate_all_data()
    
    print(f" Data generated:")
    print(f"   - Companies: {len(company_df)}")
    print(f"   - Relationships: {len(relationships_df)}")
    print(f"   - Time series records: {len(time_series_df)}")
    print(f"   - Satellite images: {len(satellite_images)}")
    
    # Train Computer Vision Model
    print("\n  Training Computer Vision model...")
    cv_model = train_cv_model(
        satellite_images, 
        epochs=args.cv_epochs,
        device=args.device
    )
    
    # Train Time Series Model  
    print("\n Training Time Series model...")
    ts_model, ts_scaler = train_time_series_model(
        time_series_df,
        epochs=args.ts_epochs,
        device=args.device
    )
    
    # Prepare and train Graph Neural Network
    print("\n  Preparing graph data and training GNN...")
    graph_data = prepare_graph_data(company_df, relationships_df, time_series_df)
    gnn_model, graph_data = train_gnn_model(
        company_df, relationships_df, time_series_df,
        epochs=args.gnn_epochs,
        device=args.device
    )
    
    # Create ensemble predictions
    print("\n Creating ensemble predictions")
    ensemble_preds, gnn_preds, ts_preds, cv_preds = create_ensemble_predictions(
        cv_model, ts_model, gnn_model, ts_scaler, graph_data,
        company_df, time_series_df, satellite_images, args.device
    )
    
    # Save everything
    print("\n Saving models and data")
    save_models_and_data(
        cv_model, ts_model, gnn_model, ts_scaler, graph_data,
        company_df, relationships_df, time_series_df, satellite_images
    )
    
    # Generate visualizations and analysis
    if not args.skip_viz:
        print("\n Generating visualizations")
        
        # Import visualization functions
        from src.visualization.risk_analysis import visualize_results, analyze_risk_propagation
        
        # Create results dataframe
        results_df = visualize_results(company_df, ensemble_preds, gnn_preds, ts_preds, cv_preds)
        
        # Visualize supply network
        supply_network = visualize_supply_network(
            company_df, relationships_df, ensemble_preds.detach().numpy()
        )
        
        # Analyze risk propagation
        results_df = analyze_risk_propagation(supply_network, results_df)
        
        # Save results
        results_df.to_csv('results/reports/risk_analysis_results.csv', index=False)
        
        print(" Visualizations saved to results/figures/")
    
    # Print summary statistics
    print("\n" + "="*60)
    print(" TRAINING COMPLETE - SUMMARY STATISTICS")
    print("="*60)
    
    ensemble_np = ensemble_preds.detach().cpu().numpy()
    
    print(f" Average Risk Score: {ensemble_np.mean():.3f}")
    print(f"High Risk Companies (>0.7): {(ensemble_np > 0.7).sum()}")
    print(f" Total Companies Analyzed: {len(ensemble_np)}")
    
    # Regional analysis
    results_df = company_df.copy()
    results_df['ensemble_risk'] = ensemble_np
    
    print(f"\nREGIONAL RISK ANALYSIS:")
    regional_risk = results_df.groupby('region')['ensemble_risk'].mean().sort_values(ascending=False)
    for region, risk in regional_risk.items():
        print(f"   - {region}: {risk:.3f}")
    
    print(f"\nINDUSTRY RISK ANALYSIS:")
    industry_risk = results_df.groupby('industry')['ensemble_risk'].mean().sort_values(ascending=False)
    for industry, risk in industry_risk.items():
        print(f"   - {industry}: {risk:.3f}")
    
    print(f"\n TOP 5 COMPANIES AT RISK:")
    top_risk = results_df.nlargest(5, 'ensemble_risk')
    for _, company in top_risk.iterrows():
        print(f"   - {company['company_name']} ({company['region']}): {company['ensemble_risk']:.3f}")
    
    print("\n All models trained and saved successfully!")
    print(" Use 'python scripts/predict_risks.py' for inference")


if __name__ == "__main__":
    main()
