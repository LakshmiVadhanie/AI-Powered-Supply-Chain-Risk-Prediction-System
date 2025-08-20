import warnings
warnings.filterwarnings('ignore')

from data.data_generator import set_seeds, generate_supply_chain_data, generate_supply_relationships, generate_time_series_data, generate_synthetic_satellite_images
from models.computer_vision import train_cv_model
from models.time_series import prepare_time_series_data, train_time_series_model
from models.graph_neural_network import prepare_graph_data, train_gnn_model

def train_all_models():
    # Set seeds for reproducibility
    set_seeds()
    
    # Generate data
    print("Generating synthetic data...")
    company_df = generate_supply_chain_data()
    print("Company Data Generated:")
    print(company_df.head())
    
    relationships_df = generate_supply_relationships(company_df)
    print(f"\nSupply Relationships Generated: {len(relationships_df)} edges")
    print(relationships_df.head())
    
    time_series_df = generate_time_series_data(company_df)
    print(f"\nTime Series Data Generated: {len(time_series_df)} records")
    print(time_series_df.head())
    
    satellite_images = generate_synthetic_satellite_images()
    print(f"\nSynthetic Satellite Images Generated: {len(satellite_images)} images")
    
    # Train Computer Vision Model
    print("\nTraining Computer Vision Model...")
    cv_model = train_cv_model(satellite_images)
    
    # Train Time Series Model
    print("\nTraining Time Series Model...")
    X_seq, y_seq, ts_scaler = prepare_time_series_data(time_series_df)
    print(f"Time Series Sequences: {X_seq.shape}, Labels: {y_seq.shape}")
    ts_model = train_time_series_model(X_seq, y_seq)
    
    # Train Graph Neural Network
    print("\nTraining Graph Neural Network...")
    graph_data = prepare_graph_data(company_df, relationships_df, time_series_df)
    print(f"Graph Data - Nodes: {graph_data.x.shape[0]}, Edges: {graph_data.edge_index.shape[1]}")
    gnn_model = train_gnn_model(graph_data)
    
    return {
        'cv_model': cv_model,
        'ts_model': ts_model,
        'gnn_model': gnn_model,
        'graph_data': graph_data,
        'X_seq': X_seq,
        'ts_scaler': ts_scaler,
        'company_df': company_df,
        'relationships_df': relationships_df,
        'time_series_df': time_series_df,
        'satellite_images': satellite_images
    }

if __name__ == "__main__":
    models_data = train_all_models()
