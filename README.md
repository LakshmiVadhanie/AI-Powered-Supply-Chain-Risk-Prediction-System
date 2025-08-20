# AI-Powered-Supply-Chain-Risk-Prediction-System
An AI-powered supply chain risk prediction system that combines Computer Vision, Time Series Analysis, and Graph Neural Networks to predict and analyze supply chain disruptions.




##  Features

- **Multi-Modal Risk Assessment**: Combines satellite imagery, temporal data, and network relationships
- **Real-time Monitoring**: Continuous risk assessment using time series forecasting
- **Network Analysis**: Graph neural networks to model supplier relationships
- **Risk Propagation**: Analysis of how disruptions spread through supply networks
- **Ensemble Predictions**: Weighted combination of multiple AI models for improved accuracy

##  Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Satellite     │    │   Time Series   │    │   Supply Chain  │
│   Images        │    │   Data          │    │   Network       │
│                 │    │                 │    │                 │
│   CNN Model     │    │   LSTM Model    │    │   GNN Model     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                         ┌───────▼───────┐
                         │   Ensemble    │
                         │   Predictor   │
                         └───────────────┘
```

## Installation

### Clone the repository
```bash
git clone https://github.com/yourusername/supply-chain-risk-prediction.git
cd supply-chain-risk-prediction
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Install the package
```bash
pip install -e .
```

##  Quick Start

### 1. Generate Sample Data
```python
from data.data_generator import generate_supply_chain_data
from scripts.train_models import train_all_models

# Generate synthetic supply chain data
company_df, relationships_df, time_series_df, satellite_images = generate_supply_chain_data()
```

### 2. Train Models
```python
# Train all models
models = train_all_models(company_df, relationships_df, time_series_df, satellite_images)
```

### 3. Run Predictions
```python
from scripts.run_prediction import predict_supply_chain_risk

# Get risk predictions
results = predict_supply_chain_risk(models, company_df, relationships_df, time_series_df)
print(f"Average risk score: {results['ensemble_risk'].mean():.3f}")
```

### 4. Visualize Results
```python
from utils.visualization import create_risk_dashboard

# Generate comprehensive risk dashboard
create_risk_dashboard(results, relationships_df)
```

##  Models

### Computer Vision Model
- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: 64x64 RGB satellite images
- **Output**: Factory activity level (0-1)
- **Use Case**: Detecting production activity from satellite imagery

### Time Series Model  
- **Architecture**: Long Short-Term Memory (LSTM)
- **Input**: 30-day sequences of risk indicators
- **Output**: Disruption probability
- **Use Case**: Forecasting supply chain disruptions

### Graph Neural Network
- **Architecture**: Graph Attention Network (GAT)
- **Input**: Company features + supplier relationships
- **Output**: Network-based risk scores
- **Use Case**: Modeling supply chain interdependencies

## Results

The system achieves:
- **Computer Vision**: MSE Loss < 0.05 on activity detection
- **Time Series**: 85%+ accuracy on disruption prediction  
- **Graph Neural Network**: 80%+ accuracy on network risk assessment
- **Ensemble Model**: Improved robustness through model combination


##  Project Structure

```
supply-chain-risk-prediction/
├── data/                 # Data generation and processing
├── models/              # AI model implementations
├── utils/               # Visualization and analysis utilities  
├── scripts/             # Training and prediction scripts
├── notebooks/           # Jupyter notebooks for exploration
├── tests/               # Unit tests
└── docs/                # Documentation
```

## 🔧 Configuration

Modify `config/config.yaml` to customize:
- Model hyperparameters
- Training settings
- Data generation parameters
- Visualization options

##  Usage Examples

### Custom Data Training
```python
# Train on your own data
from models.ensemble import SupplyChainEnsemble

ensemble = SupplyChainEnsemble()
ensemble.train(your_company_data, your_relationships, your_time_series)
predictions = ensemble.predict(new_data)
```

### Risk Analysis
```python
from utils.risk_analysis import RiskAnalyzer

analyzer = RiskAnalyzer(supply_network)
propagation_risks = analyzer.calculate_propagation_risk()
critical_nodes = analyzer.identify_critical_suppliers()
```


##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

