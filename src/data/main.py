#!/usr/bin/env python3
"""
AI-Powered Supply Chain Risk Prediction System
Main execution script
"""

from scripts.run_prediction import predict_supply_chain_risk

def main():
    """Main function to run the complete supply chain risk prediction system"""
    print("AI-POWERED SUPPLY CHAIN RISK PREDICTION SYSTEM")
    
    try:
        # Run the complete prediction pipeline
        results_df = predict_supply_chain_risk()
        return results_df
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("check dependencies")
        return None

if __name__ == "__main__":
    results = main()
