import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def set_seeds():
    import torch
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

def generate_supply_chain_data():
    # Create synthetic supplier network
    companies = ['Apple', 'Samsung', 'TSMC', 'Foxconn', 'Intel', 'Nvidia', 'AMD', 'Qualcomm',
                'Sony', 'LG', 'Tesla', 'Toyota', 'Ford', 'GM', 'VW', 'BMW', 'Mercedes', 'Honda',
                'Walmart', 'Amazon', 'Target', 'Nike', 'Adidas', 'H&M', 'Zara', 'Uniqlo']

    regions = ['North America', 'Europe', 'East Asia', 'Southeast Asia', 'South America', 'Africa']
    industries = ['Electronics', 'Automotive', 'Retail', 'Textiles', 'Semiconductors', 'Manufacturing']

    # Generate company data
    company_data = []
    for i, company in enumerate(companies):
        company_data.append({
            'company_id': i,
            'company_name': company,
            'region': random.choice(regions),
            'industry': random.choice(industries),
            'size': random.choice(['Small', 'Medium', 'Large']),
            'financial_health': random.uniform(0.3, 1.0)
        })

    return pd.DataFrame(company_data)

def generate_supply_relationships(company_df):
    relationships = []
    n_companies = len(company_df)

    # Create supply chain edges
    for i in range(n_companies):
        # Each company has 2-5 suppliers
        n_suppliers = random.randint(2, 5)
        suppliers = random.sample(range(n_companies), min(n_suppliers, n_companies-1))

        for supplier in suppliers:
            if supplier != i:  # No self-loops
                relationships.append({
                    'supplier_id': supplier,
                    'buyer_id': i,
                    'relationship_strength': random.uniform(0.1, 1.0),
                    'lead_time_days': random.randint(7, 90),
                    'volume_score': random.uniform(0.2, 1.0)
                })

    return pd.DataFrame(relationships)

def generate_time_series_data(company_df, days=365):
    time_series_data = []
    start_date = datetime(2023, 1, 1)

    for _, company in company_df.iterrows():
        company_id = company['company_id']

        for day in range(days):
            current_date = start_date + timedelta(days=day)

            # Simulate various risk factors with trends and seasonality
            base_risk = 0.1
            seasonal_factor = 0.05 * np.sin(2 * np.pi * day / 365)  # Yearly cycle
            trend_factor = 0.001 * day  # Slight upward trend
            noise = random.gauss(0, 0.02)

            # company-specific factors
            region_risk = {'North America': 0.05, 'Europe': 0.07, 'East Asia': 0.15,
                          'Southeast Asia': 0.20, 'South America': 0.25, 'Africa': 0.30}

            risk_score = (base_risk + seasonal_factor + trend_factor + noise +
                         region_risk.get(company['region'], 0.1))
            risk_score = max(0, min(1, risk_score))  # Clamp between 0 and 1

            time_series_data.append({
                'company_id': company_id,
                'date': current_date,
                'risk_score': risk_score,
                'demand_volatility': random.uniform(0.1, 0.8),
                'price_volatility': random.uniform(0.1, 0.6),
                'inventory_level': random.uniform(0.2, 1.0),
                'disruption_occurred': 1 if risk_score > 0.7 else 0
            })

    return pd.DataFrame(time_series_data)

def generate_synthetic_satellite_images(n_images=100):
    images_data = []

    for i in range(n_images):
        # Create synthetic satellite image (64x64 RGB)
        # Simulate factory activity levels
        activity_level = random.uniform(0, 1)

        # Base image with some industrial patterns
        img = np.random.randint(50, 150, (64, 64, 3), dtype=np.uint8)

        # activity indicators (brighter spots for higher activity)
        if activity_level > 0.5:
            # bright spots for high activity
            for _ in range(int(activity_level * 10)):
                x, y = random.randint(5, 58), random.randint(5, 58)
                img[x:x+6, y:y+6] = np.minimum(img[x:x+6, y:y+6] + 50, 255)

        images_data.append({
            'image_id': i,
            'company_id': random.randint(0, 25),  # Assuming 26 companies
            'activity_level': activity_level,
            'image_array': img
        })

    return images_data
