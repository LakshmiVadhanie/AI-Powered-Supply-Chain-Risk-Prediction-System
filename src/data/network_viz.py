import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd

def visualize_results(company_df, ensemble_preds, gnn_preds, ts_preds, cv_preds):
    # Create results dataframe
    results_df = company_df.copy()
    results_df['ensemble_risk'] = ensemble_preds.detach().numpy()
    results_df['gnn_risk'] = gnn_preds.detach().numpy()
    results_df['ts_risk'] = ts_preds.detach().numpy()
    results_df['cv_risk'] = cv_preds.detach().numpy()

    # Plot 1: Risk distribution by region
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.boxplot(data=results_df, x='region', y='ensemble_risk')
    plt.title('Supply Chain Risk by Region')
    plt.xticks(rotation=45)

    # Plot 2: Model comparison
    plt.subplot(2, 2, 2)
    model_risks = {
        'GNN': results_df['gnn_risk'].mean(),
        'Time Series': results_df['ts_risk'].mean(),
        'Computer Vision': results_df['cv_risk'].mean(),
        'Ensemble': results_df['ensemble_risk'].mean()
    }
    plt.bar(model_risks.keys(), model_risks.values())
    plt.title('Average Risk Prediction by Model')
    plt.ylabel('Risk Score')

    # Plot 3: Top risk companies
    plt.subplot(2, 2, 3)
    top_risk = results_df.nlargest(10, 'ensemble_risk')
    plt.barh(top_risk['company_name'], top_risk['ensemble_risk'])
    plt.title('Top 10 Companies at Risk')
    plt.xlabel('Risk Score')

    # Plot 4: Risk by industry
    plt.subplot(2, 2, 4)
    industry_risk = results_df.groupby('industry')['ensemble_risk'].mean().sort_values(ascending=False)
    plt.bar(industry_risk.index, industry_risk.values)
    plt.title('Average Risk by Industry')
    plt.xticks(rotation=45)
    plt.ylabel('Risk Score')

    plt.tight_layout()
    plt.show()

    return results_df

def visualize_supply_network(company_df, relationships_df, results_df):
    # Create network graph
    G = nx.DiGraph()

    # Add nodes
    for _, company in company_df.iterrows():
        risk_score = results_df[results_df['company_id'] == company['company_id']]['ensemble_risk'].iloc[0]
        G.add_node(company['company_id'],
                  name=company['company_name'],
                  region=company['region'],
                  risk=risk_score)

    # Add edges
    for _, rel in relationships_df.iterrows():
        G.add_edge(rel['supplier_id'], rel['buyer_id'],
                  weight=rel['relationship_strength'])

    # Plot network
    fig, ax = plt.subplots(figsize=(12, 8)) # Create figure and axes
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Color nodes by risk level
    node_colors = [results_df[results_df['company_id'] == node]['ensemble_risk'].iloc[0]
                   for node in G.nodes()]

    nx.draw(G, pos, node_color=node_colors, cmap='Reds',
            node_size=300, with_labels=False, arrows=True,
            edge_color='gray', alpha=0.7, ax=ax) # Pass axes to draw function

    # Add labels for high-risk nodes
    high_risk_nodes = {node: data['name'] for node, data in G.nodes(data=True)
                       if data['risk'] > 0.7}
    nx.draw_networkx_labels(G, pos, high_risk_nodes, font_size=8, ax=ax) # Pass axes to draw_networkx_labels

    plt.title('Supply Chain Network (Red = High Risk)')
    plt.colorbar(plt.cm.ScalarMappable(cmap='Reds'), ax=ax, label='Risk Score') # Pass axes to colorbar
    plt.axis('off')
    plt.show()

    return G
