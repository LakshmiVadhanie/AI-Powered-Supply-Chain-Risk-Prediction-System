# RISK PROPAGATION ANALYSIS
def analyze_risk_propagation(supply_network, results_df):
    propagation_scores = {}

    for node in supply_network.nodes():
        # Calculate how risk might propagate from this node
        successors = list(supply_network.successors(node))
        predecessors = list(supply_network.predecessors(node))

        # Current node risk
        current_risk = results_df[results_df['company_id'] == node]['ensemble_risk'].iloc[0]

        # Impact on downstream (successors)
        downstream_impact = len(successors) * current_risk

        # Vulnerability from upstream (predecessors)
        upstream_risk = sum([results_df[results_df['company_id'] == pred]['ensemble_risk'].iloc[0]
                            for pred in predecessors]) / max(len(predecessors), 1)

        # Combined propagation score
        propagation_score = 0.6 * downstream_impact + 0.4 * upstream_risk
        propagation_scores[node] = propagation_score

    # Add propagation scores to results
    results_df['propagation_risk'] = results_df['company_id'].map(propagation_scores)

    # Visualize top propagation risks
    plt.figure(figsize=(12, 6))
    top_propagation = results_df.nlargest(10, 'propagation_risk')

    plt.barh(top_propagation['company_name'], top_propagation['propagation_risk'])
    plt.title('Top 10 Companies by Risk Propagation Potential')
    plt.xlabel('Propagation Risk Score')
    plt.tight_layout()
    plt.show()

    return results_df

results_df = analyze_risk_propagation(supply_network, results_df)
