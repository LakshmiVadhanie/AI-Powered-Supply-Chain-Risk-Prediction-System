"""
Network visualization utilities for supply chain analysis.
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import seaborn as sns


def create_supply_network(
    company_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    risk_scores: Optional[pd.Series] = None
) -> nx.DiGraph:
    """Create a NetworkX graph from supply chain data."""
    
    G = nx.DiGraph()
    
    # Add nodes with company information
    for _, company in company_df.iterrows():
        risk_score = 0.5  # Default risk
        if risk_scores is not None and company['company_id'] in risk_scores.index:
            risk_score = risk_scores[company['company_id']]
        
        G.add_node(
            company['company_id'],
            name=company['company_name'],
            region=company['region'],
            industry=company['industry'],
            size=company['size'],
            risk=risk_score
        )
    
    # Add edges for supply relationships
    for _, rel in relationships_df.iterrows():
        G.add_edge(
            rel['supplier_id'],
            rel['buyer_id'],
            weight=rel['relationship_strength'],
            lead_time=rel['lead_time_days'],
            volume=rel['volume_score']
        )
    
    return G


def visualize_supply_network(
    company_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    risk_scores: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 12),
    node_size_factor: int = 500,
    show_labels: bool = True,
    high_risk_threshold: float = 0.7
) -> nx.DiGraph:
    """Visualize the supply chain network with risk coloring."""
    
    # Convert risk scores to pandas Series if provided as array
    if risk_scores is not None and isinstance(risk_scores, np.ndarray):
        risk_series = pd.Series(risk_scores, index=range(len(company_df)))
    else:
        risk_series = risk_scores
    
    # Create network graph
    G = create_supply_network(company_df, relationships_df, risk_series)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate layout
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    
    # Get node colors based on risk scores
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        risk = G.nodes[node]['risk']
        node_colors.append(risk)
        
        # Size based on degree (connectivity)
        degree = G.degree(node)
        node_sizes.append(max(node_size_factor + degree * 50, 200))
    
    # Draw the network
    edges = nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        alpha=0.6,
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        ax=ax
    )
    
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap='Reds',
        vmin=0, vmax=1,
        alpha=0.8,
        ax=ax
    )
    
    # Add labels for high-risk nodes or all nodes if requested
    if show_labels:
        if risk_series is not None:
            # Only label high-risk nodes
            high_risk_labels = {
                node: data['name'] 
                for node, data in G.nodes(data=True) 
                if data['risk'] > high_risk_threshold
            }
        else:
            # Label all nodes
            high_risk_labels = {
                node: data['name'] 
                for node, data in G.nodes(data=True)
            }
        
        nx.draw_networkx_labels(
            G, pos, 
            high_risk_labels, 
            font_size=8, 
            font_weight='bold',
            ax=ax
        )
    
    # Add colorbar
    if nodes is not None:
        cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
        cbar.set_label('Risk Score', fontsize=12)
    
    # Customize plot
    ax.set_title('Supply Chain Network Analysis\n(Node size = connectivity, Color = risk)', 
                fontsize=16, pad=20)
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=10, label=f'High Risk (>{high_risk_threshold})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, label=f'Low Risk (<{high_risk_threshold})'),
        plt.Line2D([0], [0], color='gray', label='Supply Relationship')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.show()
    
    return G


def analyze_network_centrality(G: nx.DiGraph) -> pd.DataFrame:
    """Analyze network centrality measures."""
    
    centrality_data = []
    
    # Calculate various centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    
    try:
        pagerank = nx.pagerank(G)
    except:
        pagerank = {node: 0 for node in G.nodes()}
    
    for node in G.nodes():
        centrality_data.append({
            'company_id': node,
            'company_name': G.nodes[node]['name'],
            'degree_centrality': degree_centrality[node],
            'betweenness_centrality': betweenness_centrality[node],
            'closeness_centrality': closeness_centrality[node],
            'pagerank': pagerank[node],
            'risk_score': G.nodes[node]['risk']
        })
    
    return pd.DataFrame(centrality_data)


def plot_centrality_analysis(centrality_df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)):
    """Plot centrality analysis results."""
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Degree centrality vs risk
    axes[0, 0].scatter(centrality_df['degree_centrality'], centrality_df['risk_score'], alpha=0.6)
    axes[0, 0].set_xlabel('Degree Centrality')
    axes[0, 0].set_ylabel('Risk Score')
    axes[0, 0].set_title('Degree Centrality vs Risk')
    
    # Betweenness centrality vs risk
    axes[0, 1].scatter(centrality_df['betweenness_centrality'], centrality_df['risk_score'], alpha=0.6)
    axes[0, 1].set_xlabel('Betweenness Centrality')
    axes[0, 1].set_ylabel('Risk Score')
    axes[0, 1].set_title('Betweenness Centrality vs Risk')
    
    # PageRank vs risk
    axes[1, 0].scatter(centrality_df['pagerank'], centrality_df['risk_score'], alpha=0.6)
    axes[1, 0].set_xlabel('PageRank')
    axes[1, 0].set_ylabel('Risk Score')
    axes[1, 0].set_title('PageRank vs Risk')
    
    # Top companies by centrality
    top_pagerank = centrality_df.nlargest(10, 'pagerank')
    axes[1, 1].barh(range(len(top_pagerank)), top_pagerank['pagerank'])
    axes[1, 1].set_yticks(range(len(top_pagerank)))
    axes[1, 1].set_yticklabels(top_pagerank['company_name'], fontsize=8)
    axes[1, 1].set_xlabel('PageRank Score')
    axes[1, 1].set_title('Top Companies by PageRank')
    
    plt.tight_layout()
    plt.show()


def visualize_risk_propagation(
    G: nx.DiGraph,
    source_nodes: list,
    figsize: Tuple[int, int] = (12, 8)
) -> Dict[int, float]:
    """Visualize how risk propagates through the network."""
    
    propagation_scores = {}
    
    for node in G.nodes():
        # Calculate propagation impact
        successors = list(G.successors(node))
        predecessors = list(G.predecessors(node))
        
        current_risk = G.nodes[node]['risk']
        
        # Impact on downstream
        downstream_impact = len(successors) * current_risk
        
        # Vulnerability from upstream  
        if predecessors:
            upstream_risk = sum(G.nodes[pred]['risk'] for pred in predecessors) / len(predecessors)
        else:
            upstream_risk = 0
        
        # Combined propagation score
        propagation_score = 0.6 * downstream_impact + 0.4 * upstream_risk
        propagation_scores[node] = propagation_score
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get top propagation risks
    top_propagation = sorted(propagation_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    companies = [G.nodes[node]['name'] for node, _ in top_propagation]
    scores = [score for _, score in top_propagation]
    
    ax.barh(companies, scores)
    ax.set_xlabel('Risk Propagation Score')
    ax.set_title('Top Companies by Risk Propagation Potential')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return propagation_scores
