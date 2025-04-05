"""
Visualization utilities for analyzing confidence scores and decomposition decisions.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import seaborn as sns
import networkx as nx

def plot_confidence_distribution(decomposition_history: List[List[Dict]], save_path: str = 'confidence_distribution.png'):
    """
    Plot the distribution of confidence scores across all decompositions.
    
    Args:
        decomposition_history: List of decomposition decision logs
        save_path: Path to save the plot
    """
    confidences = []
    for log in decomposition_history:
        confidences.extend([d['confidence'] for d in log])
        
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=confidences, bins=20, kde=True, ax=ax)
    ax.set_title('Distribution of Confidence Scores')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.axvline(x=0.7, color='r', linestyle='--', label='Default Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
def analyze_decomposition_decisions(decomposition_history: List[List[Dict]]) -> pd.DataFrame:
    """
    Analyze the relationship between confidence scores and decomposition decisions.
    
    Args:
        decomposition_history: List of decomposition decision logs
        
    Returns:
        DataFrame with analysis of confidence scores by depth
    """
    data = []
    for depth, log in enumerate(decomposition_history):
        for decision in log:
            data.append({
                'depth': depth,
                'confidence': decision['confidence'],
                'decomposed': decision['needs_decomposition'],
                'question': decision['question']
            })
    
    df = pd.DataFrame(data)
    summary = df.groupby('depth').agg({
        'confidence': ['mean', 'std', 'count'],
        'decomposed': 'mean'
    }).round(3)
    
    # Plot confidence by depth
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='depth', y='confidence')
    plt.title('Confidence Scores by Decomposition Depth')
    plt.xlabel('Recursion Depth')
    plt.ylabel('Confidence Score')
    plt.savefig('confidence_by_depth.png')
    plt.close()
    
    return summary

def plot_decomposition_tree(decomposition_history: List[List[Dict]], save_path: str = 'decomposition_tree.png'):
    """
    Visualize the question decomposition tree with confidence scores.
    
    Args:
        decomposition_history: List of decomposition decision logs
        save_path: Path to save the plot
    """
    import networkx as nx
    
    G = nx.DiGraph()
    node_colors = []
    
    for depth, log in enumerate(decomposition_history):
        for i, decision in enumerate(log):
            node_id = f"{depth}_{i}"
            G.add_node(node_id)
            node_colors.append(decision['confidence'])
            
            if depth > 0:
                # Connect to parent node
                parent_id = f"{depth-1}_{i//2}"
                G.add_edge(parent_id, node_id)
    
    try:
        fig, ax = plt.subplots(figsize=(15, 10))
        pos = nx.spring_layout(G)
        
        # Create mappable for colorbar with proper normalization
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
        sm.set_array([])  # Set empty array for full range
        
        # Draw network with error handling
        if len(G.nodes()) > 0:
            nx.draw(G, pos, node_color=node_colors, 
                   node_size=1000, cmap='RdYlGn',
                   with_labels=True, font_size=8,
                   vmin=0, vmax=1,  # Set fixed range for confidence scores
                   ax=ax)
            
            # Add colorbar with fixed position
            plt.colorbar(sm, ax=ax, label='Confidence Score')
            plt.title('Question Decomposition Tree')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            print("Warning: Empty graph, skipping visualization")
            
        plt.close()
    except Exception as e:
        print(f"Error generating decomposition tree plot: {str(e)}")
        plt.close('all')  # Ensure all figures are closed on error
