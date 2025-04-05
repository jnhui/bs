"""
Comprehensive analysis of ablation studies and human-computer interaction results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np

# Set style
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

def load_ablation_results():
    """Load and process ablation study results."""
    results_dir = Path("experiments/results/ablation")
    
    # Load analysis results
    with open(results_dir / "ablation_analysis.json", "r") as f:
        analysis = json.load(f)
        
    # Load detailed results
    results_df = pd.read_csv(results_dir / "ablation_results.csv")
    summary_df = pd.read_csv(results_dir / "ablation_summary.csv", index_col=0)
    
    return analysis, results_df, summary_df

def plot_component_impacts(analysis):
    """Plot component impact analysis."""
    plt.figure(figsize=(12, 6))
    
    # Component impacts
    impacts = pd.Series(analysis['component_impacts'])
    
    # Plot impacts
    ax = plt.subplot(1, 2, 1)
    impacts.plot(kind='bar')
    plt.title('Component Impact Analysis')
    plt.xlabel('Component')
    plt.ylabel('Impact on Confidence')
    plt.xticks(rotation=45)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(impacts):
        plt.text(i, v + (0.01 if v >= 0 else -0.01), 
                f'{v:.3f}', 
                ha='center',
                va='bottom' if v >= 0 else 'top')
    
    # Configuration comparison
    plt.subplot(1, 2, 2)
    config_means = pd.Series(analysis['config_means'])
    config_means.plot(kind='bar')
    plt.title('Configuration Performance')
    plt.xlabel('Configuration')
    plt.ylabel('Mean Confidence')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('experiments/results/component_impact_analysis.png')
    plt.close()

def plot_confidence_distributions(results_df):
    """Plot confidence score distributions."""
    plt.figure(figsize=(15, 5))
    
    # Confidence distribution by configuration
    plt.subplot(1, 3, 1)
    sns.boxplot(data=results_df, x='config', y='confidence')
    plt.title('Confidence Distribution by Configuration')
    plt.xticks(rotation=45)
    
    # Confidence density plot
    plt.subplot(1, 3, 2)
    for config in results_df['config'].unique():
        sns.kdeplot(data=results_df[results_df['config'] == config], 
                   x='confidence',
                   label=config)
    plt.title('Confidence Density by Configuration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Problem-wise confidence comparison
    plt.subplot(1, 3, 3)
    pivot_df = results_df.pivot(columns='config', values='confidence')
    sns.heatmap(pivot_df.corr(), annot=True, cmap='RdYlGn')
    plt.title('Configuration Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('experiments/results/confidence_distribution_analysis.png')
    plt.close()

def analyze_interaction_patterns(results_df):
    """Analyze patterns in problem-solving interactions."""
    # Extract problem characteristics
    results_df['problem_length'] = results_df['problem'].str.len()
    results_df['has_numbers'] = results_df['problem'].str.contains(r'\d+').astype(int)
    
    plt.figure(figsize=(15, 5))
    
    # Problem length vs confidence
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=results_df, x='problem_length', y='confidence', 
                   hue='config', alpha=0.6)
    plt.title('Problem Length vs Confidence')
    
    # Average confidence by problem type
    plt.subplot(1, 3, 2)
    sns.barplot(data=results_df, x='has_numbers', y='confidence', hue='config')
    plt.title('Confidence by Problem Type')
    plt.xlabel('Contains Numbers')
    
    # Configuration performance stability
    plt.subplot(1, 3, 3)
    stability_data = results_df.groupby('config')['confidence'].agg(['mean', 'std']).reset_index()
    sns.scatterplot(data=stability_data, x='mean', y='std', s=100)
    for _, row in stability_data.iterrows():
        plt.annotate(row['config'], (row['mean'], row['std']))
    plt.title('Configuration Stability Analysis')
    plt.xlabel('Mean Confidence')
    plt.ylabel('Standard Deviation')
    
    plt.tight_layout()
    plt.savefig('experiments/results/interaction_pattern_analysis.png')
    plt.close()

def main():
    """Run comprehensive analysis."""
    # Load results
    analysis, results_df, summary_df = load_ablation_results()
    
    # Generate visualizations
    plot_component_impacts(analysis)
    plot_confidence_distributions(results_df)
    analyze_interaction_patterns(results_df)
    
    # Print summary statistics
    print("\nAblation Study Summary:")
    print("=" * 50)
    print("\nComponent Impacts:")
    for component, impact in analysis['component_impacts'].items():
        print(f"{component:12}: {impact:+.3f}")
    
    print("\nConfiguration Performance:")
    print("-" * 50)
    for config, stats in analysis['summary'].items():
        print(f"\n{config}:")
        print(f"  Mean confidence: {stats['confidence_mean']:.3f}")
        print(f"  Std deviation:   {stats['confidence_std']:.3f}")
        print(f"  Problem count:   {stats['problem_count']}")

if __name__ == "__main__":
    main()
