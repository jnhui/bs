"""
Metrics collection and analysis for human-computer interaction experiments.
Tracks recursive questioning process and calculates reward signals.
"""

import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class ExperimentMetrics:
    def __init__(self, log_dir: str = "experiments/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            'question_quality': [],
            'recursion_depth': [],
            'solution_time': [],
            'user_rating': [],
            'clarity_rating': [],
            'helpfulness_rating': [],
            'system_confidence': [],
            'confidence_accuracy': []
        }
        
    def log_interaction(self, interaction_data: Dict[str, Any]):
        """Log a single interaction with metrics."""
        try:
            # Handle key mapping for clarity and helpfulness
            clarity = interaction_data.get('clarity_rating', interaction_data.get('clarity', 0))
            helpfulness = interaction_data.get('helpfulness_rating', interaction_data.get('helpfulness', 0))
            
            metrics = {
                'timestamp': float(interaction_data.get('timestamp', time.time())),
                'problem': str(interaction_data['problem']),
                'num_sub_questions': len(interaction_data.get('sub_questions', [])),
                'solution_time': float(interaction_data.get('processing_time', 0)),
                'system_confidence': float(interaction_data['system_confidence']),
                'user_rating': float(interaction_data['user_rating']),
                'clarity_rating': float(clarity),
                'helpfulness_rating': float(helpfulness),
                'recursion_depth': len(interaction_data.get('sub_questions', [])),
                'confidence_accuracy': float(interaction_data['confidence_accuracy'])
            }
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid interaction data format: {str(e)}")
        
        # Calculate question quality score
        quality_score = self._calculate_question_quality(interaction_data)
        metrics['question_quality'] = quality_score
        
        # Save individual interaction log
        log_file = self.log_dir / f"interaction_{metrics['timestamp']}.json"
        with open(log_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Update metrics collections
        for key in self.metrics:
            if key in metrics:
                self.metrics[key].append(metrics[key])
                
    def _calculate_question_quality(self, data: Dict[str, Any]) -> float:
        """
        Calculate question quality score as reward signal.
        Considers:
        1. User ratings
        2. Solution clarity
        3. Recursion effectiveness
        4. Confidence accuracy
        """
        weights = {
            'user_rating': 0.3,
            'clarity_rating': 0.2,
            'helpfulness_rating': 0.2,
            'confidence_accuracy': 0.3
        }
        
        scores = {
            'user_rating': data['user_rating'] / 5.0,
            'clarity_rating': data['clarity_rating'] / 5.0,
            'helpfulness_rating': data['helpfulness_rating'] / 5.0,
            'confidence_accuracy': data['confidence_accuracy'] / 5.0
        }
        
        return sum(weights[k] * scores[k] for k in weights)
        
    def analyze_recursion_effectiveness(self) -> Dict[str, float]:
        """Analyze effectiveness of recursive questioning strategy."""
        df = pd.DataFrame(self.metrics)
        
        analysis = {
            'mean_recursion_depth': float(df['recursion_depth'].mean()) if not df.empty else 0.0,
            'depth_quality_correlation': float(df['recursion_depth'].corr(df['question_quality'])) if not df.empty else 0.0,
            'optimal_depth': int(df.groupby('recursion_depth')['question_quality'].mean().idxmax()) if not df.empty and len(df.groupby('recursion_depth')) > 0 else 0,
            'quality_improvement': float(
                df[df['recursion_depth'] > 0]['question_quality'].mean() -
                df[df['recursion_depth'] == 0]['question_quality'].mean()
            ) if not df.empty and len(df[df['recursion_depth'] > 0]) > 0 else 0.0
        }
        
        return analysis
        
    def generate_report(self, save_dir: str = "experiments/results") -> Dict[str, Any]:
        """Generate comprehensive experiment report with visualizations."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Convert metrics to DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Generate visualizations
        plt.figure(figsize=(15, 10))
        
        # Question quality distribution
        plt.subplot(2, 2, 1)
        sns.histplot(data=df, x='question_quality', bins=20)
        plt.title('Question Quality Distribution')
        plt.xlabel('Quality Score')
        
        # Recursion depth vs quality
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=df, x='recursion_depth', y='question_quality')
        plt.title('Question Quality vs Recursion Depth')
        
        # User ratings correlation
        plt.subplot(2, 2, 3)
        sns.heatmap(df[['user_rating', 'confidence_accuracy', 'question_quality']].corr(),
                   annot=True, cmap='RdYlGn')
        plt.title('Metrics Correlation')
        
        # Solution time vs quality
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=df, x='solution_time', y='question_quality')
        plt.title('Solution Time vs Quality')
        
        plt.tight_layout()
        plt.savefig(str(save_path / 'experiment_analysis.png'))
        plt.close()
        
        # Save summary statistics
        summary = {
            'total_interactions': len(df),
            'mean_quality_score': float(df['question_quality'].mean()),
            'mean_user_rating': float(df['user_rating'].mean()),
            'confidence_correlation': float(df['confidence_accuracy'].corr(df['question_quality'])),
            'recursion_analysis': self.analyze_recursion_effectiveness()
        }
        
        with open(str(save_path / 'experiment_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary
