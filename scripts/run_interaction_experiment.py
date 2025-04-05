"""
Script for conducting human-computer interaction experiments with the Socratic questioning system.
Tracks recursive cognitive mechanism and evaluates question quality as reward signal.
"""

import json
import pandas as pd
from pathlib import Path
import os
from typing import Dict, List, Any
import time
from flask import Flask, render_template, request, jsonify

from src.socratic_questioner import SocraticQuestioner
from src.visualization import plot_confidence_distribution
from experiments.metrics.experiment_metrics import ExperimentMetrics

# 修改Flask应用初始化，指定正确的模板文件夹路径
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_dir)
questioner = SocraticQuestioner()
metrics = ExperimentMetrics()


@app.route('/')
def home():
    """Render experiment interface."""
    return render_template('experiment.html')


@app.route('/solve', methods=['POST'])
def solve_problem():
    """Process problem and return solution with confidence."""
    data = request.json
    problem = data['problem']
    start_time = time.time()

    try:
        # Get solution and track recursive process
        solution, confidence = questioner.solve_problem(problem)
        sub_questions = questioner.decompose_question(problem)
        processing_time = time.time() - start_time

        # Track decomposition history
        decomposition_data = {
            'problem': problem,
            'sub_questions': sub_questions,
            'confidence': confidence,
            'solution': solution,
            'processing_time': processing_time,
            'recursion_depth': len(questioner.decomposition_history),
            'timestamp': time.time()
        }

        return jsonify(decomposition_data)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': time.time()
        }), 500


@app.route('/feedback', methods=['POST'])
def collect_feedback():
    """Collect feedback and calculate question quality reward signal."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        # Validate required fields
        required_fields = ['problem', 'solution', 'system_confidence',
                           'rating', 'clarity', 'helpfulness', 'confidence_accuracy']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'status': 'error'
            }), 400

        # Prepare interaction data with metrics and type conversion
        try:
            interaction_data = {
                'problem': str(data['problem']),
                'solution': str(data['solution']),
                'system_confidence': float(data['system_confidence']),
                'user_rating': float(data['rating']),
                'clarity_rating': float(data['clarity']),
                'helpfulness_rating': float(data['helpfulness']),
                'confidence_accuracy': float(data['confidence_accuracy']),
                'comments': str(data.get('comments', '')),
                'processing_time': float(data.get('processing_time', 0)),
                'recursion_depth': len(questioner.decomposition_history) if hasattr(questioner,
                                                                                    'decomposition_history') else 0,
                'sub_questions': list(data.get('sub_questions', [])),
                'timestamp': time.time()
            }
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': f'Invalid data format: {str(e)}',
                'status': 'error'
            }), 400

        # Log interaction and calculate metrics
        metrics.log_interaction(interaction_data)

        # Generate updated report
        summary = metrics.generate_report()

        return jsonify({
            'status': 'success',
            'metrics_summary': summary
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get current experiment metrics and visualizations."""
    try:
        summary = metrics.generate_report()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Ensure experiment directories exist
    Path('experiments/logs').mkdir(parents=True, exist_ok=True)
    Path('experiments/results').mkdir(parents=True, exist_ok=True)

    # Start Flask server
    app.run(debug=True, port=5000)
