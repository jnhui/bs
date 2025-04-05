"""
Script to conduct automated interaction experiments with sample problems.
Tests recursive cognitive mechanism and collects metrics.
"""

import json
import time
import requests
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List

def load_test_problems(n_samples: int = 5) -> List[Dict[str, Any]]:
    """Load sample problems from MATH dataset."""
    with open("data/MATH/test.json", 'r') as f:
        data = json.load(f)
    return data['problems'][:n_samples]

def simulate_user_rating(solution: str, correct_answer: str) -> Dict[str, float]:
    """Simulate user ratings based on solution quality."""
    from Levenshtein import ratio
    
    # Calculate text similarity for basic rating
    similarity = ratio(solution.lower(), correct_answer.lower())
    
    # Check for step-by-step reasoning
    has_steps = any(indicator in solution.lower() 
                   for indicator in ['step', 'first', 'then', 'finally'])
    
    # Generate ratings (as floats)
    ratings = {
        'rating': float(min(5, max(1, round(similarity * 5)))),
        'clarity': float(min(5, max(1, round((similarity + float(has_steps)) * 2.5)))),
        'helpfulness': float(min(5, max(1, round(similarity * 5)))),
        'confidence_accuracy': float(min(5, max(1, round(similarity * 5))))
    }
    
    return ratings

def run_experiments(base_url: str = "http://localhost:5000", debug: bool = True):
    """Run automated interaction experiments with debug logging."""
    """Run automated interaction experiments."""
    problems = load_test_problems()
    results = []
    
    print(f"\nStarting interaction experiments with {len(problems)} problems...")
    
    for i, problem in enumerate(problems, 1):
        print(f"\nProcessing problem {i}/{len(problems)}:")
        print(f"Subject: {problem.get('subject', 'unknown')}")
        print(f"Problem: {problem['problem']}")
        
        try:
            # First solve the problem
            try:
                solve_response = requests.post(
                    f"{base_url}/solve",
                    json={'problem': problem['problem']},
                    headers={'Content-Type': 'application/json'}
                )
                solve_response.raise_for_status()
                solution_data = solve_response.json()
                
                if 'error' in solution_data:
                    print(f"Error solving problem: {solution_data['error']}")
                    continue
                    
            except requests.exceptions.RequestException as e:
                print(f"Solve request error: {str(e)}")
                print(f"Response content: {solve_response.text if solve_response else 'No response'}")
                continue
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {str(e)}")
                print(f"Raw response: {solve_response.text}")
                continue
                
            print(f"\nGenerated solution with confidence: {solution_data['confidence']:.2f}")
            print(f"Number of sub-questions: {len(solution_data.get('sub_questions', []))}")
            
            # Simulate user ratings
            ratings = simulate_user_rating(
                solution_data['solution'],
                problem['solution']
            )
            
            # Submit feedback with complete data
            feedback_data = {
                'problem': problem['problem'],
                'solution': solution_data['solution'],
                'system_confidence': float(solution_data['confidence']),
                'processing_time': float(solution_data.get('processing_time', 0)),
                'sub_questions': solution_data.get('sub_questions', []),
                'rating': float(ratings['rating']),
                'clarity': float(ratings['clarity']),
                'helpfulness': float(ratings['helpfulness']),
                'confidence_accuracy': float(ratings['confidence_accuracy']),
                'comments': ''
            }
            
            # Then submit feedback
            try:
                feedback_response = requests.post(
                    f"{base_url}/feedback",
                    json=feedback_data,
                    headers={'Content-Type': 'application/json'}
                )
                feedback_response.raise_for_status()
                
                if feedback_response.status_code == 200:
                    results.append({
                        'problem': problem['problem'],
                        'subject': problem.get('subject', 'unknown'),
                        'solution': solution_data['solution'],
                        'system_confidence': float(solution_data['confidence']),
                        'num_sub_questions': len(solution_data.get('sub_questions', [])),
                        'processing_time': float(solution_data.get('processing_time', 0)),
                        'user_rating': float(ratings['rating']),
                        'clarity_rating': float(ratings['clarity']),
                        'helpfulness_rating': float(ratings['helpfulness']),
                        'confidence_accuracy': float(ratings['confidence_accuracy'])
                    })
                    print("Feedback submitted successfully")
                else:
                    print(f"Error submitting feedback: {feedback_response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Feedback request error: {str(e)}")
                print(f"Response content: {feedback_response.text if feedback_response else 'No response'}")
                continue
                
        except Exception as e:
            print(f"Error processing problem: {str(e)}")
            continue
            
        time.sleep(1)  # Brief pause between problems
        
    if results:
        # Save experiment results
        results_df = pd.DataFrame(results)
        results_df.to_csv('experiments/results/test_interactions.csv', index=False)
        
        print("\nExperiment Results Summary:")
        print(f"Total problems processed: {len(results)}")
        if 'system_confidence' in results_df.columns:
            print(f"Average confidence: {results_df['system_confidence'].mean():.2f}")
        if 'user_rating' in results_df.columns:
            print(f"Average user rating: {results_df['user_rating'].mean():.2f}")
        if 'clarity_rating' in results_df.columns:
            print(f"Average clarity: {results_df['clarity_rating'].mean():.2f}")
        if 'helpfulness_rating' in results_df.columns:
            print(f"Average helpfulness: {results_df['helpfulness_rating'].mean():.2f}")
        
        # Get final metrics
        try:
            metrics_response = requests.get(f"{base_url}/metrics")
            if metrics_response.status_code == 200:
                metrics = metrics_response.json()
                print("\nFinal Metrics:")
                print(json.dumps(metrics, indent=2))
            else:
                print(f"Error fetching metrics: {metrics_response.status_code}")
        except Exception as e:
            print(f"Error fetching final metrics: {str(e)}")
    else:
        print("No results collected during experiment run.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://localhost:5000',
                      help='Base URL for the Flask server')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    args = parser.parse_args()
    
    run_experiments(base_url=args.base_url, debug=args.debug)
