"""
Script to run comprehensive validation of the Socratic questioning system.
Compares performance with baseline methods and generates visualization reports.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from src.socratic_questioner import SocraticQuestioner
from src.visualization import (
    plot_confidence_distribution,
    analyze_decomposition_decisions,
    plot_decomposition_tree
)

def load_problems(split: str = 'test') -> List[Dict[str, Any]]:
    """Load problems from the MATH dataset."""
    data_path = Path(f"data/MATH/{split}.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data['problems']

def evaluate_solution(predicted: str, actual: str) -> float:
    """
    Calculate solution similarity score.
    Uses both exact match for numerical answers and semantic similarity.
    """
    import re
    
    # Extract numerical values
    def extract_numbers(text: str) -> List[float]:
        return [float(n) for n in re.findall(r'-?\d*\.?\d+', text)]
    
    pred_nums = extract_numbers(predicted)
    actual_nums = extract_numbers(actual)
    
    # Check numerical match if present
    if pred_nums and actual_nums:
        num_match = any(abs(p - a) < 0.01 for p in pred_nums for a in actual_nums)
    else:
        num_match = False
        
    # Calculate text similarity
    from Levenshtein import ratio
    text_sim = ratio(predicted.lower(), actual.lower())
    
    # Combine scores
    return 0.7 * float(num_match) + 0.3 * text_sim

def run_validation(n_problems: int = 3, timeout: int = 60):
    """
    Run comprehensive validation and generate reports.
    
    Args:
        n_problems (int): Number of problems to test (default: 3 for initial validation)
        timeout (int): Maximum seconds per problem (default: 60)
    """
    import time
    start_time = time.time()
    
    print("\nStarting validation with following parameters:")
    print(f"Number of problems: {n_problems}")
    print(f"Timeout per problem: {timeout} seconds")
    
    def log_timing(start_time, step_name):
        elapsed = time.time() - start_time
        print(f"[TIMING] {step_name}: {elapsed:.2f} seconds")
    import signal
    from contextlib import contextmanager
    
    @contextmanager
    def timeout_handler(seconds):
        def signal_handler(signum, frame):
            raise TimeoutError("Processing took too long")
        
        # Set the timeout handler
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            # Disable the alarm
            signal.alarm(0)
    questioner = SocraticQuestioner(confidence_threshold=0.7)
    all_problems = load_problems('test')
    problems = all_problems[:n_problems]  # Start with small test set
    
    results = []
    for i, problem in enumerate(problems, 1):
        problem_start = time.time()
        print(f"\nProcessing problem {i}/{n_problems}:")
        print(f"Subject: {problem.get('subject', 'unknown')}, Level: {problem.get('level', 'unknown')}")
        print(f"Question: {problem['problem']}")
        
        # Initialize tracking variables
        sub_questions = []
        sub_answers = []
        sub_confidences = []
        
        try:
            with timeout_handler(timeout):
                # Core functionality validation
                print("\n1. Testing direct solution...")
                solution_start = time.time()
                answer, confidence = questioner.solve_problem(problem['problem'])
                print(f"Initial confidence: {confidence:.2f}")
                print(f"Initial answer: {answer[:200]}...")
                log_timing(solution_start, "Direct solution")
                
                print("\n2. Testing question decomposition...")
                decomp_start = time.time()
                sub_questions = questioner.decompose_question(problem['problem'])
                print(f"Generated {len(sub_questions)} sub-questions:")
                for j, sq in enumerate(sub_questions, 1):
                    print(f"  {j}. {sq}")
                log_timing(decomp_start, "Question decomposition")
                
                # Process sub-questions regardless of confidence
                # This helps us analyze decomposition effectiveness
                if sub_questions:
                    print("\n3. Processing sub-questions...")
                    for j, sq in enumerate(sub_questions, 1):
                        sq_start = time.time()
                        sq_answer, sq_conf = questioner.answer_question(sq)
                        print(f"  Sub-Q {j} confidence: {sq_conf:.2f}")
                        print(f"  Sub-Q {j} answer: {sq_answer[:100]}...")
                        sub_answers.append(sq_answer)
                        sub_confidences.append(sq_conf)
                        log_timing(sq_start, f"Sub-question {j}")
                    
                    # Calculate sub-question statistics
                    avg_sub_conf = float(sum(sub_confidences)) / len(sub_confidences) if sub_confidences else None
                    max_sub_conf = float(max(sub_confidences)) if sub_confidences else None
                    min_sub_conf = float(min(sub_confidences)) if sub_confidences else None
                    
                    # Record sub-question performance
                    result = {
                        'problem': problem['problem'],
                        'subject': problem.get('subject', 'unknown'),
                        'level': problem.get('level', 'unknown'),
                        'our_answer': answer,
                        'our_confidence': float(confidence),
                        'num_subquestions': len(sub_questions),
                        'avg_sub_confidence': avg_sub_conf,
                        'max_sub_confidence': max_sub_conf,
                        'min_sub_confidence': min_sub_conf,
                        'solution': problem['solution']
                    }
                else:
                    # Direct solution was confident enough
                    result = {
                        'problem': problem['problem'],
                        'subject': problem.get('subject', 'unknown'),
                        'level': problem.get('level', 'unknown'),
                        'our_answer': answer,
                        'our_confidence': float(confidence),
                        'num_subquestions': 0,
                        'avg_sub_confidence': None,
                        'max_sub_confidence': None,
                        'min_sub_confidence': None,
                        'solution': problem['solution']
                    }
                
                # Evaluate solution
                our_score = evaluate_solution(answer, problem['solution'])
                result['our_score'] = our_score
                
                # Track decomposition effectiveness
                if len(sub_questions) > 0:
                    sub_question_details = []
                    for sq, sq_ans, sq_conf in zip(sub_questions, sub_answers, sub_confidences):
                        sq_score = evaluate_solution(sq_ans, problem['solution'])
                        sub_question_details.append({
                            'question': sq,
                            'answer': sq_ans,
                            'confidence': float(sq_conf),
                            'score': float(sq_score)
                        })
                    result['sub_question_details'] = sub_question_details
                    
                    # Calculate improvement from decomposition
                    if sub_question_details:
                        avg_sq_score = sum(sq['score'] for sq in sub_question_details) / len(sub_question_details)
                        result['decomposition_improvement'] = float(our_score - avg_sq_score)
                
                results.append(result)
                log_timing(problem_start, f"Total problem {i}")
                
                # Print evaluation results
                print(f"\nEvaluation Results:")
                print(f"Score: {our_score:.2f}")
                print(f"Final confidence: {confidence:.2f}")
                print(f"Number of sub-questions: {len(sub_questions)}")
                
                if len(sub_questions) > 0:
                    print("\nSub-question Performance:")
                    for i, sq in enumerate(sub_question_details, 1):
                        print(f"  {i}. Confidence: {sq['confidence']:.2f}, Score: {sq['score']:.2f}")
                    if 'decomposition_improvement' in result:
                        print(f"\nDecomposition Improvement: {result['decomposition_improvement']:.3f}")
                
        except TimeoutError:
            print(f"\n[ERROR] Timeout after {timeout} seconds")
            continue
        except Exception as e:
            print(f"\n[ERROR] Error: {str(e)}")
            continue
        print("\n" + "="*50)
        
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Generate performance visualizations
    plt.figure(figsize=(12, 6))
    
    # Plot score vs confidence
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='our_confidence', y='our_score')
    plt.title('Score vs Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Score')
    
    # Plot confidence distribution
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='our_confidence', bins=10)
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence Score')
    
    # Plot score distribution
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='our_score', bins=10)
    plt.title('Score Distribution')
    plt.xlabel('Score')
    
    # Plot performance by subject
    plt.subplot(2, 2, 4)
    subject_performance = df.groupby('subject')['our_score'].mean()
    subject_performance.plot(kind='bar')
    plt.title('Score by Subject')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png')
    plt.close()
    
    # Plot sub-question analysis
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, y='num_subquestions')
    plt.title('Distribution of Sub-questions')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df.dropna(subset=['avg_sub_confidence']), y='avg_sub_confidence')
    plt.title('Sub-question Confidence')
    
    plt.tight_layout()
    plt.savefig('decomposition_analysis.png')
    plt.close()
    
    # Generate summary statistics with proper type conversion
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
        
    # Calculate decomposition effectiveness
    decomposition_improvements = []
    sub_question_scores = []
    sub_question_confidences = []
    
    for _, row in df.iterrows():
        if 'sub_question_details' in row and row['sub_question_details']:
            for sq in row['sub_question_details']:
                sub_question_scores.append(sq['score'])
                sub_question_confidences.append(sq['confidence'])
            if 'decomposition_improvement' in row:
                decomposition_improvements.append(row['decomposition_improvement'])
    
    summary = {
        'overall_performance': {
            'mean_score': convert_numpy(df['our_score'].mean()),
            'std_score': convert_numpy(df['our_score'].std()),
            'min_score': convert_numpy(df['our_score'].min()),
            'max_score': convert_numpy(df['our_score'].max())
        },
        'confidence_stats': {
            'mean': convert_numpy(df['our_confidence'].mean()),
            'std': convert_numpy(df['our_confidence'].std()),
            'correlation_with_score': convert_numpy(df['our_confidence'].corr(df['our_score'])),
            'min_confidence': convert_numpy(df['our_confidence'].min()),
            'max_confidence': convert_numpy(df['our_confidence'].max())
        },
        'decomposition_stats': {
            'avg_subquestions': convert_numpy(df['num_subquestions'].mean()),
            'max_subquestions': convert_numpy(df['num_subquestions'].max()),
            'min_subquestions': convert_numpy(df['num_subquestions'].min()),
            'problems_decomposed': convert_numpy((df['num_subquestions'] > 0).sum()),
            'sub_question_performance': {
                'mean_score': convert_numpy(np.mean(sub_question_scores)) if sub_question_scores else None,
                'mean_confidence': convert_numpy(np.mean(sub_question_confidences)) if sub_question_confidences else None,
                'avg_improvement': convert_numpy(np.mean(decomposition_improvements)) if decomposition_improvements else None
            }
        },
        'by_subject': {
            subject: {
                'score': convert_numpy(group['our_score'].mean()),
                'confidence': convert_numpy(group['our_confidence'].mean())
            }
            for subject, group in df.groupby('subject')
        }
    }
    
    try:
        # Save results with error handling
        df.to_csv('validation_results.csv', index=False)
        with open('validation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Generate additional visualizations
        plot_confidence_distribution(questioner.decomposition_history, 'confidence_distribution.png')
        analyze_decomposition_decisions(questioner.decomposition_history).to_csv('decomposition_analysis.csv')
        plot_decomposition_tree(questioner.decomposition_history, 'decomposition_tree.png')
        
        print("\nValidation Summary:")
        print(json.dumps(summary, indent=2))
        print("\nVisualization files generated:")
        print("- validation_results.csv")
        print("- validation_summary.json")
        print("- confidence_distribution.png")
        print("- decomposition_analysis.csv")
        print("- decomposition_tree.png")
        print("- performance_comparison.png")
        print("- subject_performance.png")
        
        return df, summary
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        # Return data even if saving fails
        return df, summary

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_problems', type=int, default=3,
                      help='Number of problems to test')
    parser.add_argument('--timeout', type=int, default=60,
                      help='Maximum seconds per problem')
    args = parser.parse_args()
    
    df, summary = run_validation(n_problems=args.n_problems, 
                               timeout=args.timeout)
    print("\nValidation Summary:")
    print(json.dumps(summary, indent=2))
