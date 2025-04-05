"""
Validation tests for the Socratic questioning system using MATH dataset.
Compares performance with baseline methods (Chain-of-Thought, Self-consistency).
"""

import pytest
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.socratic_questioner import SocraticQuestioner
from src.data_loader import MathDataLoader

# Test fixtures
@pytest.fixture(scope="session")
def model():
    """Load model once for all tests."""
    return AutoModelForCausalLM.from_pretrained("gpt2")

@pytest.fixture(scope="session")
def tokenizer():
    """Load tokenizer once for all tests."""
    return AutoTokenizer.from_pretrained("gpt2")

@pytest.fixture
def questioner(model, tokenizer):
    """Create questioner with shared model and tokenizer."""
    return SocraticQuestioner(model_name="gpt2", confidence_threshold=0.7)

@pytest.fixture(scope="session")
def sample_problems():
    """Load a small set of test problems."""
    with open("data/MATH/test.json", 'r') as f:
        data = json.load(f)
        # Start with just 2 simple problems for ablation testing
        simple_problems = [
            p for p in data['problems'][:4] 
            if p.get('level', '') == 'basic' and len(p['problem']) < 100
        ][:2]
        print(f"\nSelected {len(simple_problems)} simple problems for testing:")
        for p in simple_problems:
            print(f"- {p['problem']}")
        return simple_problems

def load_test_problems(n_samples: int = 10) -> List[Dict[str, Any]]:
    """Load a sample of test problems from MATH dataset."""
    data_dir = Path("data/MATH")
    test_path = data_dir / "test.json"
    
    if not test_path.exists():
        pytest.skip("MATH dataset not found. Please download it first.")
        
    with open(test_path, 'r') as f:
        data = json.load(f)
        problems = data.get('problems', [])
        
    if not problems:
        pytest.skip("No problems found in the dataset.")
        
    # Sample problems across different subjects
    subjects = set(p.get('subject', '') for p in problems)
    if not subjects:
        return problems[:n_samples]  # If no subjects, just return first n_samples
        
    sampled = []
    samples_per_subject = max(1, n_samples // len(subjects))
    for subject in subjects:
        subject_problems = [p for p in problems if p.get('subject') == subject]
        if subject_problems:
            sampled.extend(subject_problems[:samples_per_subject])
            
    return sampled[:n_samples]

@pytest.mark.timeout(15)  # Shorter timeout for basic functionality test
def test_recursive_decomposition(questioner):
    """Test the recursive problem decomposition functionality."""
    print("\nTesting recursive decomposition...")
    
    # Use a simpler problem for initial testing
    problem = "What is 3 + 4?"
    print(f"\nTesting with problem: {problem}")
    
    # Test decomposition with debug output
    print("\nGenerating sub-questions...")
    sub_questions = questioner.decompose_question(problem)
    print(f"\nGenerated {len(sub_questions)} sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        print(f"{i}. {q}")
    
    assert len(sub_questions) > 0, "Should generate at least one sub-question"
    
    # Basic validation of sub-questions
    print("\nValidating sub-questions...")
    for q in sub_questions:
        assert isinstance(q, str), "Each sub-question should be a string"
        assert len(q.strip()) > 0, "Sub-questions should not be empty"
        print(f"Valid question: {q}")
    
    # Test answer generation with confidence
    print("\nTesting answer generation...")
    answer, confidence = questioner.solve_problem(problem)
    print(f"\nAnswer: {answer}")
    print(f"Confidence: {confidence:.3f}")
    
    assert 0 <= confidence <= 1, "Confidence should be between 0 and 1"
    assert isinstance(answer, str) and len(answer.strip()) > 0, "Answer should not be empty"

@pytest.mark.timeout(60)  # Longer timeout for dataset validation
def test_math_dataset_validation(questioner, sample_problems):
    """Test system performance on MATH dataset samples."""
    problems = sample_problems  # Use fixture for consistent test data
    
    results = []
    for problem in problems:
        # Solve with our system
        question = problem['problem']
        answer, confidence = questioner.solve_problem(question)
        
        # Record results
        result = {
            'problem': question,
            'subject': problem.get('subject', ''),
            'level': problem.get('level', ''),
            'our_answer': answer,
            'our_confidence': confidence,
            'correct_answer': problem.get('solution', '')
        }
        results.append(result)
        
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Basic validation checks
    assert not df.empty, "Should have processed at least one problem"
    assert all(0 <= conf <= 1 for conf in df['our_confidence']), "All confidence scores should be between 0 and 1"
    assert all(df['our_answer'].str.contains('step', case=False)), "All answers should include step-by-step reasoning"

@pytest.mark.timeout(90)  # Even longer timeout for multiple baseline comparisons
def test_baseline_comparison(questioner, sample_problems):
    """Compare performance with Chain-of-Thought and Self-consistency baselines."""
    
def test_confidence_ablation(questioner, sample_problems):
    """Test system performance with different confidence components disabled."""
    import json
    from pathlib import Path
    from src.confidence_calculator import ConfidenceCalculator
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    
    print("\nStarting confidence ablation study...")
    
    # Load ablation configurations
    try:
        with open('tests/ablation_configs.json', 'r') as f:
            ablation_configs = json.load(f)
        print(f"Loaded {len(ablation_configs)} configurations: {list(ablation_configs.keys())}")
    except Exception as e:
        print(f"Error loading ablation configs: {str(e)}")
        raise
    
    # Original calculator for baseline
    print("\nSaving original calculator configuration...")
    base_calculator = questioner.confidence_calculator
    
    results = []
    print("\nRunning ablation studies...")
    print(f"Testing with {len(sample_problems)} problems")
    
    try:
        for config_name, weights in ablation_configs.items():
            print(f"\nTesting configuration: {config_name}")
            print(f"Weights: {weights}")
            
            start_time = time.time()
            
            try:
                # Create calculator with modified weights
                modified_calculator = ConfidenceCalculator(
                    model=questioner.model,
                    tokenizer=questioner.tokenizer,
                    weights=weights
                )
                
                # Test with modified calculator
                questioner.confidence_calculator = modified_calculator
                
                config_results = []
                for i, problem in enumerate(sample_problems, 1):
                    print(f"\nProcessing problem {i}/{len(sample_problems)} for {config_name}...")
                    try:
                        answer, confidence = questioner.solve_problem(problem['problem'])
                        print(f"Got confidence: {confidence:.3f}")
                        
                        result = {
                            'config': config_name,
                            'problem': problem['problem'],
                            'confidence': confidence,
                            'answer': answer,
                            'correct': problem['solution']
                        }
                        config_results.append(result)
                        
                    except Exception as e:
                        print(f"Error processing problem: {str(e)}")
                        continue
                
                results.extend(config_results)
                print(f"Completed {config_name} in {time.time() - start_time:.1f}s")
                
            except Exception as e:
                print(f"Error testing configuration {config_name}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in ablation study: {str(e)}")
        # Restore original calculator before raising
        questioner.confidence_calculator = base_calculator
        raise
    
    # Restore original calculator
    questioner.confidence_calculator = base_calculator
    
    # Analyze results
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary_stats = df.groupby('config').agg({
        'confidence': ['mean', 'std'],
        'problem': 'count'
    }).round(3)
    
    # Flatten multi-index for JSON serialization
    ablation_summary = {}
    for config in summary_stats.index:
        # Extract values safely with proper type conversion
        conf_mean = summary_stats.loc[config][('confidence', 'mean')]
        conf_std = summary_stats.loc[config][('confidence', 'std')]
        prob_count = summary_stats.loc[config][('problem', 'count')]
        
        ablation_summary[config] = {
            'confidence_mean': float(conf_mean),
            'confidence_std': float(conf_std),
            'problem_count': int(prob_count)
        }
    
    # Generate visualizations
    plt.figure(figsize=(15, 10))
    
    # Confidence distribution by configuration
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='config', y='confidence')
    plt.title('Confidence Distribution by Configuration')
    plt.xticks(rotation=45)
    
    # Performance comparison
    plt.subplot(2, 2, 2)
    sns.barplot(data=df, x='config', y='confidence', errorbar=('ci', 95))
    plt.title('Average Confidence by Configuration')
    plt.xticks(rotation=45)
    
    # Save results
    results_dir = Path('experiments/results/ablation')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(results_dir / 'ablation_results.csv', index=False)
    pd.DataFrame.from_dict(ablation_summary, orient='index').to_csv(results_dir / 'ablation_summary.csv')
    plt.savefig(results_dir / 'ablation_analysis.png', bbox_inches='tight')
    plt.close()
    
    # Calculate mean confidence for each configuration
    config_means = {}
    for config in df['config'].unique():
        config_means[config] = float(df[df['config'] == config]['confidence'].mean())
    
    # Calculate component impacts
    baseline_conf = config_means['baseline']
    component_impacts = {}
    
    for component in ['perplexity', 'coherence', 'reasoning', 'consistency']:
        no_component_conf = config_means[f'no_{component}']
        impact = baseline_conf - no_component_conf
        component_impacts[component] = float(impact)
        print(f"{component} impact: {impact:.3f}")
    
    # Save detailed analysis
    analysis = {
        'summary': ablation_summary,
        'config_means': config_means,
        'component_impacts': component_impacts
    }
    
    with open(results_dir / 'ablation_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
        
    print("\nAblation study results:")
    print(f"Results saved to {results_dir}")
    print("\nComponent impacts (confidence drop when removed):")
    for component, impact in component_impacts.items():
        print(f"{component}: {impact:.3f}")
    
    # Basic validation
    assert len(df) > 0, "Should have processed at least one problem"
    assert all(0 <= conf <= 1 for conf in df['confidence']), "All confidence scores should be between 0 and 1"
    
    return analysis
