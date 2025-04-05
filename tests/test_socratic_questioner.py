"""
Tests for the Socratic questioning system.
"""

import pytest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.socratic_questioner import SocraticQuestioner
from src.confidence_calculator import ConfidenceCalculator

@pytest.fixture(scope="module")
def questioner(model_and_tokenizer):
    """Create shared questioner instance for tests."""
    model, tokenizer = model_and_tokenizer
    return SocraticQuestioner(model_name="gpt2", confidence_threshold=0.7)

def test_basic_decomposition(questioner):
    """Test basic question decomposition functionality."""
    question = "What is 3 + 4?"
    sub_questions = questioner.decompose_question(question)
    
    assert len(sub_questions) > 0, "Should generate at least one sub-question"
    assert all(isinstance(q, str) for q in sub_questions), "All sub-questions should be strings"
    assert all(len(q.strip()) > 0 for q in sub_questions), "No empty sub-questions"

def test_equation_decomposition(questioner):
    """Test equation-specific decomposition."""
    question = "Solve: x + 5 = 12"
    sub_questions = questioner.decompose_question(question)
    
    assert len(sub_questions) >= 2, "Should break equation into multiple steps"
    assert any("solve" in q.lower() or "equation" in q.lower() for q in sub_questions), "Should analyze equation"

@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Shared model and tokenizer for tests."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer

def test_confidence_components(model_and_tokenizer):
    """Test individual confidence calculation components."""
    model, tokenizer = model_and_tokenizer
    calculator = ConfidenceCalculator(model=model, tokenizer=tokenizer)
    
    # Test mathematical coherence
    assert calculator.check_mathematical_coherence("2 + 2 = 4") > 0.5
    assert calculator.check_mathematical_coherence("(1 + 2) * 3 = 9") > 0.7
    assert calculator.check_mathematical_coherence("(1 + 2 * (3") < 0.3
    
    # Test step reasoning detection
    assert calculator.check_step_reasoning("First, we add 2. Then multiply by 3.") > 0.5
    assert calculator.check_step_reasoning("x = 5") < 0.3

def test_answer_consistency(model_and_tokenizer):
    """Test answer consistency calculation."""
    model, tokenizer = model_and_tokenizer
    calculator = ConfidenceCalculator(model=model, tokenizer=tokenizer)
    
    # Test answer consistency
    samples = ["The answer is 42", "The result is 42", "We get 42"]
    assert calculator.calculate_answer_consistency(samples) > 0.5

def test_overall_confidence(model_and_tokenizer):
    """Test overall confidence calculation."""
    model, tokenizer = model_and_tokenizer
    calculator = ConfidenceCalculator(model=model, tokenizer=tokenizer)
    
    samples = ["The answer is 42", "The result is 42", "We get 42"]
    mock_outputs = {
        'samples': samples,
        'loss': torch.tensor(2.0)
    }
    confidence = calculator.calculate_confidence(
        question="What is 6 * 7?",
        answer="First multiply: 6 * 7 = 42",
        model_outputs=mock_outputs
    )
    assert 0 <= confidence <= 1, "Confidence should be between 0 and 1"

def test_weighted_confidence():
    """Test weighted confidence combination."""
    sub_confidences = [0.8, 0.9, 0.7]
    weights = np.array(sub_confidences) / sum(sub_confidences)
    weighted_conf = sum(weights * sub_confidences)
    assert 0.7 < weighted_conf <= 1.0, "Weighted confidence should be reasonable"
