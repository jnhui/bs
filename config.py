"""
Configuration settings for the Socratic Math system.
"""

# Model configuration
MODEL_NAME = "gpt2"  # Will be updated based on requirements
CONFIDENCE_THRESHOLD = 0.7

# Confidence calculation weights
CONFIDENCE_WEIGHTS = {
    'perplexity': 0.3,    # Weight for model perplexity score
    'coherence': 0.3,     # Weight for mathematical term coherence
    'reasoning': 0.2,     # Weight for step-by-step reasoning presence
    'consistency': 0.2    # Weight for answer consistency across samples
}

# Sampling configuration
NUM_ANSWER_SAMPLES = 5    # Number of samples for consistency check

# Data configuration
DATA_DIR = "data/"
CACHE_DIR = "cache/"

# Training configuration
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
