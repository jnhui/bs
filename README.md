# Socratic Math Questioning System

A Python-based implementation of recursive Socratic questioning for mathematical problem solving, using the MATH dataset.

## Project Structure

```
socratic_math/
├── src/
│   ├── __init__.py
│   ├── socratic_questioner.py    # Main questioning system
│   ├── confidence_calculator.py  # Novel confidence calculation
│   └── data_loader.py           # MATH dataset handling
├── tests/
│   └── test_socratic_questioner.py
└── config.py                    # Configuration settings
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install transformers torch pandas numpy pytest tqdm matplotlib scikit-learn
```

## Environment

- Python 3.12
- Key dependencies:
  - transformers: For language model integration
  - torch: Deep learning framework
  - pandas: Data manipulation
  - numpy: Numerical operations
  - pytest: Testing framework
  - scikit-learn: Machine learning utilities

## Development Status

Initial setup complete. Next steps:
1. Implement novel confidence calculation method
2. Develop recursive question decomposition
3. Integrate with MATH dataset
4. Implement testing and validation
