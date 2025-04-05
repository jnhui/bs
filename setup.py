from setuptools import setup, find_packages

setup(
    name="socratic_math",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pytest>=8.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.2",
        "networkx>=3.1",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "python-Levenshtein>=0.21.0"
    ],
    python_requires=">=3.8",
)
