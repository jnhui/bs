"""
Module for loading and processing the MATH dataset.
Handles both training and test sets across different mathematical subjects.
"""

import json
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

class MathDataLoader:
    """
    Loader for the MATH dataset that handles both training and test sets.
    Supports sampling and filtering by subject area and difficulty level.
    """
    
    SUBJECTS = [
        'algebra', 'counting_and_probability', 'geometry',
        'intermediate_algebra', 'number_theory', 'prealgebra',
        'precalculus'
    ]
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Path to the MATH dataset directory
        """
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / 'train'
        self.test_dir = self.data_dir / 'test'
        
    def load_dataset(self, split: str = 'train', subjects: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load the MATH dataset.
        
        Args:
            split (str): Either 'train' or 'test'
            subjects (List[str], optional): List of subject areas to load
            
        Returns:
            pd.DataFrame: DataFrame containing problems and solutions
        """
        if split not in ['train', 'test']:
            raise ValueError("Split must be either 'train' or 'test'")
            
        base_dir = self.train_dir if split == 'train' else self.test_dir
        subjects = subjects or self.SUBJECTS
        
        data = []
        for subject in tqdm(subjects, desc=f"Loading {split} data"):
            subject_dir = base_dir / subject
            if not subject_dir.exists():
                continue
                
            for file_path in subject_dir.glob('*.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        problem_data = json.load(f)
                    problem_data['subject'] = subject
                    problem_data['id'] = file_path.stem
                    data.append(problem_data)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
                    
        return pd.DataFrame(data)
        
    def sample_problems(self, n: int = 10, split: str = 'train',
                       subjects: Optional[List[str]] = None,
                       level: Optional[str] = None) -> pd.DataFrame:
        """
        Sample a random subset of problems.
        
        Args:
            n (int): Number of problems to sample
            split (str): Either 'train' or 'test'
            subjects (List[str], optional): List of subject areas to sample from
            level (str, optional): Specific difficulty level to sample
            
        Returns:
            pd.DataFrame: DataFrame containing sampled problems
        """
        df = self.load_dataset(split=split, subjects=subjects)
        
        if level:
            df = df[df['level'] == level]
            
        if len(df) < n:
            print(f"Warning: Requested {n} samples but only {len(df)} available")
            return df
            
        return df.sample(n=n, random_state=42)
        
    def preprocess_problem(self, problem: str) -> str:
        """
        Preprocess a math problem for the model.
        Handles LaTeX formatting and special characters.
        
        Args:
            problem (str): Raw problem text
            
        Returns:
            str: Preprocessed problem text
        """
        # Remove unnecessary whitespace
        problem = ' '.join(problem.split())
        
        # Ensure LaTeX delimiters are properly spaced
        problem = problem.replace('$', ' $ ')
        problem = ' '.join(problem.split())
        
        return problem
        
    def get_subject_distribution(self, split: str = 'train') -> pd.Series:
        """
        Get the distribution of problems across subjects.
        
        Args:
            split (str): Either 'train' or 'test'
            
        Returns:
            pd.Series: Count of problems per subject
        """
        df = self.load_dataset(split=split)
        return df['subject'].value_counts()
