"""
Module for calculating confidence scores in answers.
Implements a novel multi-factor confidence calculation method that considers:
1. Answer consistency across multiple samplings
2. Model perplexity and logit distributions
3. Mathematical term coherence
4. Step-by-step reasoning presence
"""

from typing import Dict, Any, List
import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

class ConfidenceCalculator:
    def __init__(self, model: PreTrainedModel = None, tokenizer: PreTrainedTokenizer = None,
                 weights: Dict[str, float] | None = None):
        self.confidence_threshold = 0.7
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = 5  # Number of answer samples to generate
        
        # Default weights for confidence components
        self._weights = {
            'perplexity': 0.3,
            'coherence': 0.3,
            'reasoning': 0.2,
            'consistency': 0.2
        }
        
        # Update weights if provided
        if weights is not None:
            self.set_weights(weights)
            
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update confidence calculation weights."""
        # Validate weights
        required_components = {'perplexity', 'coherence', 'reasoning', 'consistency'}
        if not all(k in weights for k in required_components):
            raise ValueError(f"Missing required weight components: {required_components - set(weights.keys())}")
            
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            self._weights = {k: v/total for k, v in weights.items()}
        else:
            raise ValueError("Weights must sum to a positive value")
        
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate the perplexity of the generated answer.
        Returns a normalized score between 0 and 1.
        """
        if not self.model or not self.tokenizer:
            return 0.5  # Default score if models aren't available
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Calculate loss manually if not provided
                if outputs.loss is None:
                    labels = inputs.input_ids
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                  shift_labels.view(-1))
                else:
                    loss = outputs.loss
                    
                # Clip loss to prevent overflow
                loss = torch.clamp(loss, -100, 100)
                perplexity = torch.exp(loss).item()
                # Normalize to 0-1 range with reasonable bounds
                score = 1.0 / (1.0 + min(perplexity, 1e6))  # Cap maximum perplexity
                return float(score)
                
        except Exception as e:
            print(f"Error calculating perplexity: {str(e)}")
            return 0.5  # Default score on error
        
    def check_mathematical_coherence(self, text: str) -> float:
        """
        Check for mathematical term coherence and completeness.
        Looks for matching brackets, equation balance, and mathematical symbols.
        """
        # Check for balanced brackets and equations
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        math_symbols = set(['=', '+', '-', '*', '/', '^', '√', '∫', '∑'])
        symbol_count = 0
        
        for char in text:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack or brackets[stack.pop()] != char:
                    return 0.0
            elif char in math_symbols:
                symbol_count += 1
                
        coherence_score = 1.0 if len(stack) == 0 else 0.0
        symbol_score = min(1.0, symbol_count / 5)  # Normalize symbol count
        return (coherence_score + symbol_score) / 2
        
    def check_step_reasoning(self, text: str) -> float:
        """
        Check if the answer contains step-by-step reasoning.
        Looks for numbered steps, transition words, and logical flow.
        """
        step_indicators = ['first', 'second', 'then', 'next', 'finally', 'step', 'therefore']
        found_indicators = sum(1 for indicator in step_indicators if indicator in text.lower())
        return min(1.0, found_indicators / 3)  # Normalize score
        
    def calculate_answer_consistency(self, samples: List[str]) -> float:
        """
        Calculate consistency score across multiple answer samples.
        Uses both Levenshtein ratio for text similarity and numerical comparison.
        Includes robust error handling for invalid inputs and numerical edge cases.
        """
        try:
            if not isinstance(samples, list) or len(samples) < 2:
                return 0.0
                
            # Filter out invalid samples
            valid_samples = [s for s in samples if isinstance(s, str) and s.strip()]
            if len(valid_samples) < 2:
                return 0.0
                
            from Levenshtein import ratio
            import re
            
            # Extract numerical values from answers with error handling
            def extract_numbers(text):
                try:
                    return [float(n) for n in re.findall(r'-?\d*\.?\d+', text)]
                except ValueError:
                    return []
                    
            numbers = [extract_numbers(s) for s in valid_samples]
            has_numbers = any(len(n) > 0 for n in numbers)
            
            # Calculate text similarity using Levenshtein ratio
            text_similarities = []
            for i in range(len(valid_samples)):
                for j in range(i + 1, len(valid_samples)):
                    try:
                        sim = ratio(valid_samples[i], valid_samples[j])
                        if 0 <= sim <= 1:  # Validate similarity score
                            text_similarities.append(sim)
                    except Exception as e:
                        print(f"Warning: Error calculating similarity: {str(e)}")
                        continue
                        
            if not text_similarities:
                return 0.0
                
            text_score = float(np.clip(np.mean(text_similarities), 0, 1))
            
            # If we found numbers, calculate numerical consistency
            if has_numbers:
                try:
                    # Flatten and remove duplicates, filter invalid numbers
                    all_numbers = [n for sublist in numbers for n in sublist 
                                 if isinstance(n, (int, float)) and not np.isnan(n)]
                    if not all_numbers:
                        return text_score
                        
                    if len(set(all_numbers)) == 1:
                        # Perfect numerical consistency
                        number_score = 1.0
                    else:
                        # Calculate coefficient of variation with bounds
                        number_std = float(np.clip(np.std(all_numbers), 0, float('inf')))
                        number_mean = float(np.mean(all_numbers))
                        if abs(number_mean) > 1e-10:  # Avoid division by very small numbers
                            number_score = 1.0 / (1.0 + abs(number_std / number_mean))
                        else:
                            number_score = 0.0
                            
                        number_score = float(np.clip(number_score, 0, 1))
                        
                    # Weight text and numerical scores
                    final_score = float(np.clip(0.4 * text_score + 0.6 * number_score, 0, 1))
                    return final_score
                except Exception as e:
                    print(f"Warning: Error in numerical consistency: {str(e)}")
                    return text_score
                    
            return text_score
            
        except Exception as e:
            print(f"Error in calculate_answer_consistency: {str(e)}")
            return 0.0
        
    def calculate_confidence(self, 
                           question: str, 
                           answer: str, 
                           model_outputs: Dict[str, Any]) -> float:
        """
        Calculate confidence score using a novel multi-factor method.
        Includes robust error handling and validation of inputs.
        
        Args:
            question (str): The original question
            answer (str): The generated answer
            model_outputs (Dict[str, Any]): Additional model outputs including samples
            
        Returns:
            float: Confidence score between 0 and 1, defaults to 0 on error
        """
        try:
            # Validate inputs
            if not isinstance(answer, str) or not answer.strip():
                return 0.0
                
            # Calculate individual confidence factors with validation
            try:
                perplexity_score = 1.0 / (1.0 + self.calculate_perplexity(answer))
                perplexity_score = float(np.clip(perplexity_score, 0, 1))
            except Exception as e:
                print(f"Warning: Error calculating perplexity: {str(e)}")
                perplexity_score = 0.0
                
            try:
                coherence_score = float(np.clip(self.check_mathematical_coherence(answer), 0, 1))
            except Exception as e:
                print(f"Warning: Error checking coherence: {str(e)}")
                coherence_score = 0.0
                
            try:
                reasoning_score = float(np.clip(self.check_step_reasoning(answer), 0, 1))
            except Exception as e:
                print(f"Warning: Error checking reasoning: {str(e)}")
                reasoning_score = 0.0
                
            # Generate multiple samples and check consistency
            try:
                samples = model_outputs.get('samples', [answer])
                consistency_score = float(np.clip(self.calculate_answer_consistency(samples), 0, 1))
            except Exception as e:
                print(f"Warning: Error calculating consistency: {str(e)}")
                consistency_score = 0.0
                
            # Calculate weighted confidence with validation
            scores = {
                'perplexity': perplexity_score,
                'coherence': coherence_score,
                'reasoning': reasoning_score,
                'consistency': consistency_score
            }
            
            # Validate all scores are in valid range
            for name, score in scores.items():
                if not (isinstance(score, (int, float)) and 0 <= score <= 1):
                    print(f"Warning: Invalid {name} score: {score}, using 0.0")
                    scores[name] = 0.0
                    
            confidence = sum(self._weights[k] * scores[k] for k in self._weights)
            
            # Ensure final confidence is valid
            return float(np.clip(confidence, 0, 1))
            
        except Exception as e:
            print(f"Error calculating confidence: {str(e)}")
            return 0.0  # Safe default on error
