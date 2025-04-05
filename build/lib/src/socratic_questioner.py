"""
Main module for the Socratic questioning system.
Implements recursive questioning and answer validation using divide-and-conquer approach.
"""

from typing import List, Dict, Any, Tuple, Optional
import torch
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from .confidence_calculator import ConfidenceCalculator

class SocraticQuestioner:
    def __init__(self, model_name: str = "gpt2", confidence_threshold: float = 0.7):
        """
        Initialize the Socratic questioning system.
        
        Args:
            model_name (str): Name of the pretrained model to use
            confidence_threshold (float): Base threshold for question decomposition (0-1)
            
        Example:
            >>> questioner = SocraticQuestioner("gpt2", confidence_threshold=0.75)
            >>> answer, confidence = questioner.solve_problem("What is 5 + 7?")
        """
        self.model_name = model_name
        self.base_threshold = confidence_threshold
        self.max_recursion_depth = 2  # Limit recursion for testing
        self.min_confidence = 0.4  # Absolute minimum confidence to accept
        self.decomposition_history = []  # Track decomposition decisions
        self.debug = True  # Enable debug logging
        self.seen_questions = {}  # Dict to track questions and their best confidence
        self.max_attempts = 3  # Maximum attempts to decompose a question
        self.attempt_count = {}  # Track decomposition attempts per question
        self.question_cache = {}  # Cache decomposed questions
        
        # Complexity patterns for threshold adjustment
        self.complexity_patterns = {
            'basic': r'\b(what|find|calculate|solve)\b.*?\b(\d+[\s]*[\+\-\*/]\s*\d+)\b',
            'medium': r'\b(equation|system|expression)\b',
            'complex': r'\b(prove|theorem|lemma|optimize)\b'
        }
        
        # Threshold adjustment factors
        self.threshold_decay = 0.8  # Decay factor for depth-based threshold
        self.complexity_factors = {
            'basic': 0.8,  # Lower threshold for basic questions
            'medium': 1.0,  # No adjustment for medium questions
            'complex': 1.2  # Higher threshold for complex questions
        }
        
        # Default generation configurations
        self.subq_generation_config = {
            'max_new_tokens': 48,  # Reduced for faster generation
            'min_length': 8,   # Slightly reduced minimum
            'num_return_sequences': 1,
            'do_sample': True,
            'temperature': 0.6,  # Lower temperature for more focused output
            'top_p': 0.85,
            'top_k': 20,  # More restrictive filtering
            'repetition_penalty': 1.2,
            'num_beams': 1,  # Remove beam search for speed
            'no_repeat_ngram_size': 2  # Prevent repetition
        }
        
        self.answer_generation_config = {
            'max_new_tokens': 128,  # Reduced for faster generation
            'num_return_sequences': 2,  # Reduced samples
            'do_sample': True,
            'temperature': 0.6,  # Lower temperature
            'top_p': 0.85,
            'repetition_penalty': 1.2  # Prevent repetitive outputs
        }
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.confidence_calculator = ConfidenceCalculator(self.model, self.tokenizer)
            
            # Set pad token IDs after tokenizer initialization
            self.subq_generation_config['pad_token_id'] = self.tokenizer.eos_token_id
            self.answer_generation_config['pad_token_id'] = self.tokenizer.eos_token_id
            
            if self.debug:
                print(f"[DEBUG] Initialized {model_name} with confidence threshold {confidence_threshold}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")
            
    def _ensure_initialized(self):
        """Ensure model, tokenizer, and calculator are initialized."""
        if not all([self.model, self.tokenizer, self.confidence_calculator]):
            raise RuntimeError("Models not properly initialized. Create a new instance.")
        
    def generate_subquestions(self, question: str) -> List[str]:
        """
        Generate sub-questions using the language model.
        Uses a specific prompt template to guide question decomposition.
        
        Args:
            question (str): The math problem to decompose
            
        Returns:
            List[str]: List of sub-questions
            
        Example:
            >>> questioner = SocraticQuestioner()
            >>> subs = questioner.generate_subquestions("Solve 3x + 5 = 14")
            >>> print(subs)
            ['What is the coefficient of x?', 'What number is being added to 3x?', ...]
            
        Raises:
            RuntimeError: If models are not properly initialized
        """
        self._ensure_initialized()
        prompt = f"""Break down this specific math problem into sub-questions that directly help solve it:
Original Problem: {question}

Rules for sub-questions:
1. Each sub-question must use terms/numbers/variables from the original problem
2. Each sub-question must be a necessary step toward solving the original problem
3. Sub-questions must follow standard mathematical solution steps
4. No general math questions - stay focused on this specific problem

Examples by subject:

For Algebra (e.g., "Solve (x + 2)(x - 3) = 0"):
1. What are the two factors? [(x + 2) and (x - 3)]
2. Using zero product property, what equations do we get? [(x + 2) = 0 or (x - 3) = 0]
3. How do we solve x + 2 = 0? [Subtract 2: x = -2]
4. How do we solve x - 3 = 0? [Add 3: x = 3]

For Geometry (e.g., "Find the area of a circle with radius 5"):
1. What is the formula for circle area? [A = πr²]
2. What is the value of radius (r)? [r = 5]
3. What is r² in this case? [5² = 25]
4. How do we calculate the final area? [A = 25π]

For Arithmetic (e.g., "Calculate 23 × 17"):
1. Can we break down 23 into simpler parts? [23 = 20 + 3]
2. How do we multiply 20 by 17? [20 × 17 = 340]
3. How do we multiply 3 by 17? [3 × 17 = 51]
4. What is the sum of the partial products? [340 + 51 = 391]

Now break down {question} into similar focused sub-questions that directly contribute to its solution:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        # Configure generation parameters for faster, more focused output
        if self.debug:
            print(f"[DEBUG] Generating sub-questions for: {question}")
            print(f"[DEBUG] Using sub-question generation config: {self.subq_generation_config}")
            
        # Use stored generation config
        generation_config = self.subq_generation_config.copy()
        
        outputs = self.model.generate(
            **inputs,
            **generation_config
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract sub-questions from generated text, focusing on actual mathematical questions
        sub_questions = []
        for line in generated_text.split("\n"):
            line = line.strip()
            # Skip prompt text and non-questions
            if not line or line.startswith("Problem:") or line.startswith("To solve") or \
               line.startswith("Let's") or "?" not in line:
                continue
            sub_questions.append(line)
        return sub_questions
        
    def get_confidence_threshold(self, depth: int, question: str = "") -> float:
        """
        Calculate depth and complexity adjusted confidence threshold.
        
        Args:
            depth (int): Current recursion depth
            question (str): The question to analyze for complexity
            
        Returns:
            float: Adjusted confidence threshold
        """
        # Start with depth-based adjustment
        threshold = self.base_threshold * (self.threshold_decay ** depth)
        
        # Apply complexity-based adjustment if question provided
        if question:
            for complexity, pattern in self.complexity_patterns.items():
                if re.search(pattern, question.lower()):
                    threshold *= self.complexity_factors[complexity]
                    break
        
        # Never go below minimum confidence
        return max(self.min_confidence, threshold)
        
    def decompose_question(self, question: str, depth: int = 0) -> List[str]:
        """
        Recursively decompose a complex question into simpler sub-questions.
        Uses top-down approach with confidence-based decomposition control.
        
        Args:
            question (str): The question to decompose
            depth (int): Current recursion depth
            
        Returns:
            List[str]: List of decomposed sub-questions
            
        Example:
            >>> questioner = SocraticQuestioner()
            >>> subs = questioner.decompose_question("Solve (3x + 5)(2x - 1)")
            >>> print(subs)
            ['What is 3x + 5?', 'What is 2x - 1?', 'How do we multiply these terms?']
            
        Raises:
            RuntimeError: If models are not properly initialized
        """
        self._ensure_initialized()
        # Check cache first
        if question in self.question_cache:
            if self.debug:
                print(f"[DEBUG] Using cached decomposition for: {question}")
            return self.question_cache[question]
            
        # Check recursion depth
        if depth >= self.max_recursion_depth:
            if self.debug:
                print(f"[DEBUG] Max recursion depth {depth} reached for: {question}")
            return [question]
            
        # Check attempt count
        self.attempt_count[question] = self.attempt_count.get(question, 0) + 1
        if self.attempt_count[question] > self.max_attempts:
            if self.debug:
                print(f"[DEBUG] Max attempts ({self.max_attempts}) reached for: {question}")
            return [question]
            
        # Check if we've seen this question with sufficient confidence
        if question in self.seen_questions:
            prev_conf = self.seen_questions[question]
            if prev_conf >= self.min_confidence:
                if self.debug:
                    print(f"[DEBUG] Using cached question with confidence {prev_conf}: {question}")
                return [question]
                
        # Generate and validate sub-questions
        sub_questions = self.generate_subquestions(question)
        valid_sub_questions = []
        
        for sub_q in sub_questions:
            # Skip empty or too-short questions
            if not sub_q or len(sub_q.strip()) < 10:
                continue
                
            # Skip if sub-question is too similar to original
            if sub_q.lower() == question.lower():
                continue
                
            # Extract key terms from original question
            original_terms = set(re.findall(r'[a-zA-Z]+|\d+|[+\-*/=()^]', question))
            sub_terms = set(re.findall(r'[a-zA-Z]+|\d+|[+\-*/=()^]', sub_q))
            
            # Skip if sub-question doesn't share enough terms with original
            common_terms = original_terms & sub_terms
            if len(common_terms) < 2:  # Must share at least 2 terms
                if self.debug:
                    print(f"[DEBUG] Skipping question with insufficient context: {sub_q}")
                continue
                
            # Skip if sub-question is not a valid mathematical operation
            math_patterns = [
                # Algebra patterns
                r'solve .*[=]',          # Basic equation solving
                r'(?:solve|find).*\([^)]+\).*=',  # Complex equation solving
                r'factor[s]? .*[+\-*/()]',        # Factoring with grouping
                r'expand .*\([^)]+\)',            # Expansion with parentheses
                r'simplify .*[+\-*/()]',          # Simplification with grouping
                r'what (?:is|are) .*(?:[+\-*/=()]|\b(?:coefficient|term|factor)s?\b)', # Terms and operations
                r'how (?:do|can) we .*(?:[+\-*/=()]|\b(?:isolate|solve for|eliminate)\b)', # Process questions
                r'\b(?:coefficient|term|factor)s? .*(?:of|in|for) [^?]+\?', # Algebraic components
                r'what (?:happens?|do we (?:get|have)) (?:when|if) .*[=]', # Conditional steps
                r'why (?:can|should|do) we .*(?:[+\-*/=()]|\b(?:factor|expand|simplify)\b)', # Reasoning questions
                
                # Geometry patterns
                r'area .*(?:circle|square|triangle|rectangle)',  # Area calculations
                r'volume .*(?:cube|sphere|cylinder)',           # Volume calculations
                r'radius|diameter|circumference',               # Circle properties
                r'length|width|height|side',                    # Measurements
                r'angle|degree|perpendicular|parallel',         # Angles and lines
                
                # Number patterns
                r'\d+(?:\.\d+)?\s*(?:[+\-*/^]|\b(?:plus|minus|times|divided by)\b)',  # Operations
                r'π|pi|sqrt|square root|cube root',            # Mathematical constants/functions
                
                # General mathematical inquiry patterns
                r'(?:find|calculate|determine|evaluate)\s+(?:the|a)?\s*(?:value|sum|product|quotient|difference)',
                r'what\s+(?:number|value|result)',
                r'how\s+(?:much|many|long|far)'
            ]
            if not any(re.search(pattern, sub_q.lower()) for pattern in math_patterns):
                if self.debug:
                    print(f"[DEBUG] Skipping non-mathematical question: {sub_q}")
                continue
                
            valid_sub_questions.append(sub_q)
            
        # If no valid sub-questions, use structured fallback
        if not valid_sub_questions:
            if self.debug:
                print(f"[DEBUG] Using fallback decomposition for: {question}")
                
            if "=" in question:  # Equation solving
                # Check for factored form
                if "(" in question and ")" in question and "0" in question:
                    return [
                        f"What are the factors in the equation: {question}?",
                        "What property can we use when a product equals zero?",
                        "What equations do we get from each factor?",
                        "How do we solve each equation?"
                    ]
                # Check for quadratic form
                elif re.search(r'x\s*[\^2]|x\s*\*\s*x', question):
                    return [
                        f"Is this equation in standard quadratic form (ax² + bx + c = 0)?",
                        "What are the values of a, b, and c?",
                        "What method should we use to solve this quadratic equation?",
                        "How do we apply this method to find x?"
                    ]
                # Linear equation
                else:
                    return [
                        f"What terms contain the variable in: {question}?",
                        f"What constant terms are in: {question}?",
                        "How do we get all variable terms on one side?",
                        "How do we isolate the variable?"
                    ]
            elif any(op in question for op in ['+', '-', '*', '/', '^']):  # Arithmetic
                return [
                    f"What are the numbers and operations in: {question}?",
                    "What is the order of operations we should follow?",
                    "How do we perform each step?"
                ]
            else:  # Default fallback
                return [question]
            
        # Recursively decompose sub-questions if needed
        all_questions = []
        decomposition_log = []  # Track decomposition decisions
        
        for sub_q in sub_questions:
            # Check if we've seen this sub-question before
            if sub_q in self.seen_questions:
                confidence = self.seen_questions[sub_q]
                if self.debug:
                    print(f"[DEBUG] Using cached confidence {confidence:.3f} for: {sub_q}")
            else:
                # Try to answer the sub-question and get confidence score
                answer, confidence = self.answer_question(sub_q)
                self.seen_questions[sub_q] = confidence
            
            # Get complexity-adjusted threshold
            threshold = self.get_confidence_threshold(depth, sub_q)
            
            # Log decomposition decision
            decision = {
                'question': sub_q,
                'confidence': confidence,
                'threshold': threshold,
                'needs_decomposition': confidence < threshold,
                'complexity': next((c for c, p in self.complexity_patterns.items() 
                                if re.search(p, sub_q.lower())), 'medium')
            }
            decomposition_log.append(decision)
            
            if self.debug:
                print(f"[DEBUG] Confidence {confidence:.3f} vs threshold {threshold:.3f} at depth {depth}")
                
            # Only decompose further if:
            # 1. Confidence is low
            # 2. Haven't hit max depth
            # 3. Haven't exceeded max attempts
            # 4. Question hasn't been seen with good confidence
            should_decompose = (
                confidence < threshold and
                depth < self.max_recursion_depth and
                self.attempt_count.get(sub_q, 0) < self.max_attempts and
                (sub_q not in self.seen_questions or 
                 self.seen_questions[sub_q] < self.min_confidence)
            )
            
            if should_decompose:
                if self.debug:
                    print(f"[DEBUG] Low confidence ({confidence:.3f}) for: {sub_q}")
                    print(f"[DEBUG] Further decomposing at depth {depth}...")
                further_questions = self.decompose_question(sub_q, depth + 1)
                # Only add questions we haven't seen with good confidence
                for q in further_questions:
                    if q not in self.seen_questions or \
                       self.seen_questions[q] < self.min_confidence:
                        all_questions.append(q)
            else:
                if self.debug:
                    print(f"[DEBUG] Adding question with confidence {confidence:.3f}: {sub_q}")
                all_questions.append(sub_q)
                
        # Save decomposition log for analysis
        if not hasattr(self, 'decomposition_history'):
            self.decomposition_history = []
        self.decomposition_history.append(decomposition_log)
        
        # Cache the decomposition result
        self.question_cache[question] = all_questions
                
        return all_questions
        
    def answer_question(self, question: str) -> Tuple[str, float]:
        """
        Answer a question and return confidence score.
        Uses the language model to generate an answer and confidence calculator
        to evaluate the answer quality.
        
        Args:
            question (str): The question to answer
            
        Returns:
            Tuple[str, float]: (answer with steps, confidence score)
            
        Example:
            >>> questioner = SocraticQuestioner()
            >>> answer, conf = questioner.answer_question("What is 7 + 3?")
            >>> print(f"Answer (confidence: {conf:.2f}): {answer}")
            Answer (confidence: 0.92): 7 + 3 = 10
            
        Raises:
            RuntimeError: If models are not properly initialized
        """
        self._ensure_initialized()
        prompt = f"""Solve this math problem step by step:
Question: {question}
Solution:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        # Configure generation parameters for sampling
        if self.debug:
            print(f"[DEBUG] Answering question: {question}")
            print(f"[DEBUG] Using answer generation config: {self.answer_generation_config}")
            
        # Use stored generation config
        generation_config = self.answer_generation_config.copy()
        
        outputs = self.model.generate(
            **inputs,
            **generation_config
        )
        
        # Get multiple answer samples
        answers = [self.tokenizer.decode(output, skip_special_tokens=True) 
                  for output in outputs]
        
        # Calculate confidence using our novel method
        model_outputs = {
            'samples': answers,
            'loss': outputs.loss if hasattr(outputs, 'loss') else None
        }
        
        # Use the first answer as the main response
        answer = answers[0]
        confidence = self.confidence_calculator.calculate_confidence(
            question, answer, model_outputs
        )
        
        return answer, confidence
        
    def solve_problem(self, question: str) -> Tuple[str, float]:
        """
        Main entry point for solving a math problem using Socratic questioning.
        Implements the complete recursive solution process with confidence-based
        decomposition and answer propagation.
        
        Args:
            question (str): The math problem to solve
            
        Returns:
            Tuple[str, float]: (solution with steps, confidence score)
            
        Example:
            >>> questioner = SocraticQuestioner()
            >>> solution, conf = questioner.solve_problem("What is 12 * 5?")
            >>> print(f"Solution (confidence: {conf:.2f}):")
            >>> print(solution)
            Solution (confidence: 0.85):
            Step-by-step solution:
            1. First, let's break down 12 * 5
            2. 12 * 5 = (10 + 2) * 5
            3. = (10 * 5) + (2 * 5)
            4. = 50 + 10
            5. = 60
        
        Raises:
            RuntimeError: If models are not properly initialized
        """
        self._ensure_initialized()
            
        # First attempt to solve directly
        answer, confidence = self.answer_question(question)
        
        # If confidence is high enough, return the answer
        threshold = self.get_confidence_threshold(0)  # Use base threshold for initial question
        if confidence >= threshold:
            return answer, confidence
            
        # Otherwise, decompose and solve sub-questions
        sub_questions = self.decompose_question(question)
        sub_answers = []
        sub_confidences = []
        
        # Solve each sub-question
        for sub_q in sub_questions:
            sub_ans, sub_conf = self.answer_question(sub_q)
            threshold = self.get_confidence_threshold(1)  # Sub-questions are at depth 1
            if sub_conf >= threshold:
                sub_answers.append(f"For {sub_q}\nAnswer: {sub_ans}")
                sub_confidences.append(sub_conf)
                
        # If we have sub-answers, combine them with weighted confidence
        if sub_answers:
            # Weight sub-confidences by their relative values
            weights = np.array(sub_confidences) / sum(sub_confidences)
            weighted_confidence = sum(weights * sub_confidences)
            
            # Combine answers with step numbering
            steps = []
            for i, (sub_q, sub_ans) in enumerate(zip(sub_questions, sub_answers), 1):
                steps.append(f"{i}. {sub_ans}")
            
            combined_answer = "Step-by-step solution:\n" + "\n".join(steps)
            
            # Calculate final confidence as weighted average of:
            # 1. Sub-question confidences (weighted by their values)
            # 2. Overall solution coherence
            coherence_conf = self.confidence_calculator.calculate_confidence(
                question, combined_answer, {'samples': [combined_answer]}
            )
            
            final_confidence = 0.7 * weighted_confidence + 0.3 * coherence_conf
            return combined_answer, final_confidence
            
        # If decomposition didn't help, return original answer
        return answer, confidence
