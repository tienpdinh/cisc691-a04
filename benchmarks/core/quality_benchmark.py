"""
Response Quality Benchmarking Module

Evaluates the quality, relevance, and accuracy of RAG system responses.
Uses multiple evaluation metrics including BLEU, ROUGE, BERTScore, and LLM-based evaluation.
"""

import json
import logging
import statistics
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import asyncio
import httpx

# Import evaluation libraries
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge-score not available. Install with: pip install rouge-score")

try:
    from bert_score import BERTScorer
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    logging.warning("bert-score not available. Install with: pip install bert-score")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

@dataclass
class QualityMetrics:
    """Container for response quality evaluation results."""
    query: str
    generated_response: str
    reference_response: str
    rouge_scores: Dict[str, float]
    bert_score: float
    semantic_similarity: float
    length_ratio: float
    coherence_score: float
    relevance_score: float
    factual_accuracy_score: float
    llm_judge_score: float
    overall_quality_score: float
    timestamp: str

@dataclass
class QualityTestCase:
    """Test case for quality evaluation."""
    query: str
    reference_answer: str
    context_documents: List[str]
    evaluation_criteria: List[str]
    difficulty: str  # "easy", "medium", "hard"
    category: str    # "factual", "analytical", "comparative", etc.

class QualityBenchmark:
    """
    Comprehensive quality benchmarking for RAG system responses.
    
    Evaluates:
    - ROUGE scores (overlap with reference)
    - BERTScore (semantic similarity)
    - Coherence and fluency
    - Factual accuracy
    - Relevance to query
    - LLM-based evaluation
    """
    
    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 15):
        """
        Initialize quality benchmark.
        
        Args:
            base_url: Base URL of the RAG API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Initialize evaluation models
        self._init_evaluation_models()
    
    def _init_evaluation_models(self):
        """Initialize evaluation models and scorers."""
        # Initialize ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.logger.info("Initialized ROUGE scorer")
        else:
            self.rouge_scorer = None
        
        # Initialize BERTScore
        if BERT_SCORE_AVAILABLE:
            try:
                self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
                self.logger.info("Initialized BERTScore model")
            except Exception as e:
                self.logger.warning(f"Could not initialize BERTScore: {e}")
                self.bert_scorer = None
        else:
            self.bert_scorer = None
        
        # Initialize sentence transformer for semantic similarity
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Initialized sentence transformer for semantic similarity")
            except Exception as e:
                self.logger.warning(f"Could not load sentence transformer: {e}")
                self.similarity_model = None
        else:
            self.similarity_model = None
    
    async def generate_response(self, query: str, use_rag: bool = True) -> str:
        """
        Generate response from RAG API.
        
        Args:
            query: Input query
            use_rag: Whether to use RAG or direct LLM
            
        Returns:
            Generated response text
        """
        url = f"{self.base_url}/query"
        payload = {"query": query, "use_rag": use_rag}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                return data.get('response', '')
                
            except Exception as e:
                self.logger.error(f"Error generating response for query '{query}': {e}")
                return ""
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores between generated and reference text.
        
        Args:
            generated: Generated response
            reference: Reference response
            
        Returns:
            Dictionary of ROUGE scores
        """
        if not self.rouge_scorer or not generated or not reference:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, generated)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            self.logger.error(f"Error calculating ROUGE scores: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_bert_score(self, generated: str, reference: str) -> float:
        """
        Calculate BERTScore between generated and reference text.
        
        Args:
            generated: Generated response
            reference: Reference response
            
        Returns:
            BERTScore F1 score
        """
        if not self.bert_scorer or not generated or not reference:
            return 0.0
        
        try:
            P, R, F1 = self.bert_scorer.score([generated], [reference])
            return float(F1.item())
        except Exception as e:
            self.logger.error(f"Error calculating BERTScore: {e}")
            return 0.0
    
    def calculate_semantic_similarity(self, generated: str, reference: str) -> float:
        """
        Calculate semantic similarity using sentence transformers.
        
        Args:
            generated: Generated response
            reference: Reference response
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not self.similarity_model or not generated or not reference:
            return 0.0
        
        try:
            # Encode texts
            embeddings = self.similarity_model.encode([generated, reference])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_length_ratio(self, generated: str, reference: str) -> float:
        """
        Calculate length ratio between generated and reference text.
        
        Args:
            generated: Generated response
            reference: Reference response
            
        Returns:
            Length ratio (1.0 = same length)
        """
        if not reference:
            return 0.0
        
        gen_len = len(generated.split())
        ref_len = len(reference.split())
        
        if ref_len == 0:
            return 0.0
        
        return min(gen_len / ref_len, ref_len / gen_len)
    
    def calculate_coherence_score(self, text: str) -> float:
        """
        Calculate coherence score based on text structure and flow.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Coherence score (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        # Simple heuristic-based coherence scoring
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.5  # Single sentence gets neutral score
        
        # Check for repetition (lower score if too repetitive)
        unique_sentences = set(sentences)
        repetition_penalty = len(unique_sentences) / len(sentences)
        
        # Check for appropriate sentence length variation
        lengths = [len(s.split()) for s in sentences]
        if lengths:
            length_variation = statistics.stdev(lengths) / statistics.mean(lengths) if statistics.mean(lengths) > 0 else 0
            length_score = min(1.0, length_variation)
        else:
            length_score = 0.0
        
        # Combine factors
        coherence = (repetition_penalty + length_score) / 2
        return min(1.0, coherence)
    
    def calculate_relevance_score(self, query: str, response: str) -> float:
        """
        Calculate relevance of response to the query.
        
        Args:
            query: Original query
            response: Generated response
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        return self.calculate_semantic_similarity(query, response)
    
    def calculate_factual_accuracy_score(self, response: str, context_docs: List[str]) -> float:
        """
        Calculate factual accuracy based on alignment with source documents.
        
        Args:
            response: Generated response
            context_docs: Source documents
            
        Returns:
            Factual accuracy score (0.0 to 1.0)
        """
        if not context_docs or not response:
            return 0.0
        
        # Calculate similarity with source documents
        combined_context = ' '.join(context_docs)
        return self.calculate_semantic_similarity(response, combined_context)
    
    async def calculate_llm_judge_score(self, query: str, response: str, reference: str) -> float:
        """
        Use LLM as a judge to evaluate response quality.
        
        Args:
            query: Original query
            response: Generated response
            reference: Reference response
            
        Returns:
            LLM judge score (0.0 to 1.0)
        """
        # Create evaluation prompt
        eval_prompt = f"""
Please evaluate the quality of the following response to a question on a scale of 0-10:

Question: {query}

Response to evaluate: {response}

Reference answer: {reference}

Consider the following criteria:
1. Accuracy and correctness
2. Completeness and thoroughness
3. Clarity and coherence
4. Relevance to the question

Provide only a numeric score from 0-10:
"""
        
        try:
            eval_response = await self.generate_response(eval_prompt, use_rag=False)
            
            # Extract numeric score
            import re
            scores = re.findall(r'\b([0-9](?:\.[0-9])?|10(?:\.0)?)\b', eval_response)
            
            if scores:
                score = float(scores[0])
                return min(10.0, max(0.0, score)) / 10.0  # Normalize to 0-1
            else:
                return 0.5  # Default neutral score if parsing fails
                
        except Exception as e:
            self.logger.error(f"Error in LLM judge evaluation: {e}")
            return 0.5
    
    def calculate_overall_quality_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall quality score from individual metrics.
        
        Args:
            metrics: Dictionary of individual metric scores
            
        Returns:
            Overall quality score (0.0 to 1.0)
        """
        # Weighted combination of metrics
        weights = {
            'rouge_avg': 0.2,
            'bert_score': 0.2,
            'semantic_similarity': 0.15,
            'coherence_score': 0.15,
            'relevance_score': 0.15,
            'factual_accuracy_score': 0.1,
            'llm_judge_score': 0.05
        }
        
        rouge_avg = statistics.mean(metrics.get('rouge_scores', {}).values()) if metrics.get('rouge_scores') else 0
        
        weighted_score = (
            weights['rouge_avg'] * rouge_avg +
            weights['bert_score'] * metrics.get('bert_score', 0) +
            weights['semantic_similarity'] * metrics.get('semantic_similarity', 0) +
            weights['coherence_score'] * metrics.get('coherence_score', 0) +
            weights['relevance_score'] * metrics.get('relevance_score', 0) +
            weights['factual_accuracy_score'] * metrics.get('factual_accuracy_score', 0) +
            weights['llm_judge_score'] * metrics.get('llm_judge_score', 0)
        )
        
        return weighted_score
    
    async def evaluate_response_quality(self, test_case: QualityTestCase) -> QualityMetrics:
        """
        Evaluate the quality of a response for a given test case.
        
        Args:
            test_case: QualityTestCase to evaluate
            
        Returns:
            QualityMetrics with evaluation results
        """
        self.logger.info(f"Evaluating response quality for: '{test_case.query}'")
        
        # Generate response
        generated_response = await self.generate_response(test_case.query, use_rag=True)
        
        if not generated_response:
            self.logger.warning(f"No response generated for query: '{test_case.query}'")
            # Return zero metrics
            return QualityMetrics(
                query=test_case.query,
                generated_response="",
                reference_response=test_case.reference_answer,
                rouge_scores={'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0},
                bert_score=0.0,
                semantic_similarity=0.0,
                length_ratio=0.0,
                coherence_score=0.0,
                relevance_score=0.0,
                factual_accuracy_score=0.0,
                llm_judge_score=0.0,
                overall_quality_score=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        # Calculate all metrics
        rouge_scores = self.calculate_rouge_scores(generated_response, test_case.reference_answer)
        bert_score = self.calculate_bert_score(generated_response, test_case.reference_answer)
        semantic_similarity = self.calculate_semantic_similarity(generated_response, test_case.reference_answer)
        length_ratio = self.calculate_length_ratio(generated_response, test_case.reference_answer)
        coherence_score = self.calculate_coherence_score(generated_response)
        relevance_score = self.calculate_relevance_score(test_case.query, generated_response)
        factual_accuracy_score = self.calculate_factual_accuracy_score(generated_response, test_case.context_documents)
        llm_judge_score = await self.calculate_llm_judge_score(test_case.query, generated_response, test_case.reference_answer)
        
        # Calculate overall score
        all_metrics = {
            'rouge_scores': rouge_scores,
            'bert_score': bert_score,
            'semantic_similarity': semantic_similarity,
            'coherence_score': coherence_score,
            'relevance_score': relevance_score,
            'factual_accuracy_score': factual_accuracy_score,
            'llm_judge_score': llm_judge_score
        }
        overall_score = self.calculate_overall_quality_score(all_metrics)
        
        metrics = QualityMetrics(
            query=test_case.query,
            generated_response=generated_response,
            reference_response=test_case.reference_answer,
            rouge_scores=rouge_scores,
            bert_score=bert_score,
            semantic_similarity=semantic_similarity,
            length_ratio=length_ratio,
            coherence_score=coherence_score,
            relevance_score=relevance_score,
            factual_accuracy_score=factual_accuracy_score,
            llm_judge_score=llm_judge_score,
            overall_quality_score=overall_score,
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(f"Quality evaluation completed. Overall score: {overall_score:.3f}")
        
        return metrics
    
    def load_test_cases(self, file_path: Path) -> List[QualityTestCase]:
        """
        Load quality test cases from JSON file.
        
        Args:
            file_path: Path to test cases JSON file
            
        Returns:
            List of QualityTestCase objects
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            test_cases = []
            for item in data:
                test_cases.append(QualityTestCase(
                    query=item['query'],
                    reference_answer=item['reference_answer'],
                    context_documents=item.get('context_documents', []),
                    evaluation_criteria=item.get('evaluation_criteria', []),
                    difficulty=item.get('difficulty', 'medium'),
                    category=item.get('category', 'general')
                ))
            
            self.logger.info(f"Loaded {len(test_cases)} quality test cases")
            return test_cases
            
        except Exception as e:
            self.logger.error(f"Error loading test cases from {file_path}: {e}")
            return []
    
    def save_results(self, metrics_list: List[QualityMetrics], output_file: Optional[Path] = None):
        """
        Save quality benchmark results to file.
        
        Args:
            metrics_list: List of QualityMetrics to save
            output_file: Output file path
        """
        if output_file is None:
            output_file = Path("benchmarks/results") / f"quality_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        results = [asdict(metrics) for metrics in metrics_list]
        
        # Add summary statistics
        if results:
            summary = {
                'total_queries': len(results),
                'average_overall_score': statistics.mean([r['overall_quality_score'] for r in results]),
                'average_rouge1': statistics.mean([r['rouge_scores']['rouge1'] for r in results]),
                'average_bert_score': statistics.mean([r['bert_score'] for r in results]),
                'average_semantic_similarity': statistics.mean([r['semantic_similarity'] for r in results]),
                'average_coherence': statistics.mean([r['coherence_score'] for r in results]),
                'average_relevance': statistics.mean([r['relevance_score'] for r in results]),
                'average_factual_accuracy': statistics.mean([r['factual_accuracy_score'] for r in results])
            }
            
            output_data = {
                'summary': summary,
                'individual_results': results,
                'timestamp': datetime.now().isoformat()
            }
        else:
            output_data = {'summary': {}, 'individual_results': [], 'timestamp': datetime.now().isoformat()}
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Quality results saved to {output_file}")
    
    async def run_full_benchmark(self, test_cases_file: Path) -> List[QualityMetrics]:
        """
        Run comprehensive quality benchmark.
        
        Args:
            test_cases_file: Path to test cases JSON file
            
        Returns:
            List of QualityMetrics for all test cases
        """
        self.logger.info("Starting full quality benchmark...")
        
        # Load test cases
        test_cases = self.load_test_cases(test_cases_file)
        
        if not test_cases:
            self.logger.error("No test cases available")
            return []
        
        # Evaluate each test case
        results = []
        for test_case in test_cases:
            try:
                metrics = await self.evaluate_response_quality(test_case)
                results.append(metrics)
            except Exception as e:
                self.logger.error(f"Error evaluating test case '{test_case.query}': {e}")
        
        # Save results
        self.save_results(results)
        
        self.logger.info(f"Quality benchmark completed. Evaluated {len(results)} test cases.")
        
        return results
    
    def create_sample_test_cases(self, output_file: Path):
        """
        Create sample test cases for quality evaluation.
        
        Args:
            output_file: Path where to save the sample file
        """
        sample_data = [
            {
                "query": "What are the main principles of effective project management?",
                "reference_answer": "Effective project management involves several key principles: clear scope definition, realistic timeline planning, proper resource allocation, stakeholder communication, risk management, and continuous monitoring and control. These principles help ensure projects are completed on time, within budget, and meet their objectives.",
                "context_documents": [
                    "Project management requires careful planning and execution to achieve success.",
                    "Key elements include scope, time, cost, quality, and stakeholder management."
                ],
                "evaluation_criteria": ["accuracy", "completeness", "clarity"],
                "difficulty": "medium",
                "category": "factual"
            },
            {
                "query": "How does artificial intelligence impact modern business operations?",
                "reference_answer": "Artificial intelligence significantly impacts modern business operations by automating routine tasks, enabling data-driven decision making, improving customer service through chatbots and personalization, optimizing supply chains, and enhancing predictive analytics. AI helps businesses increase efficiency, reduce costs, and gain competitive advantages through better insights and automation.",
                "context_documents": [
                    "AI technologies are transforming how businesses operate and make decisions.",
                    "Machine learning and automation are key drivers of business efficiency."
                ],
                "evaluation_criteria": ["analytical depth", "practical examples", "comprehensiveness"],
                "difficulty": "medium",
                "category": "analytical"
            },
            {
                "query": "What are the potential risks and challenges in implementing RAG systems?",
                "reference_answer": "Implementing RAG systems presents several challenges: ensuring retrieval accuracy and relevance, managing context length limitations, maintaining response quality and consistency, handling domain-specific knowledge, addressing privacy and security concerns, and managing computational costs. Additionally, there are challenges in evaluation metrics, keeping knowledge bases updated, and ensuring system reliability.",
                "context_documents": [
                    "RAG systems combine retrieval and generation but face technical challenges.",
                    "Key issues include accuracy, relevance, cost, and system complexity."
                ],
                "evaluation_criteria": ["technical accuracy", "comprehensive coverage", "practical insights"],
                "difficulty": "hard",
                "category": "technical"
            }
        ]
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        self.logger.info(f"Sample test cases created at {output_file}")