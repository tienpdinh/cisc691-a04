"""
Retrieval Accuracy Benchmarking Module

Evaluates the precision and recall of document retrieval in the RAG system.
Measures how well the system finds relevant documents for given queries.
"""

import json
import logging
import statistics
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import httpx
import asyncio

# For semantic similarity calculation
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers scikit-learn")

@dataclass
class AccuracyMetrics:
    """Container for retrieval accuracy results."""
    query: str
    precision_at_k: Dict[int, float]  # precision@1, @3, @5
    recall_at_k: Dict[int, float]     # recall@1, @3, @5
    mean_reciprocal_rank: float
    normalized_dcg: float
    semantic_similarity_score: float
    relevant_docs_found: int
    total_relevant_docs: int
    retrieved_docs_count: int
    timestamp: str

@dataclass
class GroundTruthItem:
    """Ground truth query-document pair."""
    query: str
    relevant_doc_ids: List[str]
    relevant_content: List[str]
    query_intent: str
    difficulty: str  # "easy", "medium", "hard"

class AccuracyBenchmark:
    """
    Comprehensive accuracy benchmarking for document retrieval.
    
    Evaluates:
    - Precision@K (how many retrieved docs are relevant)
    - Recall@K (how many relevant docs were retrieved)
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (NDCG)
    - Semantic similarity of retrieved content
    """
    
    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 15):
        """
        Initialize accuracy benchmark.
        
        Args:
            base_url: Base URL of the RAG API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Load sentence transformer model for semantic similarity
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Loaded sentence transformer model for semantic similarity")
            except Exception as e:
                self.logger.warning(f"Could not load sentence transformer: {e}")
                self.similarity_model = None
        else:
            self.similarity_model = None
    
    async def retrieve_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents from the RAG API.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        url = f"{self.base_url}/retrieve"
        payload = {"query": query, "top_k": top_k}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                return data.get('results', [])
                
            except Exception as e:
                self.logger.error(f"Error retrieving documents for query '{query}': {e}")
                return []
    
    def calculate_precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Calculate Precision@K metric.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Number of top documents to consider
            
        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if k == 0 or not retrieved_docs:
            return 0.0
        
        top_k_retrieved = retrieved_docs[:k]
        relevant_retrieved = sum(1 for doc in top_k_retrieved if doc in relevant_docs)
        
        return relevant_retrieved / min(k, len(top_k_retrieved))
    
    def calculate_recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Calculate Recall@K metric.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Number of top documents to consider
            
        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not relevant_docs:
            return 0.0
        
        top_k_retrieved = retrieved_docs[:k]
        relevant_retrieved = sum(1 for doc in top_k_retrieved if doc in relevant_docs)
        
        return relevant_retrieved / len(relevant_docs)
    
    def calculate_mean_reciprocal_rank(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            
        Returns:
            MRR score (0.0 to 1.0)
        """
        for rank, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                return 1.0 / rank
        return 0.0
    
    def calculate_ndcg(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG).
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Number of top documents to consider
            
        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        def dcg(relevance_scores: List[int]) -> float:
            """Calculate DCG from relevance scores."""
            dcg_score = 0.0
            for i, rel in enumerate(relevance_scores):
                if i == 0:
                    dcg_score += rel
                else:
                    dcg_score += rel / (math.log2(i + 1))
            return dcg_score
        
        import math
        
        # Create relevance scores (1 for relevant, 0 for not relevant)
        retrieved_relevance = []
        for doc in retrieved_docs[:k]:
            retrieved_relevance.append(1 if doc in relevant_docs else 0)
        
        # Calculate ideal relevance (all relevant docs first)
        ideal_relevance = [1] * min(len(relevant_docs), k) + [0] * max(0, k - len(relevant_docs))
        
        # Calculate DCG for retrieved and ideal rankings
        retrieved_dcg = dcg(retrieved_relevance)
        ideal_dcg = dcg(ideal_relevance)
        
        # Calculate NDCG
        if ideal_dcg == 0:
            return 0.0
        return retrieved_dcg / ideal_dcg
    
    def calculate_semantic_similarity(self, retrieved_content: List[str], relevant_content: List[str]) -> float:
        """
        Calculate semantic similarity between retrieved and relevant content.
        
        Args:
            retrieved_content: List of retrieved document contents
            relevant_content: List of relevant document contents
            
        Returns:
            Average semantic similarity score (0.0 to 1.0)
        """
        if not self.similarity_model or not retrieved_content or not relevant_content:
            return 0.0
        
        try:
            # Encode texts
            retrieved_embeddings = self.similarity_model.encode(retrieved_content)
            relevant_embeddings = self.similarity_model.encode(relevant_content)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(retrieved_embeddings, relevant_embeddings)
            
            # Return average maximum similarity for each retrieved document
            max_similarities = [max(row) for row in similarities]
            return float(statistics.mean(max_similarities))
            
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    async def evaluate_query(self, ground_truth: GroundTruthItem, top_k: int = 10) -> AccuracyMetrics:
        """
        Evaluate retrieval accuracy for a single query.
        
        Args:
            ground_truth: Ground truth data for the query
            top_k: Number of documents to retrieve
            
        Returns:
            AccuracyMetrics for the query
        """
        self.logger.info(f"Evaluating query: '{ground_truth.query}'")
        
        # Retrieve documents
        retrieved_results = await self.retrieve_documents(ground_truth.query, top_k)
        
        # Extract document IDs and content
        retrieved_doc_ids = [result.get('id', f"doc_{i}") for i, result in enumerate(retrieved_results)]
        retrieved_content = [result.get('text', '') for result in retrieved_results]
        
        # Calculate metrics
        precision_at_k = {}
        recall_at_k = {}
        
        for k in [1, 3, 5, 10]:
            precision_at_k[k] = self.calculate_precision_at_k(
                retrieved_doc_ids, ground_truth.relevant_doc_ids, k
            )
            recall_at_k[k] = self.calculate_recall_at_k(
                retrieved_doc_ids, ground_truth.relevant_doc_ids, k
            )
        
        mrr = self.calculate_mean_reciprocal_rank(retrieved_doc_ids, ground_truth.relevant_doc_ids)
        ndcg = self.calculate_ndcg(retrieved_doc_ids, ground_truth.relevant_doc_ids, top_k)
        
        # Calculate semantic similarity
        semantic_similarity = self.calculate_semantic_similarity(
            retrieved_content, ground_truth.relevant_content
        )
        
        # Count relevant documents found
        relevant_found = sum(1 for doc_id in retrieved_doc_ids if doc_id in ground_truth.relevant_doc_ids)
        
        metrics = AccuracyMetrics(
            query=ground_truth.query,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mean_reciprocal_rank=mrr,
            normalized_dcg=ndcg,
            semantic_similarity_score=semantic_similarity,
            relevant_docs_found=relevant_found,
            total_relevant_docs=len(ground_truth.relevant_doc_ids),
            retrieved_docs_count=len(retrieved_doc_ids),
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(f"Query evaluation completed. Precision@5: {precision_at_k.get(5, 0):.3f}, "
                        f"Recall@5: {recall_at_k.get(5, 0):.3f}, MRR: {mrr:.3f}")
        
        return metrics
    
    def load_ground_truth(self, file_path: Path) -> List[GroundTruthItem]:
        """
        Load ground truth data from JSON file.
        
        Args:
            file_path: Path to ground truth JSON file
            
        Returns:
            List of GroundTruthItem objects
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            ground_truth_items = []
            for item in data:
                ground_truth_items.append(GroundTruthItem(
                    query=item['query'],
                    relevant_doc_ids=item['relevant_doc_ids'],
                    relevant_content=item.get('relevant_content', []),
                    query_intent=item.get('query_intent', 'unknown'),
                    difficulty=item.get('difficulty', 'medium')
                ))
            
            self.logger.info(f"Loaded {len(ground_truth_items)} ground truth items")
            return ground_truth_items
            
        except Exception as e:
            self.logger.error(f"Error loading ground truth from {file_path}: {e}")
            return []
    
    def save_results(self, metrics_list: List[AccuracyMetrics], output_file: Optional[Path] = None):
        """
        Save accuracy benchmark results to file.
        
        Args:
            metrics_list: List of AccuracyMetrics to save
            output_file: Output file path
        """
        if output_file is None:
            output_file = Path("benchmarks/results") / f"accuracy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        results = [asdict(metrics) for metrics in metrics_list]
        
        # Add summary statistics
        if results:
            summary = {
                'total_queries': len(results),
                'average_precision_at_5': statistics.mean([r['precision_at_k'][5] for r in results]),
                'average_recall_at_5': statistics.mean([r['recall_at_k'][5] for r in results]),
                'average_mrr': statistics.mean([r['mean_reciprocal_rank'] for r in results]),
                'average_ndcg': statistics.mean([r['normalized_dcg'] for r in results]),
                'average_semantic_similarity': statistics.mean([r['semantic_similarity_score'] for r in results])
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
        
        self.logger.info(f"Accuracy results saved to {output_file}")
    
    async def run_full_benchmark(self, ground_truth_file: Path) -> List[AccuracyMetrics]:
        """
        Run comprehensive accuracy benchmark.
        
        Args:
            ground_truth_file: Path to ground truth JSON file
            
        Returns:
            List of AccuracyMetrics for all queries
        """
        self.logger.info("Starting full accuracy benchmark...")
        
        # Load ground truth data
        ground_truth_items = self.load_ground_truth(ground_truth_file)
        
        if not ground_truth_items:
            self.logger.error("No ground truth data available")
            return []
        
        # Evaluate each query
        results = []
        for item in ground_truth_items:
            try:
                metrics = await self.evaluate_query(item)
                results.append(metrics)
            except Exception as e:
                self.logger.error(f"Error evaluating query '{item.query}': {e}")
        
        # Save results
        self.save_results(results)
        
        self.logger.info(f"Accuracy benchmark completed. Evaluated {len(results)} queries.")
        
        return results
    
    def create_sample_ground_truth(self, output_file: Path):
        """
        Create a sample ground truth file for testing.
        
        Args:
            output_file: Path where to save the sample file
        """
        sample_data = [
            {
                "query": "What are the key factors in project management?",
                "relevant_doc_ids": ["doc_0", "doc_1"],
                "relevant_content": [
                    "Project management involves planning, executing, and monitoring projects to achieve specific goals.",
                    "Key factors include scope, time, cost, quality, resources, and stakeholder management."
                ],
                "query_intent": "factual",
                "difficulty": "easy"
            },
            {
                "query": "How can artificial intelligence improve business processes?",
                "relevant_doc_ids": ["doc_2", "doc_3"],
                "relevant_content": [
                    "AI can automate repetitive tasks and provide data-driven insights.",
                    "Machine learning algorithms can optimize business workflows and reduce costs."
                ],
                "query_intent": "analytical",
                "difficulty": "medium"
            },
            {
                "query": "What are the challenges in implementing RAG systems?",
                "relevant_doc_ids": ["doc_4"],
                "relevant_content": [
                    "RAG systems face challenges in retrieval accuracy, context relevance, and response generation quality."
                ],
                "query_intent": "technical",
                "difficulty": "hard"
            }
        ]
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        self.logger.info(f"Sample ground truth created at {output_file}")