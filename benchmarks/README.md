# RAG System Performance Benchmarking

Comprehensive benchmarking suite for evaluating RAG system performance across multiple dimensions.

## 🎯 Features & Capabilities

### 1. **Latency Benchmarking** (`core/latency_benchmark.py`)
- **Response time measurement** for all API endpoints
- **Concurrent load testing** with configurable users (1-10+ concurrent)
- **Throughput analysis** (requests per second)
- **Statistical analysis** (P95, P99, mean, median, min, max)
- **Performance under load** testing with error tracking

### 2. **Retrieval Accuracy Assessment** (`core/accuracy_benchmark.py`)
- **Precision@K and Recall@K** metrics (K=1,3,5,10)
- **Mean Reciprocal Rank (MRR)** calculation
- **Normalized Discounted Cumulative Gain (NDCG)**
- **Semantic similarity** scoring using sentence transformers
- **Ground truth evaluation** with customizable test sets

### 3. **Response Quality Evaluation** (`core/quality_benchmark.py`)
- **ROUGE scores** (ROUGE-1, ROUGE-2, ROUGE-L)
- **BERTScore** semantic similarity (when internet available)
- **Coherence and fluency** analysis
- **LLM-as-judge** evaluation using the API itself
- **Factual accuracy** assessment against source documents
- **Multi-dimensional quality scoring** with weighted averages

### 4. **Baseline Comparison** (`core/baseline_comparison.py`)
- **RAG vs Direct LLM** performance comparison
- **Improvement measurement** across all metrics
- **Side-by-side response analysis**
- **Win/loss statistics** and improvement percentages
- **Latency comparison** between RAG and non-RAG approaches

### 5. **Results Analysis & Visualization** (`scripts/view_results.py`)
- **Comprehensive results viewer** with multiple output formats
- **Trend analysis** and comparison tools across multiple runs
- **Export capabilities** (JSON, CSV, text summaries)
- **Summary reporting** with key performance indicators

## 📁 Project Structure

```
benchmarks/
├── README.md                          # 📖 This comprehensive guide
├── __init__.py                        # 🔗 Main package imports
├── requirements_benchmarks.txt        # 📦 Additional dependencies
│
├── core/                              # 🧠 Core benchmark modules
│   ├── __init__.py                    
│   ├── latency_benchmark.py          # ⏱️ Response time measurement
│   ├── accuracy_benchmark.py         # 🎯 Retrieval accuracy assessment
│   ├── quality_benchmark.py          # ⭐ Response quality evaluation
│   ├── baseline_comparison.py        # 🥊 RAG vs baseline comparison
│   └── benchmark_runner.py           # 🏃 Orchestrates all benchmarks
│
├── scripts/                           # 🛠️ Execution utilities
│   ├── __init__.py
│   ├── run_benchmarks.py             # 🚀 Main benchmark runner
│   ├── test_benchmarks.py            # ✅ System validation
│   └── view_results.py               # 📊 Results analysis tool
│
├── configs/                           # ⚙️ Configuration files
│   └── benchmark_config.yaml         # 📝 Benchmark parameters
│
├── data/                              # 📁 Test datasets (auto-generated)
├── results/                           # 📈 Benchmark outputs & reports
└── metrics/                           # 📊 Custom evaluation metrics
```

## 🚀 Quick Start Guide

### **Prerequisites**

1. **Install benchmark dependencies:**
```bash
pip install -r benchmarks/requirements_benchmarks.txt
```

2. **Start your RAG system with the specific setup you mentioned:**

**Option A: Full Docker Setup**
```bash
# Start all services
docker compose up -d

# Verify services
curl http://localhost:8001/health  # RAG API
curl http://localhost:8000         # ChromaDB
curl http://localhost:11434/api/tags  # Ollama (if in Docker)
```

**Option B: Your Setup (RAG+ChromaDB in Docker, Ollama on Mac)**
```bash
# 1. Start Ollama locally on Mac M2
ollama serve
ollama pull llama3.1:8b  # If not already downloaded

# 2. Update docker-compose.yml for host connectivity:
# Add to rag-api service:
#   extra_hosts:
#     - "host.docker.internal:host-gateway"

# 3. Update config to point to local Ollama:
# "llm_api_url": "http://host.docker.internal:11434/api/generate"

# 4. Start Docker services
docker compose up -d chromadb rag-api

# 5. Verify connectivity
curl http://localhost:8001/health  # Should show healthy
curl http://localhost:8000         # ChromaDB
curl http://localhost:11434/api/tags  # Local Ollama
```

### **Instant Validation**

**Test that everything works:**
```bash
python benchmarks/scripts/test_benchmarks.py
```

Expected output:
```
🚀 RAG Benchmark System Validation
==================================================
🧪 Testing Basic Benchmark Functionality...
📊 Testing Latency Benchmark...
  ✅ Query latency: 0.269s
  ✅ Status: 200
  ✅ Success: True
📈 Testing Concurrent Requests...
  ✅ Mean latency: 0.271s
  ✅ Success rate: 100.0%
  ✅ Throughput: 15.53 req/s
⭐ Testing Quality Benchmark...
  ✅ Overall quality: 0.782
  ✅ Coherence: 0.891
🥊 Testing RAG vs Baseline Comparison...
  ✅ RAG quality: 0.316
  ✅ Baseline quality: 0.297
  ✅ Improvement: 0.066
✅ All benchmark components are working correctly!
```

## 💻 Execution Options

### **Option 1: Quick Individual Component Tests**

**Latency Test:**
```bash
python -c "
import asyncio
from benchmarks.core.latency_benchmark import LatencyBenchmark

async def test():
    bench = LatencyBenchmark('http://localhost:8001')
    result = await bench.benchmark_query_endpoint(['What is AI?', 'How does ML work?'])
    print(f'Latency: {result.mean_latency:.3f}s')
    print(f'Throughput: {result.throughput_rps:.2f} req/s')
    print(f'P95 Latency: {result.p95_latency:.3f}s')
    print(f'Success Rate: {result.success_rate:.1f}%')

asyncio.run(test())
"
```

**RAG vs Baseline Comparison:**
```bash
python -c "
import asyncio
from benchmarks.core.baseline_comparison import BaselineComparison

async def test():
    comp = BaselineComparison('http://localhost:8001')
    result = await comp.compare_responses('What is machine learning?')
    print(f'RAG Quality: {result.rag_quality_score:.3f}')
    print(f'Baseline Quality: {result.baseline_quality_score:.3f}')
    print(f'Improvement: {result.improvement_score:.3f}')
    print(f'RAG Latency: {result.rag_latency:.3f}s')
    print(f'Baseline Latency: {result.baseline_latency:.3f}s')

asyncio.run(test())
"
```

**Quality Assessment:**
```bash
python -c "
import asyncio
from benchmarks.core.quality_benchmark import QualityBenchmark, QualityTestCase

async def test():
    quality = QualityBenchmark('http://localhost:8001')
    test_case = QualityTestCase(
        query='Explain artificial intelligence',
        reference_answer='AI is technology that enables machines to simulate human intelligence...',
        context_documents=['AI definition doc'],
        evaluation_criteria=['accuracy', 'clarity'],
        difficulty='medium',
        category='educational'
    )
    result = await quality.evaluate_response_quality(test_case)
    print(f'Overall Quality: {result.overall_quality_score:.3f}')
    print(f'ROUGE-1: {result.rouge_scores[\"rouge1\"]:.3f}')
    print(f'Coherence: {result.coherence_score:.3f}')

asyncio.run(test())
"
```

### **Option 2: Script-Based Execution**

**Quick Benchmark Run:**
```bash
python benchmarks/scripts/run_benchmarks.py --url http://localhost:8001 --quick
```

**Full Benchmark Suite:**
```bash
python benchmarks/scripts/run_benchmarks.py --url http://localhost:8001
```

**Custom Configuration:**
```bash
python benchmarks/scripts/run_benchmarks.py --url http://localhost:8001 --config benchmarks/configs/benchmark_config.yaml
```

### **Option 3: Programmatic Usage**

```python
import asyncio
from benchmarks import LatencyBenchmark, BaselineComparison, QualityBenchmark

async def comprehensive_benchmark():
    api_url = "http://localhost:8001"
    
    # Latency benchmark
    print("🏃 Running Latency Benchmark...")
    latency_bench = LatencyBenchmark(api_url)
    queries = ["What is data science?", "How does AI work?", "Explain machine learning"]
    latency_result = await latency_bench.benchmark_query_endpoint(queries, concurrent_users=3)
    
    # RAG vs Baseline comparison
    print("🥊 Running RAG vs Baseline Comparison...")
    comparison = BaselineComparison(api_url)
    comp_result = await comparison.compare_responses("What is artificial intelligence?")
    
    # Results summary
    print("\\n📊 BENCHMARK RESULTS:")
    print("=" * 50)
    print(f"⏱️  Average Latency: {latency_result.mean_latency:.3f}s")
    print(f"🚀 Throughput: {latency_result.throughput_rps:.2f} req/s")
    print(f"📈 P95 Latency: {latency_result.p95_latency:.3f}s")
    print(f"✅ Success Rate: {latency_result.success_rate:.1f}%")
    print(f"🎯 RAG Quality: {comp_result.rag_quality_score:.3f}")
    print(f"📊 Baseline Quality: {comp_result.baseline_quality_score:.3f}")
    print(f"📈 Improvement: {comp_result.improvement_score:.3f}")

# Run the benchmark
asyncio.run(comprehensive_benchmark())
```

## 📊 Viewing & Analyzing Results

### **Quick Results Check**
```bash
# List all result files
python benchmarks/scripts/view_results.py --list

# View latest results with summary
python benchmarks/scripts/view_results.py --latest

# View specific result file
python benchmarks/scripts/view_results.py --view latency_results_20250619_024822.json

# View detailed breakdown of results
python benchmarks/scripts/view_results.py --detailed filename.json

# Compare multiple benchmark runs
python benchmarks/scripts/view_results.py --compare

# Export results to CSV for analysis
python benchmarks/scripts/view_results.py --export filename.json
```

### **Direct File Access**
```bash
# Check results directory
ls -la benchmarks/results/

# View latest JSON results
find benchmarks/results/ -name "*.json" -type f -exec ls -lt {} + | head -5

# Quick search for specific metrics
grep -r "success_rate" benchmarks/results/
grep -r "mean_latency" benchmarks/results/

# View with JSON formatting (if jq installed)
cat benchmarks/results/latest_file.json | jq '.'
```

### **Understanding Results**

**Latency Results Example:**
```json
{
  "benchmark_info": {
    "timestamp": "2025-06-19T02:48:22.945170",
    "base_url": "http://localhost:8001",
    "benchmark_type": "latency_test"
  },
  "summary": {
    "latency": {
      "query_with_rag_avg_latency": 0.269,
      "p95_latency": 0.306,
      "throughput_rps": 15.53,
      "success_rate": 100.0
    }
  }
}
```

**Performance Benchmarks:**
- ✅ **Latency < 2s**: Excellent performance
- ✅ **Success Rate > 95%**: Good reliability  
- ✅ **Throughput > 5 req/s**: Adequate performance
- ✅ **Quality Score > 0.7**: High quality responses
- ✅ **RAG Improvement > 10%**: Effective RAG system

## ⚙️ Configuration & Customization

### **Built-in Configuration**
The system works with sensible defaults. For custom parameters:

```yaml
# benchmarks/configs/benchmark_config.yaml
latency:
  concurrent_users: [1, 3, 5, 10]
  requests_per_user: 5
  test_queries:
    - "What is artificial intelligence?"
    - "How does machine learning work?"
    - "Explain data science concepts"
    - "What are neural networks?"

accuracy:
  top_k_values: [1, 3, 5, 10]
  ground_truth_file: "benchmarks/data/ground_truth.json"
  semantic_similarity_threshold: 0.7

quality:
  test_cases_file: "benchmarks/data/quality_test_cases.json"
  evaluation_metrics: ["rouge", "semantic_similarity", "coherence"]
  use_llm_judge: true

comparison:
  test_queries_file: "benchmarks/data/comparison_queries.json"
  include_latency_comparison: true
```

### **Auto-Generated Test Data**
The system creates sample data automatically:

**Ground Truth Data** (`benchmarks/data/ground_truth.json`):
```json
[
  {
    "query": "What are the key principles of project management?",
    "relevant_doc_ids": ["doc_1", "doc_2"],
    "relevant_content": ["Project management involves...", "Key principles include..."],
    "query_intent": "factual",
    "difficulty": "easy"
  }
]
```

**Quality Test Cases** (`benchmarks/data/quality_test_cases.json`):
```json
[
  {
    "query": "Explain artificial intelligence",
    "reference_answer": "AI is technology that enables machines...",
    "context_documents": ["AI definition", "AI applications"],
    "evaluation_criteria": ["accuracy", "completeness"],
    "difficulty": "medium",
    "category": "educational"
  }
]
```

## 📈 Expected Performance Results

### **Typical Performance Ranges**

**Latency Performance:**
- **Excellent**: < 0.5s average latency
- **Good**: 0.5-2.0s average latency  
- **Acceptable**: 2.0-5.0s average latency
- **Poor**: > 5.0s average latency

**Throughput Performance:**
- **Excellent**: > 20 req/s
- **Good**: 10-20 req/s
- **Acceptable**: 5-10 req/s  
- **Poor**: < 5 req/s

**Quality Scores:**
- **Excellent**: > 0.8 overall quality
- **Good**: 0.6-0.8 overall quality
- **Acceptable**: 0.4-0.6 overall quality
- **Poor**: < 0.4 overall quality

### **Current Validated Performance**
Based on testing with the working system:
- ✅ **Latency**: ~0.27s (excellent)
- ✅ **Throughput**: ~15.5 req/s (good)
- ✅ **Success Rate**: 100% (perfect)
- ✅ **RAG Quality**: 0.3+ (functional)
- ✅ **System Status**: Fully operational

## 🔧 Troubleshooting Guide

### **Common Issues & Solutions**

1. **Import Errors**
   ```bash
   # Install all benchmark dependencies
   pip install -r benchmarks/requirements_benchmarks.txt
   
   # Verify installation
   python -c "import rouge_score, sentence_transformers; print('✅ Dependencies OK')"
   ```

2. **API Connection Issues**
   ```bash
   # Check if APIs are running
   curl http://localhost:8001/health  # RAG API
   curl http://localhost:8000         # ChromaDB
   curl http://localhost:11434/api/tags  # Ollama
   
   # Start services if needed
   python main.py                     # Direct Python
   docker compose up -d               # Docker services
   ollama serve                       # Local Ollama on Mac
   ```

3. **Docker + Local Ollama Connectivity**
   ```bash
   # Verify Docker can reach local Ollama
   docker exec -it <rag-api-container> curl http://host.docker.internal:11434/api/tags
   
   # Check docker-compose.yml has:
   # extra_hosts:
   #   - "host.docker.internal:host-gateway"
   
   # Check config.json has:
   # "llm_api_url": "http://host.docker.internal:11434/api/generate"
   ```

4. **BERTScore Warnings** (Safe to ignore)
   ```
   WARNING: Could not initialize BERTScore: We couldn't connect to 'https://huggingface.co'
   ```
   - This warning is normal when offline
   - Core benchmarks work without BERTScore
   - System uses alternative similarity metrics
   - Connect to internet for full BERTScore functionality

5. **No Results Found**
   ```bash
   # Run validation test first
   python benchmarks/scripts/test_benchmarks.py
   
   # Check results directory
   ls -la benchmarks/results/
   
   # Check permissions
   mkdir -p benchmarks/results && chmod 755 benchmarks/results
   ```

### **Performance Tips**

**For Faster Benchmarks:**
- Use `--quick` flag: `python benchmarks/scripts/run_benchmarks.py --quick`
- Reduce concurrent users: 1-3 instead of 5-10
- Test with fewer queries (2-3 instead of 5-10)
- Skip expensive metrics in configuration

**For Comprehensive Evaluation:**
- Run individual components separately for detailed analysis
- Use the full benchmark suite for complete assessment
- Enable internet connectivity for full BERTScore functionality
- Increase test dataset sizes for more robust metrics

### **Validation Checklist**

1. **✅ API Connectivity**: `curl http://localhost:8001/health`
2. **✅ Basic Functionality**: `python benchmarks/scripts/test_benchmarks.py`  
3. **✅ Dependencies**: `pip list | grep -E "(rouge|bert|sentence)"`
4. **✅ Results Directory**: `ls -la benchmarks/results/`
5. **✅ View Results**: `python benchmarks/scripts/view_results.py --list`

## 📚 Dependencies & Requirements

### **Core Libraries**
```txt
# HTTP client for API calls
httpx>=0.24.0

# Evaluation metrics
rouge-score>=0.1.2
bert-score>=0.3.13 
sentence-transformers>=2.2.0
scikit-learn>=1.3.0

# Statistical analysis  
numpy>=1.24.0
scipy>=1.10.0

# Optional NLP tools
nltk>=3.8
spacy>=3.6.0
```

### **Installation**
```bash
# Install all benchmark dependencies
pip install -r benchmarks/requirements_benchmarks.txt

# Verify installation
python -c "
try:
    import rouge_score, sentence_transformers, httpx, sklearn
    print('✅ All dependencies installed successfully')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
"
```

## 🤝 Contributing & Extension

### **Adding New Benchmark Metrics**

1. **Create new benchmark module** in `core/`:
```python
# benchmarks/core/custom_benchmark.py
class CustomBenchmark:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    async def run_custom_metric(self, query: str) -> float:
        # Your custom evaluation logic
        return metric_score
```

2. **Update imports** in `core/__init__.py`:
```python
from .custom_benchmark import CustomBenchmark
__all__.append('CustomBenchmark')
```

3. **Add to main package** in `__init__.py`:
```python
from .core.custom_benchmark import CustomBenchmark
```

### **Development Guidelines**

1. **Follow async/await patterns** for all API calls
2. **Add comprehensive logging** with proper levels
3. **Include error handling** with meaningful messages
4. **Use type hints** for better code clarity
5. **Write docstrings** for all public methods
6. **Update configuration options** for new parameters
7. **Add tests** in validation scripts
8. **Update documentation** in README

### **Testing New Features**
```bash
# Test individual components
python -c "from benchmarks.core.new_feature import NewFeature; print('✅ Import works')"

# Run validation suite
python benchmarks/scripts/test_benchmarks.py

# Check integration
python benchmarks/scripts/run_benchmarks.py --url http://localhost:8001 --quick
```

## 📋 Quick Reference Commands

### **Essential Commands**
```bash
# 🚀 Quick validation
python benchmarks/scripts/test_benchmarks.py

# 📊 Run benchmarks  
python benchmarks/scripts/run_benchmarks.py --url http://localhost:8001

# 📈 View results
python benchmarks/scripts/view_results.py --latest

# 📁 List all results
python benchmarks/scripts/view_results.py --list

# 📤 Export to CSV
python benchmarks/scripts/view_results.py --export filename.json
```

### **API Verification**
```bash
# Check all services
curl http://localhost:8001/health    # RAG API
curl http://localhost:8000           # ChromaDB  
curl http://localhost:11434/api/tags # Ollama

# Test RAG query
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "use_rag": false}'
```

### **Docker Commands**
```bash
# Start services
docker compose up -d chromadb rag-api

# Check logs
docker compose logs rag-api
docker compose logs chromadb

# Check status
docker compose ps
```