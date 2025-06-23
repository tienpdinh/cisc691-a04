# Error Analysis System

Comprehensive error analysis and failure tracking system for the RAG pipeline. Provides real-time error classification, component health monitoring, failure prediction, and system analytics.

## 🚀 Features

- **Real-time Error Classification** - Automatically categorizes errors by severity and type
- **Component Health Monitoring** - Tracks health scores for each RAG component  
- **Failure Prediction** - Predicts potential failures based on error patterns
- **Error Recovery** - Automatic retry, fallback, and circuit breaker mechanisms
- **Performance Tracking** - Monitors response times, success rates, throughput
- **Comprehensive Analytics** - Detailed error analysis and trend reporting
- **Alert System** - Configurable thresholds and notifications
- **Easy Integration** - Decorator-based monitoring for existing code
- **Persistent Storage** - SQLite database with export capabilities

## 📁 Structure

```
error_analysis/
├── README.md                    # This documentation
├── __init__.py                  # Package initialization
├── requirements_error_analysis.txt # Dependencies
├── core/                        # Core error analysis components
│   ├── __init__.py
│   ├── error_classifier.py     # Error classification and categorization
│   ├── error_logger.py         # Error logging and storage
│   ├── failure_tracker.py      # Failure mode tracking and prediction
│   └── error_analysis_manager.py # Central coordination system
├── recovery/                    # Error recovery mechanisms
│   ├── __init__.py
│   └── error_recovery.py       # Recovery strategies and circuit breakers
├── monitors/                    # Component-specific monitors
│   ├── __init__.py
│   ├── base_monitor.py         # Base monitoring functionality
│   ├── api_endpoint_monitor.py # FastAPI endpoint monitoring
│   ├── llm_client_monitor.py   # LLM client monitoring
│   ├── cache_monitor.py        # Redis cache monitoring
│   ├── vector_store_monitor.py # ChromaDB monitoring
│   └── document_processor_monitor.py # Document processing monitoring
├── tests/                       # Comprehensive test suite
│   ├── __init__.py
│   └── test_error_analysis.py  # All error analysis tests
├── scripts/                     # Utility scripts
│   ├── __init__.py
│   ├── run_error_analysis.py   # Main analysis runner
│   ├── view_results.py         # Results viewer
│   ├── test_error_analysis.py  # Test runner
│   ├── integration_example.py  # Integration examples
│   └── live_rag_monitoring.py  # Live RAG system monitoring
├── middleware/                  # RAG API integration
│   ├── __init__.py
│   └── rag_api_middleware.py   # FastAPI middleware integration
├── config/                      # Configuration files
│   └── error_analysis_config.json # Main configuration
├── data/                        # Sample data and patterns
│   ├── error_patterns.json     # Common error patterns
│   └── sample_config.json      # Sample configuration
└── results/                     # Analysis results and logs
    └── (generated result files)
```

## 🏁 Quick Start

### 1. Basic Setup

```python
from error_analysis import get_error_analysis_manager

# Initialize error analysis system
error_manager = get_error_analysis_manager()

# Track an error
try:
    # Your code here
    pass
except Exception as e:
    error_manager.track_error(e, component="llm_client", context={"query": "test"})
```

### 2. Monitor Functions with Decorators

```python
# Get component monitor
llm_monitor = error_manager.get_monitor('llm_client')

# Decorate functions for automatic monitoring
@llm_monitor.monitor_function('llm_query')
def query_llm(prompt):
    # Your LLM code here
    return llm_response

# Now all calls to query_llm() are automatically monitored
```

### 3. Get System Health Status

```python
# Get comprehensive health status
health = error_manager.get_system_health_status()
print(f"System status: {health['overall_status']}")
print(f"Components monitored: {len(health['component_health'])}")

# Get detailed analysis
analysis = error_manager.get_comprehensive_analysis(hours=24)
print(f"Total errors (24h): {analysis['error_analysis']['total_errors']}")
print(f"Recommendations: {analysis['recommendations']}")

# Use error recovery
recovery_manager = error_manager.get_recovery_manager()
recovery_stats = recovery_manager.get_recovery_statistics(hours=24)
print(f"Recovery success rate: {recovery_stats['overall_success_rate']:.2%}")
```

### 4. Run Analysis Scripts

```bash
# Run comprehensive error analysis
python error_analysis/scripts/run_error_analysis.py

# Run quick analysis (last hour only)
python error_analysis/scripts/run_error_analysis.py --quick

# View latest results
python error_analysis/scripts/view_results.py --latest

# List available result files
python error_analysis/scripts/view_results.py --list

# Test the error analysis system
python error_analysis/scripts/test_error_analysis.py

# Run quick functionality test
python error_analysis/scripts/test_error_analysis.py --quick
```

**Script Options:**

- `run_error_analysis.py`: Main analysis runner
  - `--quick`: Run 1-hour analysis
  - `--hours N`: Analyze last N hours  
  - `--output DIR`: Save results to directory
  - `--no-save`: Don't save results to file

- `view_results.py`: Results viewer
  - `--list`: List available result files
  - `--latest`: View most recent results
  - `--compare FILE1 FILE2`: Compare multiple results
  - `--summary-only`: Show only summary

- `test_error_analysis.py`: System testing
  - `--quick`: Quick functionality test
  - `--errors N`: Generate N test errors

- `live_rag_monitoring.py`: Live RAG system monitoring
  - `--duration N`: Monitor for N minutes
  - `--interval N`: Query interval in seconds

## 🚀 Live RAG System Monitoring

### Quick Integration Steps

#### Step 1: Add Middleware to Your RAG API

Add this to your `main.py` or wherever you initialize your FastAPI app:

```python
from error_analysis.middleware import integrate_error_analysis_with_rag_api

# Initialize your FastAPI app
app = FastAPI(title="RAG API")

# Add error analysis integration
integrate_error_analysis_with_rag_api(app)
```

#### Step 2: Monitor Individual Components (Optional)

In your route handlers, use the monitoring decorators:

```python
@app.post("/query")
async def rag_query(request: QueryRequest):
    # Get monitoring decorators
    decorators = request.app.state.error_analysis['decorators']
    
    try:
        # Monitor LLM calls
        @decorators['llm']
        def call_ollama(prompt):
            return ollama_client.generate(prompt)
        
        # Monitor vector searches
        @decorators['vector']
        def search_chroma(embedding):
            return chroma_client.query(query_embeddings=[embedding])
        
        # Monitor cache operations
        @decorators['cache']
        def get_from_redis(key):
            return redis_client.get(key)
        
        # Use monitored functions
        response = call_ollama(prompt)
        # ... rest of your logic
        
        return {"response": response}
        
    except Exception as e:
        # Errors are automatically tracked by middleware
        raise
```

### Run Live System Monitoring

Once your containers are running:

```bash
# Start your RAG system
docker-compose up -d
ollama serve

# Run live monitoring (monitors real API calls)
python error_analysis/scripts/live_rag_monitoring.py --duration 5 --interval 2

# Extended monitoring (10 minutes, every 1 second)
python error_analysis/scripts/live_rag_monitoring.py --duration 10 --interval 1
```

### What Live Monitoring Does

1. **Checks Service Availability**
   - ✅ RAG API (port 8000)
   - ✅ Redis (port 6379) 
   - ✅ ChromaDB (port 8001)
   - ✅ Ollama (port 11434)

2. **Tests Real Components**
   - 🔍 Makes actual API calls to your RAG endpoints
   - 🗄️ Tests Redis cache operations
   - 📊 Tests ChromaDB vector operations
   - 🤖 Tests Ollama model availability

3. **Captures Real Errors**
   - API timeout errors
   - HTTP error responses
   - Component failures
   - Performance issues

4. **Generates Performance Metrics**
   - Response times
   - Success rates
   - Error rates
   - Component health scores

### Manual Testing Workflow

#### 1. Start Your RAG System
```bash
docker-compose up -d
ollama serve
```

#### 2. Verify Services
```bash
# Check RAG API
curl http://localhost:8000/health

# Check ChromaDB  
curl http://localhost:8001/api/v1/heartbeat

# Check Ollama
curl http://localhost:11434/api/tags
```

#### 3. Run Live Monitoring
```bash
# Quick test (2 minutes)
python error_analysis/scripts/live_rag_monitoring.py --duration 2

# Full test (10 minutes)  
python error_analysis/scripts/live_rag_monitoring.py --duration 10
```

#### 4. Test Manual API Calls
```bash
# Test your RAG API manually while monitoring is running
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is project management?"}'
```

#### 5. View Results
```bash
# View monitoring results
python error_analysis/scripts/view_results.py --latest

# Check error analysis health
curl http://localhost:8000/health/error-analysis
```

### Generated Files

Live monitoring creates:

```
error_analysis/
├── results/
│   ├── live_monitoring_20250622_143052.json  # Live monitoring results
│   └── error_analysis_20250622_143100.json   # Error analysis report
└── logs/
    └── errors/
        └── errors_20250622.log                # Error logs
```

### For Assignment Submission

This live monitoring provides:

1. **Real Error Analysis** ✅
   - Actual errors from your RAG system
   - Performance bottlenecks
   - Component failures

2. **Performance Metrics** ✅
   - Response times under load
   - Success/failure rates
   - Component health scores

3. **Production Readiness** ✅
   - Error recovery mechanisms
   - Circuit breakers in action
   - System monitoring capabilities

4. **Documentation** ✅
   - Real system behavior analysis
   - Error categorization
   - Performance optimization recommendations

**This gives you comprehensive error analysis data for your assignment submission!**

## 🔧 Integration with RAG Components

### FastAPI Integration

```python
from fastapi import FastAPI, Request, HTTPException
import time

app = FastAPI()
error_manager = get_error_analysis_manager()
api_monitor = error_manager.get_monitor('api_endpoints')

@app.middleware("http")
async def error_analysis_middleware(request: Request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        response_time = time.time() - start_time
        
        # Track successful request
        api_monitor.track_request(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=response.status_code,
            response_time=response_time
        )
        
        return response
        
    except Exception as e:
        response_time = time.time() - start_time
        status_code = e.status_code if isinstance(e, HTTPException) else 500
        
        # Track failed request
        api_monitor.track_request(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=status_code,
            response_time=response_time,
            error_details={'exception': str(e)}
        )
        
        # Track error in system
        error_manager.track_error(e, 'api_endpoints')
        raise

# Add health check endpoint
@app.get("/health/error-analysis")
async def error_analysis_health():
    return error_manager.get_system_health_status()
```

### LLM Client Integration

```python
from error_analysis import get_error_analysis_manager

class MonitoredLLMClient:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.error_manager = get_error_analysis_manager()
        self.monitor = self.error_manager.get_monitor('llm_client')
    
    @property
    def query(self):
        return self.monitor.monitor_function('llm_query')(self.llm_client.query)
    
    def track_model_loading(self, model_name, loading_time, success):
        if hasattr(self.monitor, 'track_model_loading'):
            self.monitor.track_model_loading(model_name, loading_time, success)
```

### Cache Integration

```python
class MonitoredCacheManager:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.error_manager = get_error_analysis_manager()
        self.monitor = self.error_manager.get_monitor('cache_manager')
    
    @property
    def get(self):
        return self.monitor.monitor_function('cache_get')(self.cache_manager.get)
    
    @property 
    def set(self):
        return self.monitor.monitor_function('cache_set')(self.cache_manager.set)
```

## 📊 Monitoring and Analytics

### Health Check Endpoints

Add these endpoints to your FastAPI app for monitoring:

```python
@app.get("/health/error-analysis")
async def error_analysis_health():
    """Get current system health status."""
    return error_manager.get_system_health_status()

@app.get("/health/error-analysis/comprehensive")
async def comprehensive_analysis():
    """Get detailed 24-hour analysis."""
    return error_manager.get_comprehensive_analysis(hours=24)

@app.get("/health/error-analysis/component/{component_name}")
async def component_health(component_name: str):
    """Get health status for specific component."""
    monitor = error_manager.get_monitor(component_name)
    if monitor:
        return monitor.get_health_status()
    return {"error": "Component not found"}
```

### Error Categories and Severity Levels

**Severity Levels:**
- `CRITICAL` - System down, data corruption, security issues
- `HIGH` - Component failures, performance issues
- `MEDIUM` - Quality degradation, processing errors
- `LOW` - Validation warnings, configuration issues

**Error Categories:**
- `SYSTEM_DOWN` - Complete system failures
- `DATA_CORRUPTION` - Data integrity issues
- `SECURITY` - Authentication/authorization failures
- `RETRIEVAL_FAILURE` - Vector search/embedding issues
- `GENERATION_FAILURE` - LLM response issues
- `PERFORMANCE` - Latency/throughput problems
- `QUALITY_DEGRADATION` - Response quality issues
- `PROCESSING_ERRORS` - Document processing failures
- `CACHE_ISSUES` - Redis cache problems
- `VALIDATION_WARNINGS` - Input validation issues
- `CONFIGURATION` - Config/environment problems
- `MONITORING` - Metrics collection issues

## 🔄 Error Recovery Mechanisms

The system includes comprehensive error recovery capabilities with multiple strategies:

### Recovery Strategies

1. **Retry with Exponential Backoff**
   - Automatically retries failed operations
   - Configurable max attempts and delays
   - Jitter to prevent thundering herd

2. **Circuit Breaker Pattern**
   - Prevents cascade failures
   - Opens circuit after threshold failures
   - Auto-recovery testing

3. **Fallback Responses**
   - Graceful degradation
   - Pre-configured fallback responses
   - Component-specific handlers

4. **Cache Fallbacks**
   - Serve cached responses on errors
   - Reduce dependency on external services
   - Maintain service availability

### Using Error Recovery

```python
from error_analysis import with_error_recovery, get_error_analysis_manager

# Get recovery manager
manager = get_error_analysis_manager()
recovery_manager = manager.get_recovery_manager()

# Use decorator for automatic recovery
@with_error_recovery('llm_client', recovery_manager)
def query_llm(prompt):
    # Your LLM code here
    return llm_response

# Manual recovery
try:
    # Some operation
    result = risky_operation()
except Exception as e:
    recovery_result = manager.track_error(e, 'component_name', context)
    if recovery_result:
        result = recovery_result
    else:
        # Handle unrecoverable error
        raise

# Register custom fallback
def custom_fallback(error_info, context):
    return {'fallback_response': 'Custom handling'}

manager.register_recovery_fallback('my_component', custom_fallback)
```

### Recovery Configuration

Configure recovery behavior in `error_analysis_config.json`:

```json
{
  "error_recovery": {
    "retry": {
      "max_attempts": 3,
      "base_delay": 1.0,
      "max_delay": 60.0
    },
    "circuit_breaker": {
      "failure_threshold": 5,
      "recovery_timeout": 60.0
    },
    "fallback_strategies": {
      "llm_client": ["retry", "cache_fallback", "fallback"]
    }
  }
}
```

## ⚙️ Configuration

Configure the system using `config/error_analysis_config.json`:

```json
{
  "error_analysis": {
    "enabled": true,
    "log_directory": "logs/errors",
    "max_memory_entries": 1000,
    "alert_thresholds": {
      "error_rate_per_minute": 5,
      "avg_response_time_seconds": 5.0,
      "success_rate_threshold": 0.95
    },
    "component_monitoring": {
      "llm_client": {
        "enabled": true,
        "model_loading_timeout": 30,
        "prompt_processing_timeout": 60
      },
      "cache_manager": {
        "enabled": true,
        "cache_hit_rate_threshold": 0.7,
        "memory_usage_threshold": 0.9
      }
    }
  }
}
```

## 🧪 Testing

Run the error analysis tests:

```bash
# Test specific components
python -m pytest tests/test_error_analysis.py::TestErrorClassifier -v
python -m pytest tests/test_error_analysis.py::TestErrorAnalysisManager -v

# Test full integration
python -m pytest tests/test_error_analysis.py::test_full_error_analysis_workflow -v

# Run all error analysis tests
python -m pytest tests/test_error_analysis.py -v
```

## 📈 Data Export and Analysis

```python
# Export comprehensive data
export_data = error_manager.export_analysis_data(hours=24, format="json")

# Get error trends
trends = error_manager.error_logger.get_error_trends(hours=168)  # 1 week

# Get component performance
for component_name in ['llm_client', 'api_endpoints', 'cache_manager']:
    monitor = error_manager.get_monitor(component_name)
    performance = monitor.get_performance_trends(hours=24)
    print(f"{component_name}: {performance}")
```

## 🚨 Alerts and Notifications

The system automatically generates alerts for:
- High error rates (>5 errors/minute)
- Poor success rates (<95%)
- Slow response times (>5 seconds)
- Component health degradation
- Cascade failures
- Performance anomalies

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Make sure error_analysis is in your Python path
   import sys
   sys.path.append('/path/to/your/project')
   from error_analysis import get_error_analysis_manager
   ```

2. **Database Issues**
   ```python
   # Check if logs directory exists and is writable
   import os
   log_dir = "logs/errors"
   os.makedirs(log_dir, exist_ok=True)
   ```

3. **Memory Usage**
   ```python
   # Reduce memory usage if needed
   config = {
       'max_memory_entries': 500,  # Reduce from default 1000
       'max_failure_events': 5000  # Reduce from default 10000
   }
   error_manager = get_error_analysis_manager(config)
   ```

### Performance Tips

- Use decorators for automatic monitoring instead of manual tracking
- Configure appropriate alert thresholds for your use case
- Regularly export and clean up old error data
- Monitor the error analysis system's own performance

## 📝 Contributing

When adding new error types or monitors:

1. Add error patterns to `error_classifier.py`
2. Create component-specific monitors in `monitors/`
3. Update configuration schema in `config/`
4. Add tests in `tests/test_error_analysis.py`
5. Update this README with usage examples

## 🔗 Integration with Existing Systems

This error analysis system is designed to work alongside the existing benchmarking system in the `benchmarks/` folder. While benchmarks provide performance measurement, error analysis provides real-time monitoring and failure detection.

Both systems can be used together for comprehensive system observability.

## 📋 Dependencies

### Base Requirements
```bash
pip install -r error_analysis/requirements_error_analysis.txt
```

### Live Monitoring Additional Requirements
```bash
pip install -r error_analysis/requirements_live_monitoring.txt
```

Required packages for live monitoring:
- `aiohttp>=3.8.0` - Async HTTP client for API monitoring
- `requests>=2.28.0` - HTTP requests for service checks
- `redis>=4.5.0` - Redis client for cache monitoring