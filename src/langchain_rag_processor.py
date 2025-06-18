import logging
import os
from typing import Optional, Dict, Any, List, Tuple
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from .langchain_vector_store import LangChainVectorStore


class LangChainRAGProcessor:
    """
    LangChain-based RAG processor that handles retrieval and generation.
    Replaces custom RAG query processing logic.
    """
    
    def __init__(self,
                 vector_store: LangChainVectorStore,
                 llm_provider: str = "ollama",
                 llm_model_name: str = "llama3.2",
                 llm_api_url: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 max_context_length: int = 4000):
        """
        Initialize the LangChain RAG processor.
        
        :param vector_store: LangChainVectorStore instance
        :param llm_provider: LLM provider ("ollama" or "openai")
        :param llm_model_name: Name of the LLM model
        :param llm_api_url: API URL for Ollama
        :param openai_api_key: OpenAI API key
        :param max_context_length: Maximum context length for LLM
        """
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.llm_model_name = llm_model_name
        self.max_context_length = max_context_length
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM
        self._initialize_llm(llm_api_url, openai_api_key)
        
        # Initialize prompt template
        self._initialize_prompt_template()
        
        # Create RAG chain
        self._create_rag_chain()
        
        self.logger.info(f"Initialized LangChainRAGProcessor: provider={llm_provider}, "
                         f"model={llm_model_name}")
    
    def _initialize_llm(self, llm_api_url: Optional[str], openai_api_key: Optional[str]):
        """Initialize the LLM based on provider."""
        try:
            if self.llm_provider == "ollama":
                if llm_api_url:
                    # Handle different URL formats for Ollama
                    if llm_api_url.endswith("/api/generate"):
                        base_url = llm_api_url.replace("/api/generate", "")
                    elif llm_api_url.endswith("/"):
                        base_url = llm_api_url.rstrip("/")
                    else:
                        base_url = llm_api_url
                    
                    self.llm = OllamaLLM(
                        model=self.llm_model_name,
                        base_url=base_url
                    )
                else:
                    # Get Ollama URL from environment if not provided
                    llm_api_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
                    base_url = llm_api_url.split('/api/generate')[0].rstrip('/')
                    
                    self.llm = OllamaLLM(
                        model=self.llm_model_name,
                        base_url=base_url
                    )
                
                self.logger.info(f"Initialized Ollama LLM: {self.llm_model_name}")
                
            elif self.llm_provider == "openai":
                self.llm = ChatOpenAI(
                    model=self.llm_model_name,
                    api_key=openai_api_key,
                    temperature=0.7
                )
                self.logger.info(f"Initialized OpenAI LLM: {self.llm_model_name}")
                
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _initialize_prompt_template(self):
        """Initialize the RAG prompt template."""
        self.prompt_template = ChatPromptTemplate.from_template("""
You are an AI assistant answering user queries using retrieved context from documents.
Use the provided context to answer the question accurately and comprehensively.
If the context is insufficient to answer the question, say "I don't have enough information to answer this question based on the provided context."

Context:
{context}

Question: {question}

Answer:""")
        
        self.logger.info("Initialized RAG prompt template")
    
    def _create_rag_chain(self):
        """Create the LangChain RAG processing chain."""
        def format_docs(docs: List[Document]) -> str:
            """Format retrieved documents into context string."""
            return "\n\n".join([doc.page_content for doc in docs])
        
        # Create the RAG chain
        self.rag_chain = (
            {
                "context": lambda x: format_docs(
                    self.vector_store.similarity_search(x["question"], k=4)
                ),
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        self.logger.info("Created LangChain RAG processing chain")
    
    def query(self, question: str, use_rag: bool = True) -> dict:
        """
        Process a query using RAG or direct LLM.
        
        :param question: User question
        :param use_rag: Whether to use RAG (retrieve context) or direct LLM
        :return: Dictionary with response and metadata
        """
        try:
            if use_rag:
                return self._query_with_rag(question)
            else:
                return self._query_without_rag(question)
                
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "response": f"Error processing query: {str(e)}",
                "context_used": False,
                "retrieved_docs": [],
                "error": str(e)
            }
    
    def _query_with_rag(self, question: str) -> dict:
        """Process query with RAG (retrieval + generation)."""
        self.logger.info("Processing query with RAG")
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search_with_scores(question, k=4)
        
        if not retrieved_docs:
            self.logger.warning("No relevant documents found")
            return {
                "response": "I don't have any relevant information to answer this question.",
                "context_used": False,
                "retrieved_docs": [],
                "scores": []
            }
        
        # Log retrieved documents
        for i, (doc, score) in enumerate(retrieved_docs):
            self.logger.info(f"Retrieved doc {i+1}: score={score:.3f}, "
                           f"content_length={len(doc.page_content)}")
        
        # Format context
        context = "\n\n".join([doc.page_content for doc, _ in retrieved_docs])
        
        # Truncate context if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
            self.logger.info(f"Truncated context to {self.max_context_length} characters")
        
        # Generate response using RAG chain
        response = self.rag_chain.invoke({"question": question})
        
        self.logger.info("Generated RAG response")
        
        return {
            "response": response,
            "context_used": True,
            "retrieved_docs": [doc.page_content for doc, _ in retrieved_docs],
            "scores": [score for _, score in retrieved_docs],
            "context_length": len(context)
        }
    
    def _query_without_rag(self, question: str) -> dict:
        """Process query without RAG (direct LLM)."""
        self.logger.info("Processing query without RAG")
        
        # Direct LLM query
        try:
            response = self.llm.invoke(question)
        except ConnectionError as e:
            return {
                "response": "Error processing query: Unable to connect to LLM service",
                "error": f"Connection error: {str(e)}",
                "context_used": False,
                "retrieved_docs": [],
                "scores": []
            }
        except TimeoutError as e:
            return {
                "response": "Error processing query: LLM service request timed out",
                "error": f"Timeout error: {str(e)}",
                "context_used": False,
                "retrieved_docs": [],
                "scores": []
            }
        except Exception as e:
            return {
                "response": "Error processing query",
                "error": str(e),
                "context_used": False,
                "retrieved_docs": [],
                "scores": []
            }
        
        self.logger.info("Generated direct LLM response")
        
        return {
            "response": response,
            "context_used": False,
            "retrieved_docs": [],
            "scores": []
        }
    
    def retrieve_documents(self, query: str, k: int = 4, score_threshold: Optional[float] = None) -> List[dict]:
        """
        Retrieve relevant documents for a query.
        
        :param query: Search query
        :param k: Number of documents to retrieve
        :param score_threshold: Minimum similarity score threshold
        :return: List of document dictionaries with content and metadata
        """
        try:
            docs_and_scores = self.vector_store.similarity_search_with_scores(query, k=k)
            
            # Filter by score threshold if provided
            if score_threshold is not None:
                docs_and_scores = [
                    (doc, score) for doc, score in docs_and_scores 
                    if score >= score_threshold
                ]
            
            # Format results
            results = []
            for doc, score in docs_and_scores:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
            
            self.logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return []