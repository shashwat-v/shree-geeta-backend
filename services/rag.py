from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone
from typing import List, Dict, Any, Optional, Generator
from functools import lru_cache
from datetime import datetime, timedelta
import logging
import os
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler('rag_service.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

class Config:
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "shree-geeta-ai")
    
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "shreegeeta2")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "3"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_MIN_WAIT: int = int(os.getenv("RETRY_MIN_WAIT", "1"))
    RETRY_MAX_WAIT: int = int(os.getenv("RETRY_MAX_WAIT", "10"))
    
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "128"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    
    @classmethod
    def validate(cls) -> None:
        required_vars = {
            "PINECONE_API_KEY": cls.PINECONE_API_KEY,
            "GROQ_API_KEY": cls.GROQ_API_KEY,
            "OPENAI_API_KEY": cls.OPENAI_API_KEY,
        }
        
        missing_vars = [key for key, value in required_vars.items() if not value]
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Environment variables validated successfully")

try:
    Config.validate()
except ValueError as e:
    logger.error(f"Configuration validation failed: {e}")
    raise

os.environ["LANGCHAIN_TRACING_V2"] = Config.LANGCHAIN_TRACING_V2
if Config.LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = Config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = Config.LANGCHAIN_PROJECT

class RAGException(Exception):
    pass

class EmbeddingException(RAGException):
    pass

class RetrievalException(RAGException):
    pass

class LLMException(RAGException):
    pass

class VectorStoreException(RAGException):
    pass

class SimpleCache:
    def __init__(self, max_size: int = 128, ttl_seconds: int = 3600):
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        logger.info(f"Cache initialized with max_size={max_size}, ttl={ttl_seconds}s")
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                logger.debug(f"Cache hit for key: {key[:50]}...")
                return value
            else:
                del self.cache[key]
                logger.debug(f"Cache expired for key: {key[:50]}...")
        return None
    
    def set(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            logger.debug("Cache full, removed oldest entry")
        
        self.cache[key] = (value, datetime.now())
        logger.debug(f"Cache set for key: {key[:50]}...")
    
    def clear(self) -> None:
        self.cache.clear()
        logger.info("Cache cleared")

class RAGService:
    def __init__(self):
        logger.info("Initializing RAG Service...")
        
        self.cache = SimpleCache(
            max_size=Config.CACHE_MAX_SIZE,
            ttl_seconds=Config.CACHE_TTL_SECONDS
        ) if Config.ENABLE_CACHE else None
        
        self._initialized = False
        self._init_components()
        
        logger.info("RAG Service initialized successfully")
    
    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(
            min=Config.RETRY_MIN_WAIT,
            max=Config.RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True
    )
    def _init_components(self) -> None:
        try:
            logger.info("Connecting to Pinecone...")
            self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
            self.index = self.pc.Index(Config.PINECONE_INDEX_NAME)
            
            stats = self.index.describe_index_stats()
            logger.info(
                f"Connected to Pinecone index '{Config.PINECONE_INDEX_NAME}' "
                f"with {stats.total_vector_count} vectors"
            )
            
            logger.info(f"Initializing embeddings model: {Config.EMBEDDING_MODEL}")
            self.embeddings = OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                openai_api_key=Config.OPENAI_API_KEY
            )
            
            logger.info("Setting up vector store...")
            self.vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                text_key="text"
            )
            
            self.retriever = VectorStoreRetriever(
                vectorstore=self.vectorstore,
                search_kwargs={"k": Config.RETRIEVAL_K}
            )
            
            logger.info(f"Initializing LLM: {Config.LLM_MODEL}")
            self.llm = ChatGroq(
                model=Config.LLM_MODEL,
                api_key=Config.GROQ_API_KEY,
                streaming=True,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS
            )
            
            self.prompt = self._create_prompt_template()
            
            self.rag_chain = self._build_rag_chain()
            
            self._initialized = True
            logger.info("All RAG components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
            raise RAGException(f"RAG initialization failed: {str(e)}")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        template = """You are Shree Geeta AI, a knowledgeable guide on the Bhagavad Gita.

Use the provided context from the Bhagavad Gita to answer the question accurately and meaningfully.

Respond in plain text format. DO NOT return JSON. DO NOT use curly braces or code blocks.

Format your response EXACTLY like this:

• Summary meaning: <1-2 line concise summary>

• Relevant Verses:
  - Chapter X Verse Y: <one line essence of the verse>
  - Chapter X Verse Y: <one line essence of the verse>
  (include 2-3 most relevant verses only)

• Explanation for modern practical life:
<5-10 lines of practical, relatable explanation with real-world examples from today's life. Make it actionable and inspiring.>

CONTEXT FROM BHAGAVAD GITA:
{context}

USER QUESTION:
{question}

Remember: Keep it practical, inspirational, and easy to understand. Connect ancient wisdom to modern life.
"""
        return ChatPromptTemplate.from_template(template)
    
    def _format_docs(self, docs: List[Document]) -> str:
        if not docs:
            logger.warning("No documents retrieved from vector store")
            return "No relevant context found."
        
        formatted = "\n\n".join([
            f"[Source {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])
        
        logger.debug(f"Formatted {len(docs)} documents for context")
        return formatted
    
    def _build_rag_chain(self):
        return (
            RunnableParallel({
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            })
            | self.prompt
            | self.llm
        )
    
    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RAGException("RAG Service not initialized")
    
    def answer_question(
        self,
        question: str,
        use_cache: bool = True
    ) -> str:
        self._check_initialized()
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        question = question.strip()
        logger.info(f"Answering question: {question[:100]}...")
        
        if use_cache and self.cache:
            cached_answer = self.cache.get(question)
            if cached_answer:
                logger.info("Returning cached answer")
                return cached_answer
        
        try:
            start_time = datetime.now()
            response = self.rag_chain.invoke(question)
            duration = (datetime.now() - start_time).total_seconds()
            
            answer = response.content
            
            if use_cache and self.cache:
                self.cache.set(question, answer)
            
            logger.info(f"Answer generated successfully in {duration:.2f}s")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            raise LLMException(f"Failed to generate answer: {str(e)}")
    
    def stream_answer(
        self,
        question: str,
        use_cache: bool = True
    ) -> Generator[str, None, None]:
        self._check_initialized()
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        question = question.strip()
        logger.info(f"Streaming answer for question: {question[:100]}...")
        
        if use_cache and self.cache:
            cached_answer = self.cache.get(question)
            if cached_answer:
                logger.info("Returning cached answer (streaming)")
                for chunk in cached_answer.split():
                    yield chunk + " "
                return
        
        try:
            start_time = datetime.now()
            full_answer = []
            
            for chunk in self.rag_chain.stream(question):
                if chunk.content:
                    full_answer.append(chunk.content)
                    yield chunk.content
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if use_cache and self.cache and full_answer:
                complete_answer = "".join(full_answer)
                self.cache.set(question, complete_answer)
            
            logger.info(f"Answer streamed successfully in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error streaming answer: {e}", exc_info=True)
            raise LLMException(f"Failed to stream answer: {str(e)}")
    
    def retrieve_context(
        self,
        question: str,
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        self._check_initialized()
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            search_kwargs = {"k": k if k else Config.RETRIEVAL_K}
            
            logger.info(f"Retrieving context for: {question[:100]}...")
            docs = self.vectorstore.similarity_search(question, **search_kwargs)
            
            results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": None
                }
                for doc in docs
            ]
            
            logger.info(f"Retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            raise RetrievalException(f"Failed to retrieve context: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on RAG service
        
        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "initialized": self._initialized,
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        try:
            # Check Pinecone
            if self._initialized:
                stats = self.index.describe_index_stats()
                health["components"]["pinecone"] = {
                    "status": "healthy",
                    "index": Config.PINECONE_INDEX_NAME,
                    "vector_count": stats.total_vector_count
                }
            else:
                health["components"]["pinecone"] = {"status": "not_initialized"}
            
            # Check embeddings
            health["components"]["embeddings"] = {
                "status": "healthy" if self._initialized else "not_initialized",
                "model": Config.EMBEDDING_MODEL
            }
            
            # Check LLM
            health["components"]["llm"] = {
                "status": "healthy" if self._initialized else "not_initialized",
                "model": Config.LLM_MODEL
            }
            
            # Check cache
            if self.cache:
                health["components"]["cache"] = {
                    "status": "healthy",
                    "size": len(self.cache.cache),
                    "max_size": self.cache.max_size
                }
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health
    
    def clear_cache(self) -> None:
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared by user request")
        else:
            logger.warning("Cache is disabled, nothing to clear")

try:
    rag_service = RAGService()
    logger.info("Global RAG service instance created")
except Exception as e:
    logger.critical(f"Failed to create global RAG service: {e}", exc_info=True)
    raise

rag_chain = rag_service.rag_chain
answer_question = rag_service.answer_question

def ask(question: str, use_cache: bool = True) -> str:
    return rag_service.answer_question(question, use_cache=use_cache)

def stream(question: str, use_cache: bool = True) -> Generator[str, None, None]:
    return rag_service.stream_answer(question, use_cache=use_cache)

def get_context(question: str, k: int = 3) -> List[Dict[str, Any]]:
    return rag_service.retrieve_context(question, k=k)

def check_health() -> Dict[str, Any]:
    return rag_service.health_check()

if __name__ == "__main__":
    test_question = "How to handle stress and failure in life?"
    
    print("=" * 80)
    print("Testing RAG Service")
    print("=" * 80)
    print(f"\nQuestion: {test_question}\n")
    print("-" * 80)
    print("Streaming Answer:")
    print("-" * 80)
    
    try:
        for chunk in stream(test_question):
            print(chunk, end="", flush=True)
        print("\n")
        print("-" * 80)
        
        print("\nHealth Check:")
        print("-" * 80)
        import json
        print(json.dumps(check_health(), indent=2))
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\nError: {e}")