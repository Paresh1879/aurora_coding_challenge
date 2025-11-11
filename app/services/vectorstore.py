"""
Vector Store Service

Manages FAISS vector store with:
- OpenAI embeddings for semantic search
- Cosine similarity for better contextual understanding
- Automatic keyword fallback when embeddings don't match well
- Persistent caching for performance
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from app.config import settings
from typing import List
import os
import hashlib
import json


class VectorStore:
    """FAISS vector store with intelligent hybrid search"""
    
    def __init__(self, persist_directory: str = "./vectorstore_cache"):
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
        self.vectorstore = None
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
    
    def initialize(self, messages: List[dict], use_cache: bool = True):
        """Create vector store from messages with optional caching"""
        messages_hash = self._get_messages_hash(messages)
        cache_path = self._get_cache_path(messages_hash)
        
        # Try loading from cache
        if use_cache and self._load_from_cache(cache_path, messages_hash):
            print(f"Vector store loaded from cache with {len(messages)} documents")
            return
        
        # Create new index
        print("Creating embeddings...")
        docs = self._messages_to_documents(messages)
        
        # Create FAISS index with cosine similarity
        self.vectorstore = FAISS.from_documents(
            docs, 
            self.embeddings,
            distance_strategy="COSINE"
        )
        print(f"Vector store ready with {len(docs)} documents (cosine similarity)")
        
        # Save to cache
        if use_cache:
            self._save_to_cache(cache_path, messages_hash, len(messages))
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Intelligent hybrid search
        
        Starts with embedding-based semantic search.
        Automatically falls back to keyword matching if embeddings yield poor results.
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        # Primary: semantic search with scores
        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        # Filter for good matches (cosine > 0.7 or distance < 0.5)
        good_results = [
            (doc, score) for doc, score in results_with_scores 
            if score > 0.7 or score < 0.5
        ]
        
        if len(good_results) >= k:
            return [doc for doc, _ in good_results[:k]]
        
        # Fallback: keyword search
        print(f"⚠️ Embedding search: {len(good_results)} good results, adding keyword fallback")
        keyword_results = self._keyword_search(query, k)
        
        # Combine and deduplicate
        return self._combine_results(good_results, keyword_results, k)
    
    def search_with_scores(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Document]:
        """Search with similarity scores"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        if score_threshold > 0:
            results = [(doc, score) for doc, score in results if score >= score_threshold]
        
        return [doc for doc, _ in results]
    
    def multi_query_search(self, queries: List[str], k_per_query: int = 10, max_total: int = 30) -> List[Document]:
        """Search with multiple query variations and combine results"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        all_results = []
        seen_ids = set()
        
        for query in queries:
            results = self.vectorstore.similarity_search(query, k=k_per_query)
            for doc in results:
                doc_id = self._get_doc_id(doc)
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results.append(doc)
                    if len(all_results) >= max_total:
                        return all_results[:max_total]
        
        return all_results
    
    # Private helper methods
    
    def _messages_to_documents(self, messages: List[dict]) -> List[Document]:
        """Convert messages to LangChain documents"""
        docs = []
        for msg in messages:
            user_name = msg.get('user_name', 'unknown')
            user_id = msg.get('user_id', 'unknown')
            message = msg.get('message', '')
            timestamp = msg.get('timestamp', '')
            
            # Rich text content for better embeddings
            text = f"User: {user_name} (ID: {user_id})\nTimestamp: {timestamp}\nMessage: {message}"
            
            doc = Document(
                page_content=text,
                metadata={
                    "user_id": user_id,
                    "user_name": user_name,
                    "timestamp": timestamp,
                    "message_id": msg.get("id", "")
                }
            )
            docs.append(doc)
        
        return docs
    
    def _keyword_search(self, query: str, k: int) -> List[Document]:
        """Keyword-based fallback search"""
        query_words = set(word.lower() for word in query.split() if len(word) > 2)
        
        if not query_words:
            return []
        
        # Get more candidates for filtering
        all_docs = self.vectorstore.similarity_search(query, k=k*5)
        
        # Score by keyword matching
        scored_docs = []
        for doc in all_docs:
            matches = sum(1 for word in query_words if word in doc.page_content.lower())
            if matches > 0:
                score = matches / len(query_words)
                scored_docs.append((doc, score))
        
        # Return top k by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]
    
    def _combine_results(
        self, 
        good_results: List[tuple], 
        keyword_results: List[Document], 
        k: int
    ) -> List[Document]:
        """Combine and deduplicate results from multiple sources"""
        combined = list(good_results) + [(doc, 0.0) for doc in keyword_results]
        
        seen_ids = set()
        unique_results = []
        
        for doc, _ in combined:
            doc_id = self._get_doc_id(doc)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(doc)
                if len(unique_results) >= k:
                    break
        
        return unique_results
    
    def _get_doc_id(self, doc: Document) -> str:
        """Get unique ID for document"""
        doc_id = doc.metadata.get("message_id", "") or doc.metadata.get("user_id", "")
        if not doc_id:
            doc_id = str(hash(doc.page_content))
        return doc_id
    
    def _get_messages_hash(self, messages: List[dict]) -> str:
        """Generate hash for cache validation"""
        message_ids = [msg.get("id", "") for msg in messages[:100]]
        content = f"{len(messages)}:{':'.join(sorted(message_ids))}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, messages_hash: str) -> str:
        """Get cache file path"""
        return os.path.join(self.persist_directory, f"faiss_index_{messages_hash}")
    
    def _load_from_cache(self, cache_path: str, messages_hash: str) -> bool:
        """Try loading from cache"""
        cache_info_path = os.path.join(self.persist_directory, "cache_info.json")
        
        if not (os.path.exists(cache_path) and os.path.exists(cache_info_path)):
            return False
        
        try:
            with open(cache_info_path, 'r') as f:
                cache_info = json.load(f)
            
            if cache_info.get("messages_hash") == messages_hash:
                print(f"Loading vectorstore from cache: {cache_path}")
                self.vectorstore = FAISS.load_local(
                    cache_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
        except Exception as e:
            print(f"Failed to load from cache: {e}")
        
        return False
    
    def _save_to_cache(self, cache_path: str, messages_hash: str, message_count: int):
        """Save to cache"""
        try:
            print(f"Saving vectorstore to cache: {cache_path}")
            self.vectorstore.save_local(cache_path)
            
            cache_info_path = os.path.join(self.persist_directory, "cache_info.json")
            with open(cache_info_path, 'w') as f:
                json.dump({
                    "messages_hash": messages_hash,
                    "message_count": message_count
                }, f)
            
            print("Vectorstore cached successfully")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")


# Global instance
vector_store = VectorStore()
