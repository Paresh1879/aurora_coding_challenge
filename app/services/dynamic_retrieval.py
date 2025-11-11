"""
Dynamic Retrieval System for Adaptive RAG

Implements intelligent retrieval strategies that adapt to query type and complexity:
- Query analysis (type, complexity, entities)
- Query expansion (synonyms, variations)
- Hybrid search (semantic + keyword)
- Reranking based on relevance
"""

from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from app.services.vectorstore import vector_store
import re
from datetime import datetime
import logging

try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyzes queries to determine optimal retrieval strategy"""
    
    def __init__(self):
        self.question_patterns = {
            'counting': [r'how many', r'count', r'number of', r'total'],
            'temporal': [r'when', r'what time', r'which day', r'what date', r'next', r'last'],
            'preference': [r'favorite', r'prefer', r'like', r'best', r'love'],
            'comparison': [r'compare', r'versus', r'vs', r'difference', r'better'],
            'factual': [r'what', r'who', r'where', r'how', r'does', r'is', r'has']
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query and return strategy recommendations"""
        query_lower = query.lower()
        
        return {
            "question_type": self._classify_type(query_lower),
            "complexity": self._assess_complexity(query),
            "entities": self._extract_entities(query),
            "optimal_k": self._determine_k(query_lower),
            "needs_temporal_context": "when" in query_lower or "date" in query_lower,
            "needs_exact_match": "how many" in query_lower or "count" in query_lower
        }
    
    def _classify_type(self, query_lower: str) -> str:
        """Classify question type"""
        if "how many" in query_lower or "count" in query_lower:
            return "counting"
        elif "when" in query_lower or "date" in query_lower:
            return "temporal"
        elif "favorite" in query_lower or "prefer" in query_lower:
            return "preference"
        elif "compare" in query_lower or "versus" in query_lower:
            return "comparison"
        else:
            return "factual"
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity based on structure"""
        words = query.split()
        entities = len([w for w in words if w[0].isupper()])
        clauses = query.count(" and ") + query.count(" or ")
        
        if entities > 2 or clauses > 1 or len(words) > 10:
            return "complex"
        elif entities > 1 or clauses > 0 or len(words) > 6:
            return "medium"
        else:
            return "simple"
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities (names, locations, topics) from query"""
        words = query.split()
        
        # Capitalized words (likely names)
        capitalized = [w for w in words if w and w[0].isupper()]
        
        # Known locations
        locations = ["london", "paris", "tokyo", "new york", "barcelona", "sydney"]
        found_locations = [w for w in words if w.lower() in locations]
        
        # Member names (capitalized, not locations)
        member_names = [w for w in capitalized if w.lower() not in locations]
        
        # Topics (non-capitalized, meaningful words)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "have", "has", 
                    "do", "does", "will", "can", "to", "of", "in", "on", "at", "for"}
        topics = [w.lower() for w in words if w.lower() not in stopwords and w[0].islower()]
        
        return {
            "member_names": member_names,
            "locations": found_locations,
            "topics": topics
        }
    
    def _determine_k(self, query_lower: str) -> int:
        """Determine optimal number of results to retrieve"""
        if "how many" in query_lower or "count" in query_lower:
            return 25
        elif "when" in query_lower or "date" in query_lower:
            return 20
        elif "favorite" in query_lower or "prefer" in query_lower:
            return 20
        else:
            return 15


class QueryExpander:
    """Expands queries with synonyms and variations"""
    
    def __init__(self):
        self.synonyms = {
            'cars': ['vehicles', 'automobiles', 'car', 'auto'],
            'restaurants': ['dining', 'food', 'restaurant', 'eatery'],
            'favorite': ['preferred', 'best', 'likes'],
            'travel': ['trip', 'journey', 'vacation', 'visit'],
        }
    
    def expand_query(self, query: str, entities: Dict) -> List[str]:
        """Generate query variations"""
        queries = [query]
        
        # Add member name variations
        if entities.get("member_names"):
            for name in entities["member_names"]:
                if " " in name:
                    # Add first name only variation
                    first_name = name.split()[0]
                    queries.append(query.replace(name, first_name))
        
        # Add topic synonyms
        query_lower = query.lower()
        for key, synonyms in self.synonyms.items():
            if key in query_lower:
                for synonym in synonyms[:2]:  # Limit to 2 synonyms
                    queries.append(query.lower().replace(key, synonym))
        
        return list(set(queries))[:10]  # Max 10 variations


class Reranker:
    """Reranks documents based on relevance to query"""
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        """Rerank documents by relevance score"""
        if not documents or len(documents) <= top_k:
            return documents
        
        # Limit input for efficiency
        documents = documents[:35]
        
        # Score each document
        scored_docs = [(doc, self._score(query, doc)) for doc in documents]
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def _score(self, query: str, doc: Document) -> float:
        """Calculate relevance score"""
        score = 0.0
        query_lower = query.lower()
        content_lower = doc.page_content.lower()
        
        # Keyword overlap (40%)
        query_words = set(w for w in query_lower.split() if len(w) > 2)
        content_words = set(w for w in content_lower.split() if len(w) > 2)
        if query_words:
            overlap = len(query_words & content_words)
            score += 0.4 * (overlap / len(query_words))
        
        # Entity matching (30%)
        query_entities = [w for w in query.split() if w and w[0].isupper()]
        content_entities = [w for w in doc.page_content.split() if w and w[0].isupper()]
        if query_entities:
            entity_overlap = len(set(query_entities) & set(content_entities))
            score += 0.3 * (entity_overlap / len(query_entities))
        
        # User name match (20%)
        user_name = doc.metadata.get("user_name", "").lower()
        if any(name.lower() in query_lower for name in user_name.split()):
            score += 0.2
        
        # Content quality (10%)
        if len(doc.page_content) > 100:
            score += 0.1
        
        return min(score, 1.0)


class DynamicRetrievalService:
    """Main service orchestrating dynamic retrieval"""
    
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.query_expander = QueryExpander()
        self.reranker = Reranker()
        self.vector_store = vector_store
    
    def retrieve(
        self,
        query: str,
        member_name: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> List[Document]:
        """
        Dynamically retrieve relevant documents
        
        Args:
            query: User query
            member_name: Optional member filter
            context: Optional context from previous interactions
        
        Returns:
            Ranked list of relevant documents
        """
        # Analyze query
        analysis = self.query_analyzer.analyze_query(query)
        logger.info(f"Query analysis: {analysis['question_type']}, {analysis['complexity']}")
        
        # Expand query
        expanded_queries = self.query_expander.expand_query(query, analysis["entities"])
        
        # Retrieve documents
        optimal_k = min(analysis["optimal_k"], 25)
        
        if analysis["complexity"] in ["complex", "advanced"]:
            documents = self._multi_query_retrieval(expanded_queries, optimal_k)
            documents = self.reranker.rerank(query, documents, optimal_k)
        elif len(expanded_queries) > 1:
            documents = self._multi_query_retrieval(expanded_queries, optimal_k)
        else:
            documents = self.vector_store.search(query, k=optimal_k)
        
        # Apply filters
        if member_name:
            documents = self._filter_by_member(documents, member_name)
        
        if analysis["needs_temporal_context"]:
            documents = self._apply_temporal_ordering(documents)
        
        return documents
    
    def _multi_query_retrieval(self, queries: List[str], k: int) -> List[Document]:
        """Retrieve with multiple query variations"""
        return self.vector_store.multi_query_search(
            queries, 
            k_per_query=max(3, k // len(queries) + 2), 
            max_total=k
        )
    
    def _filter_by_member(self, documents: List[Document], member_name: str) -> List[Document]:
        """Filter documents by member name (flexible matching)"""
        member_lower = member_name.lower()
        filtered = []
        
        for doc in documents:
            doc_name = doc.metadata.get("user_name", "").lower()
            if member_lower in doc_name or doc_name in member_lower:
                filtered.append(doc)
        
        return filtered if filtered else documents
    
    def _apply_temporal_ordering(self, documents: List[Document]) -> List[Document]:
        """Sort documents by timestamp (most recent first)"""
        def get_timestamp(doc):
            timestamp_str = doc.metadata.get("timestamp", "")
            if not timestamp_str:
                return datetime.min
            
            try:
                return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except:
                if date_parser:
                    try:
                        return date_parser.parse(timestamp_str)
                    except:
                        pass
                return datetime.min
        
        return sorted(documents, key=get_timestamp, reverse=True)


# Global instance
dynamic_retrieval_service = DynamicRetrievalService()
