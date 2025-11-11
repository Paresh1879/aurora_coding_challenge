"""
Search Tools for LangChain Agent

Provides semantic search, member-specific search, and member listing tools.
Uses hybrid search (embedding + keyword) with dynamic retrieval strategies.
"""

try:
    from langchain.tools import tool
except ImportError:
    from langchain_core.tools import tool

from app.services.vectorstore import vector_store
from app.services.dynamic_retrieval import dynamic_retrieval_service
from typing import List
from langchain.docstore.document import Document
import logging

logger = logging.getLogger(__name__)


def _format_search_results(documents: List[Document], query: str, limit: int, 
                           question_type: str = None) -> str:
    """
    Format search results with appropriate guidance
    
    Args:
        documents: Retrieved documents
        query: Search query
        limit: Max results to show
        question_type: Type of question (counting, temporal, preference, etc.)
    """
    if not documents:
        return f"No messages found matching: '{query}'"
    
    # Truncate content for token efficiency
    max_content_length = 400 if question_type == 'counting' else 300
    formatted = []
    
    for i, doc in enumerate(documents[:limit], 1):
        user_name = doc.metadata.get("user_name", "unknown")
        timestamp = doc.metadata.get("timestamp", "unknown")
        content = doc.page_content
        
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        formatted.append(f"{i}. {user_name} @ {timestamp}: {content}")
    
    # Build header with guidance
    header = f"Found {len(documents)} messages:\n"
    header += "=" * 60 + "\n"
    
    if question_type == 'counting':
        header += "🔍 COUNTING: Look for specific items. Count carefully.\n"
    elif question_type == 'temporal':
        header += "🔍 DATE EXTRACTION: Check timestamps AND content for dates.\n"
    elif question_type == 'preference':
        header += "🔍 PREFERENCE: Find specific names mentioned.\n"
    else:
        header += "IMPORTANT: Read ALL messages below carefully.\n"
    
    header += "=" * 60 + "\n\n"
    
    return header + "\n".join(formatted)


def _extract_names_from_query(query: str) -> List[str]:
    """Extract capitalized names from query"""
    words = query.split()
    return [w for w in words if w and w[0].isupper()]


def _match_member_name(doc_name: str, search_name: str) -> bool:
    """
    Flexible member name matching
    
    Handles variations like:
    - "Layla" matching "Layla Kawaguchi"
    - First name matches
    - Partial matches
    """
    doc_lower = doc_name.lower().strip()
    search_lower = search_name.lower().strip()
    
    # Exact match
    if doc_lower == search_lower:
        return True
    
    # One contains the other
    if search_lower in doc_lower or doc_lower in search_lower:
        return True
    
    # First name match
    doc_parts = doc_lower.split()
    search_parts = search_lower.split()
    
    if doc_parts and search_parts and doc_parts[0] == search_parts[0]:
        return True
    
    return False


@tool
def search_messages(query: str, limit: int = 25, member_name: str = None) -> str:
    """Search messages. query: search text, limit: max results (max 25), member_name: optional filter."""
    try:
        limit = max(5, min(limit, 25))
        
        # Use dynamic retrieval for intelligent search
        try:
            results = dynamic_retrieval_service.retrieve(
                query=query,
                member_name=member_name,
                context=None
            )
            results = results[:limit]
        except Exception as e:
            logger.warning(f"Dynamic retrieval failed, using fallback: {e}")
            results = vector_store.search(query, k=limit)
        
        # Filter by member name if provided
        if member_name and results:
            results = [
                doc for doc in results 
                if _match_member_name(doc.metadata.get("user_name", ""), member_name)
            ]
        
        # If no results after filtering, try broader search
        if not results and member_name:
            broader_results = vector_store.search(member_name, k=limit)
            results = [
                doc for doc in broader_results 
                if _match_member_name(doc.metadata.get("user_name", ""), member_name)
            ]
        
        if not results:
            suffix = f" for member '{member_name}'" if member_name else ""
            return f"No messages found matching: '{query}'{suffix}"
        
        return _format_search_results(results, query, limit)
    
    except Exception as e:
        return f"Error searching messages: {str(e)}"


@tool
def get_all_members() -> str:
    """Get list of all members in dataset."""
    try:
        # Search broadly to find all members
        broad_queries = ["member", "user", "people"]
        all_results = vector_store.multi_query_search(broad_queries, k_per_query=30, max_total=100)
        
        members = set()
        for doc in all_results:
            user_name = doc.metadata.get("user_name", "").strip()
            if user_name and user_name.lower() not in ["unknown", "anonymous", ""]:
                members.add(user_name)
        
        if not members:
            return "Could not retrieve member list from the dataset."
        
        member_list = sorted(list(members))
        return f"Members in dataset ({len(member_list)} found):\n" + "\n".join([f"- {m}" for m in member_list])
    
    except Exception as e:
        return f"Error getting member list: {str(e)}"


@tool
def search_member_by_name(member_name: str, limit: int = 25) -> str:
    """Find member by name. Returns messages from matching members."""
    try:
        limit = max(10, min(limit, 25))
        
        # Build query variations
        queries = [member_name, f"User: {member_name}", f"{member_name} said"]
        
        # Add first name variation
        if " " in member_name:
            first_name = member_name.split()[0]
            queries.extend([first_name, f"User: {first_name}"])
        
        # Search with multiple queries
        all_results = vector_store.multi_query_search(
            queries, 
            k_per_query=max(15, limit // len(queries) + 10), 
            max_total=limit * 3
        )
        
        # Group by matching user names
        name_variations = {}
        for doc in all_results:
            user_name = doc.metadata.get("user_name", "").strip()
            if user_name and _match_member_name(user_name, member_name):
                if user_name not in name_variations:
                    name_variations[user_name] = []
                name_variations[user_name].append(doc)
        
        if not name_variations:
            return f"No messages found for member '{member_name}'. Try get_all_members() to see available names."
        
        # Format results
        formatted_results = []
        for exact_name, docs in name_variations.items():
            formatted_results.append(f"\n{'='*60}")
            formatted_results.append(f"MEMBER: {exact_name} ({len(docs)} messages)")
            formatted_results.append(f"{'='*60}")
            
            for i, doc in enumerate(docs[:15], 1):
                timestamp = doc.metadata.get("timestamp", "unknown")
                content = doc.page_content
                if len(content) > 400:
                    content = content[:400] + "..."
                formatted_results.append(f"  {i}. @ {timestamp}: {content}")
        
        result_text = f"Found {len(name_variations)} member(s) matching '{member_name}':\n"
        result_text += "IMPORTANT: Use the EXACT MEMBER NAME shown above in your next search!\n"
        result_text += "\n".join(formatted_results)
        result_text += f"\n\nEXACT NAME TO USE: {list(name_variations.keys())[0]}"
        
        return result_text
    
    except Exception as e:
        return f"Error searching for member '{member_name}': {str(e)}"


@tool
def search_member_specific(member_name: str, topic: str, limit: int = 25) -> str:
    """
    Search for member and topic. Filters to exact member.
    
    USE THIS FIRST for person-specific questions.
    """
    try:
        limit = max(10, min(limit, 25))
        
        # Build comprehensive query variations
        queries = [
            f"{member_name} {topic}",
            f"{topic}",
            f"{member_name} mentioned {topic}",
            f"{member_name} has {topic}",
        ]
        
        # Add first name variations
        if " " in member_name:
            first_name = member_name.split()[0]
            queries.extend([f"{first_name} {topic}"])
        
        # Add topic-specific synonyms
        topic_lower = topic.lower()
        if "car" in topic_lower:
            queries.extend([f"{member_name} vehicle", f"{member_name} automobile"])
        elif "restaurant" in topic_lower:
            queries.extend([f"{member_name} dining", f"{member_name} food"])
        elif "trip" in topic_lower or "travel" in topic_lower:
            queries.extend([f"{member_name} travel", f"{member_name} journey"])
        
        # Multi-strategy search
        all_results = []
        seen_ids = set()
        
        # Strategy 1: Multi-query search
        try:
            multi_results = vector_store.multi_query_search(
                queries, 
                k_per_query=max(10, limit // len(queries) + 8), 
                max_total=limit * 3
            )
            for doc in multi_results:
                doc_id = doc.metadata.get("message_id", "") or str(hash(doc.page_content))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results.append(doc)
        except Exception as e:
            logger.warning(f"Multi-query search failed: {e}")
        
        # Strategy 2: Dynamic retrieval
        try:
            dynamic_results = dynamic_retrieval_service.retrieve(
                query=f"{member_name} {topic}",
                member_name=None,
                context={"topic": topic}
            )
            for doc in dynamic_results[:limit * 2]:
                doc_id = doc.metadata.get("message_id", "") or str(hash(doc.page_content))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results.append(doc)
        except Exception as e:
            logger.warning(f"Dynamic retrieval failed: {e}")
        
        # Filter to matching member name
        filtered_results = [
            doc for doc in all_results 
            if _match_member_name(doc.metadata.get("user_name", ""), member_name)
        ]
        
        # Fallback: broader search if no results
        if not filtered_results:
            broad_results = vector_store.search(member_name, k=100)
            filtered_results = [
                doc for doc in broad_results 
                if _match_member_name(doc.metadata.get("user_name", ""), member_name)
            ]
        
        if not filtered_results:
            return f"No messages found about '{topic}' for '{member_name}'. Try search_member_by_name('{member_name}', 25) to find the exact name."
        
        # Determine question type for formatting
        question_type = None
        if "car" in topic_lower or "how many" in topic_lower:
            question_type = 'counting'
        elif "trip" in topic_lower or "when" in topic_lower:
            question_type = 'temporal'
        elif "restaurant" in topic_lower or "favorite" in topic_lower:
            question_type = 'preference'
        
        # Format results
        max_results = min(limit, 30) if question_type == 'counting' else min(limit, 25)
        
        header = f"Search Results for '{member_name}' about '{topic}':\n"
        header += "=" * 70 + "\n"
        
        if question_type == 'counting':
            header += "🔍 COUNTING: Look for specific item names. Count carefully.\n"
        elif question_type == 'temporal':
            header += "🔍 DATE: Check timestamps AND content for exact dates.\n"
        elif question_type == 'preference':
            header += "🔍 PREFERENCE: List specific names mentioned.\n"
        
        header += "=" * 70 + "\n\n"
        
        formatted = []
        for i, doc in enumerate(filtered_results[:max_results], 1):
            timestamp = doc.metadata.get("timestamp", "unknown")
            user_name = doc.metadata.get("user_name", "unknown")
            content = doc.page_content
            
            max_len = 400 if question_type == 'counting' else 300
            if len(content) > max_len:
                content = content[:max_len] + "..."
            
            formatted.append(f"{i}. {user_name} @ {timestamp}: {content}")
        
        return header + "\n".join(formatted)
    
    except Exception as e:
        return f"Error searching for {member_name} and {topic}: {str(e)}"


# Export tools
tools = [search_messages, search_member_specific, search_member_by_name, get_all_members]
