"""
Q&A Service with Dynamic RAG (Retrieval-Augmented Generation)

This service uses LangChain agents with dynamic system prompts that adapt
to each query type (counting, temporal, preference, factual, comparison).
"""

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.services.data_fetcher import data_fetcher
from app.services.vectorstore import vector_store
from app.services.tools import tools
from app.config import settings
import logging
import re

logger = logging.getLogger(__name__)


class QAService:
    """Question-Answering service using RAG and LangChain agents"""
    
    def __init__(self):
        self.messages = []
        self.llm = None
        self.agent_config = None
        self._initialized = False
    
    async def initialize(self):
        """Load messages, create vector store, and setup agent"""
        if self._initialized:
            logger.info("QA Service already initialized")
            return
        
        try:
            logger.info("Initializing QA Service...")
            
            # Load data
            logger.info("📥 Fetching messages...")
            data = await data_fetcher.fetch_messages()
            if not data:
                raise ValueError("No messages fetched from API")
            
            self.messages = data
            logger.info(f"✅ Loaded {len(self.messages)} messages")
            
            # Create vector store
            logger.info("🔍 Building vector store...")
            vector_store.initialize(self.messages)
            logger.info("✅ Vector store ready")
            
            # Setup agent
            logger.info("🤖 Setting up AI agent...")
            self._setup_agent()
            logger.info("✅ Agent ready")
            
            self._initialized = True
            logger.info("🎉 QA Service initialization complete!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize QA Service: {e}")
            raise
    
    def _setup_agent(self):
        """Setup the LangChain LLM and agent configuration"""
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            openai_api_key=settings.openai_api_key,
            top_p=0.9,
            frequency_penalty=0.3,
            presence_penalty=0.1
        )
        
        # Store base configuration
        self.agent_config = {
            "verbose": True,
            "max_iterations": 10,
            "max_execution_time": 90,
            "handle_parsing_errors": True,
            "return_intermediate_steps": False
        }
    
    def _build_system_prompt(self, query_analysis: dict) -> str:
        """
        Build a dynamic system prompt based on query analysis
        
        The prompt adapts to query type, complexity, and detected entities.
        """
        question_type = query_analysis.get('question_type', 'factual')
        complexity = query_analysis.get('complexity', 'medium')
        entities = query_analysis.get('entities', {})
        confidence = query_analysis.get('confidence', 0.5)
        optimal_k = query_analysis.get('optimal_k', 20)
        
        prompt_parts = []
        
        # Agent identity based on complexity
        identity_map = {
            'advanced': "You are an expert AI research assistant with multi-hop reasoning.",
            'complex': "You are an intelligent AI assistant skilled at analyzing multiple sources.",
            'medium': "You are a smart AI assistant that finds accurate information.",
            'simple': "You are a helpful AI assistant that answers questions accurately."
        }
        prompt_parts.append(identity_map.get(complexity, identity_map['medium']))
        
        # Tools description
        prompt_parts.append("""
AVAILABLE TOOLS:
- search_member_specific(member_name, topic, limit) - Search for specific person + topic
- search_member_by_name(member_name, limit) - Get all messages from a person
- search_messages(query, limit, member_name=None) - Semantic + keyword search
- get_all_members() - List all members

SEARCH TECHNOLOGY:
- PRIMARY: Embedding-based semantic search with COSINE SIMILARITY
- FALLBACK: Keyword/text matching (automatic)
- HYBRID: Combines both for best coverage
""")
        
        # Strategy based on question type
        strategy = self._get_strategy_guide(question_type, entities, optimal_k)
        prompt_parts.append(strategy)
        
        # Critical rules
        prompt_parts.append("""
⚠️ CRITICAL RULES:
- NO HALLUCINATION: Only use information explicitly in messages
- UNDERSTAND CONTEXT: "requests service" ≠ "owns", "interested in" ≠ "purchased"
- SPECIFIC OUTPUT: Use actual names, dates, counts from messages
- EXACT DATES: Calculate from timestamps (e.g., Oct 23 + "next month" = Nov)
- READ ALL RESULTS: Don't stop at first message
""")
        
        # Response style
        prompt_parts.append("""
💬 OUTPUT FORMAT:
- Natural, conversational tone
- Direct and confident with facts
- Specific: dates, counts, names
- NO backslashes, NO numbered lists, NO disclaimers
""")
        
        return "\n".join(prompt_parts)
    
    def _get_strategy_guide(self, question_type: str, entities: dict, optimal_k: int) -> str:
        """Get strategy guidance based on question type"""
        
        strategy_guides = {
            'counting': f"""
🎯 COUNTING STRATEGY:
1. Search: Use search_member_specific(person, topic, {optimal_k})
2. Read ALL messages carefully
3. Count UNIQUE items (ownership language like "has" or "owns")
4. DON'T count service requests or interests
5. Output: "X has N [items]: [list names]"
""",
            'temporal': f"""
🎯 DATE EXTRACTION STRATEGY:
1. Search for person + event/location
2. Check timestamps AND content for dates
3. Calculate exact dates from relative terms
4. Output specific date (e.g., "November 7, 2025" not "next month")
""",
            'preference': f"""
🎯 PREFERENCE STRATEGY:
1. Search for person + topic
2. Look for "favorite", "like", "prefer", specific names
3. List ALL specific items found
4. Output: Specific names, not generic descriptions
""",
            'comparison': """
🎯 COMPARISON STRATEGY:
1. Search for each entity separately
2. Extract relevant attributes
3. Compare side-by-side
4. Output: Clear comparison with facts
""",
            'factual': f"""
🎯 GENERAL STRATEGY:
1. Extract entities from question
2. Use search_member_specific if person+topic clear
3. Use search_messages for broader queries
4. Synthesize information from results
"""
        }
        
        return strategy_guides.get(question_type, strategy_guides['factual'])
    
    async def answer(self, question: str) -> str:
        """Answer a question using the AI agent with dynamic RAG"""
        if not self._initialized:
            raise RuntimeError("QA Service not initialized. Call initialize() first.")
        
        try:
            logger.info(f"🤔 Question: {question}")
            
            # Analyze query
            from app.services.dynamic_retrieval import dynamic_retrieval_service
            query_analysis = dynamic_retrieval_service.query_analyzer.analyze_query(question)
            
            logger.info(f"🧠 Query Analysis: type={query_analysis['question_type']}, "
                       f"complexity={query_analysis['complexity']}")
            
            # Build dynamic prompt
            dynamic_prompt_text = self._build_system_prompt(query_analysis)
            logger.info(f"📝 Generated dynamic prompt for {query_analysis['question_type']} query")
            
            # Create agent with dynamic prompt
            dynamic_prompt = ChatPromptTemplate.from_messages([
                ("system", dynamic_prompt_text),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            dynamic_agent = create_openai_functions_agent(self.llm, tools, dynamic_prompt)
            dynamic_executor = AgentExecutor(
                agent=dynamic_agent,
                tools=tools,
                **self.agent_config
            )
            
            # Enhance question
            enhanced_question = self._enhance_question(question, query_analysis)
            
            # Execute
            result = await dynamic_executor.ainvoke({"input": enhanced_question})
            answer = result.get("output", "I couldn't generate an answer.")
            
            # Clean answer
            answer = self._clean_answer(answer)
            
            # Retry if unhelpful
            if self._is_unhelpful_answer(answer):
                logger.warning(f"⚠️ Unhelpful answer, retrying with enhanced strategy...")
                answer = await self._retry_with_enhanced_strategy(
                    question, query_analysis, dynamic_executor
                )
            
            logger.info(f"✅ Answer generated: {answer[:100]}...")
            return answer
        
        except Exception as e:
            logger.error(f"❌ Error answering question: {e}", exc_info=True)
            return f"I encountered an error: {str(e)}. Please try rephrasing your question."
    
    def _enhance_question(self, question: str, analysis: dict) -> str:
        """Add context-specific instructions to the question"""
        entities = analysis.get('entities', {})
        question_type = analysis['question_type']
        
        enhancement = f"{question}\n\n"
        
        if entities:
            enhancement += f"🎯 DETECTED: {', '.join([f'{k}={v}' for k, v in entities.items()])}\n\n"
        
        # Type-specific reminders
        type_reminders = {
            'counting': "⚠️ COUNTING: Count UNIQUE items only, list specific names.\n",
            'temporal': "⚠️ DATE: Check timestamps AND content, calculate exact dates.\n",
            'preference': "⚠️ PREFERENCE: List ALL specific names found.\n"
        }
        
        if question_type in type_reminders:
            enhancement += type_reminders[question_type]
        
        enhancement += "\n✅ Follow the strategy in your instructions."
        
        return enhancement
    
    async def _retry_with_enhanced_strategy(
        self, question: str, analysis: dict, executor: AgentExecutor
    ) -> str:
        """Retry with enhanced search strategy"""
        try:
            # Increase search depth
            analysis['optimal_k'] = min(analysis.get('optimal_k', 20) * 2, 50)
            
            forced_question = f"""{question}

RETRY WITH ENHANCED STRATEGY:
- Use DIFFERENT search approaches
- Try broader searches with higher limits
- Be thorough and creative

CRITICAL: Extract EXACT information. Be conversational."""
            
            result = await executor.ainvoke({"input": forced_question})
            answer = result.get("output", "")
            answer = self._clean_answer(answer)
            
            if answer and not self._is_unhelpful_answer(answer):
                logger.info("✅ Retry succeeded")
                return answer
        except Exception as e:
            logger.error(f"Retry failed: {e}")
        
        return "I couldn't find enough information to answer this question accurately."
    
    def _clean_answer(self, answer: str) -> str:
        """Clean answer to make it conversational"""
        if not answer:
            return answer
        
        # Remove escape sequences
        answer = answer.replace('\\"', '"').replace("\\'", "'")
        answer = answer.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
        answer = answer.replace('\\', '')
        
        # Remove numbered lists
        answer = re.sub(r'\s*\d+\.\s+', ' ', answer)
        
        # Remove formal phrases
        formal_phrases = [
            (r'\bBased on the messages,?\s*', ''),
            (r'\bFrom the messages,?\s*', ''),
            (r'\bAccording to the messages,?\s*', ''),
            (r'\bIt appears that\s+', ''),
            (r'\bIt seems that\s+', ''),
            (r'\bit\'s important to note that\s+', ''),
        ]
        
        for pattern, replacement in formal_phrases:
            answer = re.sub(pattern, replacement, answer, flags=re.IGNORECASE)
        
        # Remove hedging
        hedging = [
            (r'\bmust be\b', 'is'),
            (r'\bmight be\b', 'is'),
            (r'\bcould be\b', 'is'),
            (r'\bseems to (have|be)\b', r'\1'),
        ]
        
        for pattern, replacement in hedging:
            answer = re.sub(pattern, replacement, answer, flags=re.IGNORECASE)
        
        # Clean up quotes and spaces
        answer = re.sub(r'"(next month|next week|Monday|Tuesday|Wednesday|Thursday|Friday)"', r'\1', answer)
        answer = re.sub(r'\s+', ' ', answer)
        answer = re.sub(r'\.\.+', '.', answer)
        
        # Capitalize and add period
        answer = answer.strip()
        if answer:
            answer = answer[0].upper() + answer[1:]
            if not answer[-1] in '.!?':
                answer += '.'
        
        return answer
    
    def _is_unhelpful_answer(self, answer: str) -> bool:
        """Check if answer is unhelpful"""
        unhelpful_phrases = [
            "couldn't find", "no messages", "no information",
            "does not provide", "hasn't shared", "not available"
        ]
        answer_lower = answer.lower()
        has_unhelpful = any(phrase in answer_lower for phrase in unhelpful_phrases)
        
        return has_unhelpful and len(answer) < 200
    
    async def reinitialize(self, force_fetch: bool = False):
        """Reinitialize the service"""
        logger.info("🔄 Reinitializing QA Service...")
        self._initialized = False
        
        if force_fetch:
            data = await data_fetcher.fetch_messages(use_cache=False)
            self.messages = data
        
        await self.initialize()
    
    def get_stats(self) -> dict:
        """Get statistics about the loaded data"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "ready",
            "total_messages": len(self.messages),
            "vectorstore_initialized": vector_store.vectorstore is not None,
            "unique_members": len(set(
                msg.get("user_name", "") 
                for msg in self.messages 
                if msg.get("user_name")
            ))
        }


# Global instance
qa_service = QAService()
