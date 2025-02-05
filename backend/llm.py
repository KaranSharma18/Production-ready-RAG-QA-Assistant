import os
from typing import List, Dict, Optional, Union
import json
import logging
from dataclasses import dataclass
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration class for LLM parameters"""
    model_name: str = "deepseek-r1:1.5b"
    history_limit: int = 15          # Increased to keep more conversation context
    max_context_length: int = 97304  # ~75% of 128K tokens for context (reserves space for system prompt and response)
    max_chunk_length: int = 2048     # Increased chunk size for more comprehensive context per chunk
    default_context: str = "No specific context found. I'll answer based on my general knowledge."
    temperature: float = 0.7
    retry_attempts: int = 3
    system_prompt_tokens: int = 1000  # Estimated tokens for system prompt
    max_history_tokens: int = 16384  # ~16K tokens reserved for chat history
    max_response_tokens: int = 8192  # Maximum expected response length
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load configuration from environment variables"""
        return cls(
            model_name=os.getenv('LLM_MODEL_NAME', "deepseek-r1:1.5b"),
            history_limit=int(os.getenv('LLM_HISTORY_LIMIT', 5)),
            max_context_length=int(os.getenv('LLM_MAX_CONTEXT_LENGTH', 97304)),
            max_chunk_length=int(os.getenv('LLM_MAX_CHUNK_LENGTH', 2048)),
            system_prompt_tokens=int(os.getenv('LLM_SYSTEM_PROMPT_TOKENS', 1000)),
            max_history_tokens=int(os.getenv('LLM_MAX_HISTORY_TOKENS', 16384)),
            max_response_tokens=int(os.getenv('LLM_MAX_RESPONSE_TOKENS', 8192)),
            temperature=float(os.getenv('LLM_TEMPERATURE', 0.7)),
            retry_attempts=int(os.getenv('LLM_RETRY_ATTEMPTS', 3))
        )

class LLMError(Exception):
    """Base exception class for LLM-related errors"""
    pass

class PromptBuilder:
    """Handles prompt construction and token management"""
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def format_chat_history(self, chat_history: Optional[List[str]]) -> str:
        """Format chat history for prompt inclusion"""
        if not chat_history:
            return ""
        
        formatted_history = []
        try:
            recent_chats = chat_history[-self.config.history_limit:]
            for chat in recent_chats:
                chat_entry = json.loads(chat)  # Safer than eval()
                formatted_history.append(
                    f"Human: {chat_entry['question']}\nAssistant: {chat_entry['answer']}"
                )
            return "\n".join(formatted_history)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error formatting chat history: {e}")
            return ""

    def truncate_context(self, context: Union[List[str], str]) -> str:
        """Truncate context to fit within token limit"""
        if isinstance(context, list):
            context = "\n".join(context)
        
        if not context:
            return self.config.default_context
            
        # Simple truncation strategy - could be improved with smarter chunking
        return context[:self.config.max_chunk_length]

    def build_prompt(self, query: str, context: Union[List[str], str], 
                    chat_history: Optional[List[str]] = None) -> str:
        """Build the complete prompt with history and context"""
        history_text = self.format_chat_history(chat_history)
        context_text = self.truncate_context(context)
        
        system_prompt = """You are a helpful AI assistant. Follow these guidelines:
1. Use the provided context from documents when available to answer questions
2. Consider the chat history for context and continuity
3. If no relevant context is found, answer based on general knowledge
4. Always be clear, concise, and accurate
5. If you're unsure, acknowledge the uncertainty"""

        prompt = f"""System: {system_prompt}

Past Conversation:
{history_text if history_text else "No prior conversation."}

Relevant Context:
{context_text}

Current Question: {query}

Please provide a helpful response based on the above context and conversation history."""
        
        return prompt


class LLMService:
    """Handles LLM interactions and response generation"""
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.prompt_builder = PromptBuilder(self.config)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _call_llm(self, prompt: str) -> str:
        """Make the actual LLM API call with retry logic"""
        try:
            response = ollama.chat(
                model=self.config.model_name,
                messages=[{"role": "system", "content": prompt, "temperature": self.config.temperature}]
                # temperature=self.config.temperature,
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise LLMError(f"Failed to generate response: {str(e)}")

    def generate_response(
        self,
        query: str,
        context: Union[List[str], str],
        chat_history: Optional[List[str]] = None
    ) -> str:
        """
        Generate a response using LLM with error handling and logging.
        
        Args:
            query: User's question
            context: Retrieved document chunks or context string
            chat_history: List of previous chat exchanges (optional)
            
        Returns:
            Generated response from LLM
            
        Raises:
            LLMError: If response generation fails
        """
        try:
            logger.info(f"Generating response for query: {query[:100]}...")
            
            prompt = self.prompt_builder.build_prompt(
                query=query,
                context=context,
                chat_history=chat_history
            )
            
            response = self._call_llm(prompt)
            
            logger.info(f"Successfully generated response for query: {query[:100]}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise LLMError(f"Failed to generate response: {str(e)}")