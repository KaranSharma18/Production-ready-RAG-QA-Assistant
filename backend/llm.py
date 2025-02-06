import os
from typing import List, Dict, Optional, Union
import json
from dataclasses import dataclass
import ollama
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, AsyncRetrying
from contextlib import asynccontextmanager
from logger_config import logger
from prompt_manager import PromptManager
from config import get_settings, Settings

class LLMError(Exception):
    """Base exception class for LLM-related errors"""
    pass

class PromptBuilder:
    """Handles prompt construction and token management"""
    def __init__(self, config: Settings, prompt_manager: PromptManager):
        self.config = config
        self.prompt_manager = prompt_manager
    
    async def format_chat_history(self, chat_history: Optional[List[str]]) -> str:
        """Format chat history for prompt inclusion"""
        if not chat_history:
            return ""
        
        formatted_history = []
        try:
            recent_chats = chat_history[-self.config.llm_history_limit:]
            for chat in recent_chats:
                chat_entry = json.loads(chat)
                formatted_history.append(
                    f"Human: {chat_entry['question']}\nAssistant: {chat_entry['answer']}"
                )
            return "\n".join(formatted_history)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error formatting chat history: {e}")
            return ""

    async def truncate_context(self, context: Union[List[str], str]) -> str:
        """Truncate context to fit within token limit"""
        if isinstance(context, list):
            context = "\n".join(context)
        
        if not context:
            return self.config.default_llm_context
            
        return context[:self.config.llm_max_context_length]

    async def build_prompt(self, 
                         query: str,
                         context: Union[List[str], str],
                         chat_history: Optional[List[str]] = None) -> str:
        """Build the complete prompt with history and context"""
        history_text = await self.format_chat_history(chat_history)
        context_text = await self.truncate_context(context)
        
        return self.prompt_manager.format_prompt(
            query=query,
            context=context_text,
            history_text=history_text
        )

class LLMService:
    """Handles LLM interactions and response generation"""
    def __init__(self, prompt_dir: Optional[str] = None):
        self.config = get_settings()
        self.prompt_manager = PromptManager(prompt_dir)
        self.prompt_builder = PromptBuilder(self.config, self.prompt_manager)
        self._semaphore = asyncio.Semaphore(self.config.llm_concurrent_requests)

    async def _call_llm(self, prompt: str) -> str:
        """Make the actual LLM API call with retry logic"""
        try:
            # Use semaphore to limit concurrent requests
            async with self._semaphore:
                response = await asyncio.to_thread(
                    ollama.chat,
                    model=self.config.llm_model_name,
                    messages=[{
                        "role": "system",
                        "content": prompt,
                        "llm_temperature": self.config.llm_temperature
                    }]
                )
                return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise LLMError(f"Failed to generate response: {str(e)}")
    
    async def call_llm_with_retry(self, prompt: str) -> str:
        """Method that applies retry decorator to LLM call"""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.config.llm_retry_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=self.config.min_retry_wait,
                max=self.config.max_retry_wait
            ),
            reraise=True
        ):
            with attempt:  # Changed from async with to with
                return await self._call_llm(prompt)

    async def generate_response(
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
            
            prompt = await self.prompt_builder.build_prompt(
                query=query,
                context=context,
                chat_history=chat_history
            )
            
            response = await self.call_llm_with_retry(prompt)
            
            logger.info(f"Successfully generated response for query: {query[:100]}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise LLMError(f"Failed to generate response: {str(e)}")

# Async context manager for LLM service
@asynccontextmanager
async def get_llm_service(prompt_dir: Optional[str] = None):
    """Async context manager for LLM service"""
    service = LLMService(prompt_dir)
    try:
        yield service
    finally:
        # Cleanup code if needed
        pass