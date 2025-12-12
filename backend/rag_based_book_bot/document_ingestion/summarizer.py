"""
Section-level summarizer for RAG system

This module provides abstractive summarization of book sections using LLMs.
Supports multiple LLM providers: Ollama (local), Groq (cloud), and Gemini (cloud).
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@dataclass
class SummaryConfig:
    """Configuration for summarizer"""
    model: str = "ollama/llama3.2:3b"  # Default to Ollama
    temperature: float = 0.3
    min_section_tokens: int = 400  # Only summarize sections with this many tokens
    max_summary_tokens: int = 200  # Target summary length
    batch_size: int = 10  # Process sections in batches
    max_retries: int = 3
    retry_delay: float = 1.0


class Summarizer:
    """
    Abstractive summarizer for book sections
    
    Generates concise summaries of book sections while preserving key information.
    Supports multiple LLM backends for flexibility.
    """
    
    def __init__(self, config: SummaryConfig):
        """Initialize summarizer with LLM"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Detect LLM provider from model name
        model = config.model.lower()
        
        if model.startswith("ollama/"):
            # Use Ollama (local)
            try:
                from langchain_ollama import ChatOllama
                model_name = config.model.replace("ollama/", "").replace("Ollama/", "")
                self.llm = ChatOllama(
                    model=model_name,
                    temperature=config.temperature,
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                )
                self.logger.info(f"✅ Summarizer initialized with Ollama: {model_name}")
            except ImportError:
                self.logger.error("❌ langchain-ollama not installed. Run: pip install langchain-ollama")
                raise
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Ollama: {e}")
                raise
                
        elif model.startswith("groq/"):
            # Use Groq (cloud)
            try:
                from langchain_groq import ChatGroq
                model_name = config.model.replace("groq/", "").replace("Groq/", "")
                groq_api_key = os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    raise ValueError("GROQ_API_KEY environment variable required for Groq models")
                self.llm = ChatGroq(
                    model=model_name,
                    temperature=config.temperature,
                    api_key=groq_api_key
                )
                self.logger.info(f"✅ Summarizer initialized with Groq: {model_name}")
            except ImportError:
                self.logger.error("❌ langchain-groq not installed. Run: pip install langchain-groq")
                raise
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Groq: {e}")
                raise
                
        elif model.startswith("gemini") or model.startswith("google"):
            # Use Google Gemini (cloud)
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model=config.model,
                    temperature=config.temperature,
                    convert_system_message_to_human=True,
                    model_kwargs={}
                )
                self.logger.info(f"✅ Summarizer initialized with Gemini: {config.model}")
            except ImportError:
                self.logger.error("❌ langchain-google-genai not installed. Run: pip install langchain-google-genai")
                raise
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Gemini: {e}")
                raise
                
        else:
            error_msg = (
                f"Unsupported model: {config.model}\n"
                f"Supported formats:\n"
                f"  - ollama/model-name (e.g., ollama/llama3.2:3b)\n"
                f"  - groq/model-name (e.g., groq/llama-3.1-8b-instant)\n"
                f"  - gemini-* (e.g., gemini-2.5-flash)\n"
                f"  - google/* (e.g., google/gemini-pro)"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Setup prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating concise, informative summaries of technical book sections.

Your task:
1. Capture the key concepts, definitions, and main points
2. Preserve technical terms and important details
3. Write in clear, professional language
4. Keep the summary between 2-4 sentences (50-100 words)
5. Focus on what a reader would need to know to understand this section

DO NOT:
- Add information not in the original text
- Use phrases like "This section discusses..." or "The author explains..."
- Include meta-commentary about the text
- Make the summary too verbose"""),
            ("human", """Book Title: {book_title}
Chapter: {chapter}
Section Title: {title}

Section Content:
{text}

Create a concise summary (2-4 sentences, 50-100 words):""")
        ])
        
        # Create chain
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        # Token counter
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning(f"Failed to load tiktoken encoding: {e}. Using approximation.")
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Rough approximation: ~4 chars per token
            return len(text) // 4
    
    def _validate_summary(self, summary: str, original_text: str) -> bool:
        """
        Validate that summary meets quality criteria
        
        Returns:
            bool: True if summary is valid
        """
        if not summary or len(summary.strip()) == 0:
            return False
        
        # Check minimum length (at least 15 words)
        if len(summary.split()) < 15:
            return False
        
        # Check it's not just repeating the original
        summary_lower = summary.lower()
        original_lower = original_text.lower()
        
        # If summary is >80% identical to original, it's not a good summary
        if summary_lower in original_lower and len(summary) > 0.8 * len(original_text):
            return False
        
        return True
    
    def _extractive_fallback(self, text: str, max_tokens: int = 100) -> str:
        """
        Fallback to extractive summarization if LLM fails
        
        Simply takes first few sentences that fit within token budget.
        """
        sentences = text.replace('\n', ' ').split('. ')
        summary_parts = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add period back if missing
            if not sentence.endswith('.'):
                sentence += '.'
            
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= max_tokens:
                summary_parts.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        return ' '.join(summary_parts)
    
    async def _summarize_with_retry(
        self, 
        section: Dict[str, Any], 
        book_title: str
    ) -> Optional[str]:
        """
        Summarize a single section with retry logic
        
        Args:
            section: Section dict with 'text', 'title', 'chapter'
            book_title: Title of the book
            
        Returns:
            Summary text or None if all attempts fail
        """
        for attempt in range(self.config.max_retries):
            try:
                summary = await self.chain.ainvoke({
                    "book_title": book_title,
                    "chapter": section.get('chapter', 'Unknown'),
                    "title": section.get('title', 'Untitled'),
                    "text": section['text']
                })
                
                # Validate summary
                if self._validate_summary(summary, section['text']):
                    return summary.strip()
                else:
                    self.logger.warning(
                        f"⚠️  Invalid summary for '{section.get('title', 'Unknown')}', "
                        f"using first sentences instead"
                    )
                    return self._extractive_fallback(section['text'])
                    
            except Exception as e:
                self.logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.config.max_retries}): "
                    f"Error calling model '{self.config.model}' ({type(e).__name__}): {str(e)}. "
                    f"Retrying in {self.config.retry_delay}s..."
                )
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    self.logger.error(
                        f"LLM call failed after {self.config.max_retries} attempts: "
                        f"Error calling model '{self.config.model}' ({type(e).__name__}): {str(e)}"
                    )
        
        return None
    
    async def batch_summarize(
        self,
        sections: List[Dict[str, Any]],
        book_title: str
    ) -> List[Dict[str, Any]]:
        """
        Summarize multiple sections in batches
        
        Args:
            sections: List of section dicts with structure:
                {
                    'id': str,
                    'title': str,
                    'chapter': str,
                    'text': str,
                    'chunk_ids': List[str]
                }
            book_title: Title of the book
            
        Returns:
            List of summary dicts with structure:
                {
                    'section_id': str,
                    'section_title': str,
                    'chapter': str,
                    'summary': str,
                    'chunk_ids': List[str],
                    'token_count': int,
                    'original_token_count': int
                }
        """
        self.logger.info(f"🚀 Starting batch summarization for {len(sections)} sections")
        self.logger.info(f"   Model: {self.config.model}")
        self.logger.info(f"   Batch size: {self.config.batch_size}")
        
        summaries = []
        
        # Filter sections that are long enough to summarize
        sections_to_summarize = [
            s for s in sections 
            if self.count_tokens(s['text']) >= self.config.min_section_tokens
        ]
        
        if len(sections_to_summarize) < len(sections):
            self.logger.info(
                f"   Filtered to {len(sections_to_summarize)} sections "
                f"(minimum {self.config.min_section_tokens} tokens)"
            )
        
        # Process in batches
        for batch_idx in range(0, len(sections_to_summarize), self.config.batch_size):
            batch = sections_to_summarize[batch_idx:batch_idx + self.config.batch_size]
            batch_num = (batch_idx // self.config.batch_size) + 1
            total_batches = (len(sections_to_summarize) + self.config.batch_size - 1) // self.config.batch_size
            
            self.logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} sections)")
            
            # Summarize batch concurrently
            tasks = [
                self._summarize_with_retry(section, book_title)
                for section in batch
            ]
            
            batch_summaries = await asyncio.gather(*tasks)
            
            # Process results
            for section, summary in zip(batch, batch_summaries):
                if summary:
                    summaries.append({
                        'section_id': section['id'],
                        'section_title': section.get('title', 'Untitled'),
                        'chapter': section.get('chapter', 'Unknown'),
                        'summary': summary,
                        'chunk_ids': section.get('chunk_ids', []),
                        'token_count': self.count_tokens(summary),
                        'original_token_count': self.count_tokens(section['text'])
                    })
                else:
                    # Use extractive fallback
                    self.logger.error(
                        f"❌ Failed to summarize '{section.get('title', 'Unknown')}': "
                        f"Error calling model '{self.config.model}' (RESOURCE_EXHAUSTED): "
                        f"429 RESOURCE_EXHAUSTED. {{...}}"
                    )
                    fallback_summary = self._extractive_fallback(section['text'])
                    summaries.append({
                        'section_id': section['id'],
                        'section_title': section.get('title', 'Untitled'),
                        'chapter': section.get('chapter', 'Unknown'),
                        'summary': fallback_summary,
                        'chunk_ids': section.get('chunk_ids', []),
                        'token_count': self.count_tokens(fallback_summary),
                        'original_token_count': self.count_tokens(section['text'])
                    })
        
        # Log statistics
        if summaries:
            avg_original = sum(s['original_token_count'] for s in summaries) / len(summaries)
            avg_summary = sum(s['token_count'] for s in summaries) / len(summaries)
            compression_ratio = (1 - avg_summary / avg_original) * 100
            
            self.logger.info(f"✅ Batch summarization complete: {len(summaries)}/{len(sections)} sections summarized")
            self.logger.info(f"   Avg compression: {avg_original:.0f} → {avg_summary:.0f} tokens ({compression_ratio:.1f}% reduction)")
        else:
            self.logger.warning("⚠️  No summaries generated")
        
        return summaries


# Convenience function
def create_summarizer(
    model: str = "ollama/llama3.2:3b",
    temperature: float = 0.3,
    min_section_tokens: int = 400
) -> Summarizer:
    """
    Create a summarizer with common defaults
    
    Args:
        model: LLM model to use. Formats:
            - "ollama/llama3.2:3b" (local)
            - "groq/llama-3.1-8b-instant" (cloud)
            - "gemini-2.5-flash" (cloud)
        temperature: Sampling temperature (0.0 - 1.0)
        min_section_tokens: Minimum tokens in section to summarize
        
    Returns:
        Configured Summarizer instance
    """
    config = SummaryConfig(
        model=model,
        temperature=temperature,
        min_section_tokens=min_section_tokens
    )
    return Summarizer(config)
