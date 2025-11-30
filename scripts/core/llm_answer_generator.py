#!/usr/bin/env python3
"""LLM answer generator with cost tracking."""
import os
from dataclasses import dataclass
from typing import List, Optional, Dict
from openai import OpenAI
import tiktoken

# OpenAI pricing (per 1M tokens) - as of 2024
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # $0.15/$0.60 per 1M tokens
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
}

@dataclass
class LLMCost:
    """Cost breakdown for a single LLM call."""
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    model: str

@dataclass
class AnswerResult:
    """Answer generation result with cost."""
    answer: str
    cost: LLMCost
    latency_ms: float

class LLMAnswerGenerator:
    """Generate answers using OpenAI API with cost tracking."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 512,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that answers questions using the provided context. "
            "Answer concisely and accurately. If the context does not contain enough information, "
            "say you are unsure."
        )
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        
        # Try to get encoding for the model, fallback to cl100k_base
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for most OpenAI models
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Cost tracking
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
    
    def generate(
        self,
        question: str,
        contexts: List[str],
        query_type: str = "oltp",
    ) -> AnswerResult:
        """
        Generate answer from question and contexts.
        
        Args:
            question: User query
            contexts: List of retrieved chunk texts
            query_type: "oltp" or "olap" (affects max_tokens)
            
        Returns:
            AnswerResult with answer, cost, and latency
        """
        import time
        
        # Adjust max_tokens based on query type
        # OLAP queries need longer answers (Microsoft: $0.20-$0.50 vs $0.02-$0.05)
        max_answer_tokens = self.max_tokens * 2 if query_type == "olap" else self.max_tokens
        
        # Format context (limit total context length to avoid excessive costs)
        # Limit to top 10 chunks to control input token count
        context_block = "\n\n".join(contexts[:10])
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context_block}",
            },
        ]
        
        # Generate answer
        start_time = time.time()
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=max_answer_tokens,
            )
            latency_ms = (time.time() - start_time) * 1000
            
            answer = completion.choices[0].message.content.strip()
            output_tokens = completion.usage.completion_tokens
            total_input_tokens = completion.usage.prompt_tokens
            
            # Calculate cost
            cost = self._calculate_cost(total_input_tokens, output_tokens)
            
            # Update totals
            self.total_calls += 1
            self.total_input_tokens += total_input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost.total_cost_usd
            
            return AnswerResult(
                answer=answer,
                cost=cost,
                latency_ms=latency_ms,
            )
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> LLMCost:
        """Calculate cost for a single API call."""
        if self.model not in PRICING:
            # Default to gpt-4o-mini pricing if unknown
            pricing = PRICING["gpt-4o-mini"]
        else:
            pricing = PRICING[self.model]
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return LLMCost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=input_cost + output_cost,
            model=self.model,
        )
    
    def get_total_stats(self) -> Dict:
        """Get total cost statistics."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_per_call_usd": self.total_cost_usd / self.total_calls if self.total_calls > 0 else 0.0,
        }
    
    def reset_stats(self):
        """Reset cost tracking statistics."""
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
