import os
import threading
from typing import List, Optional, Dict, Any
import json
from transformers import pipeline

class LLMService:
    """
    A unified service class for managing various LLM backends including local models and remote APIs.
    Supports HuggingFace, vLLM, OpenAI, and Anthropic.
    """

    def __init__(self, 
                 model_name: str,
                 model_type: str = "hf",
                 device: str = "cuda", 
                 max_new_tokens: int = 2048,
                 temperature: float = 0.7, 
                 top_k: int = 30, 
                 top_p: float = 0.9,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None):
        """
        Initialize the LLM service with specified configuration.
        
        Args:
            model_name: Model identifier or path
            model_type: One of ["hf", "vllm", "openai", "anthropic"]
            device: Device for local models
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            api_key: API key for remote services
            api_base: API base URL for remote services
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.model_type = model_type.lower()
        self.api_key = api_key
        self.api_base = api_base
        
        # Initialize service-specific clients
        self.pipe = None
        self.llm = None
        self.client = None
        self.load_lock = threading.Lock()

    def start_service(self):
        """Initialize the specified LLM service."""
        with self.load_lock:
            if self.model_type == "hf":
                self._start_hf_service()
            elif self.model_type == "vllm":
                self._start_vllm_service()
            elif self.model_type in ["openai", "anthropic"]:
                self._setup_remote_api()
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _start_hf_service(self):
        """Initialize HuggingFace pipeline."""
        if self.pipe is None:
            print(f"Loading Hugging Face model '{self.model_name}' on device '{self.device}'...")
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype="auto",
                device_map=self.device
            )
            print("Hugging Face model loaded successfully.")

    def _start_vllm_service(self):
        """Initialize vLLM service."""
        try:
            from vllm import LLM, SamplingParams
            if self.llm is None:
                print(f"Loading vLLM model '{self.model_name}' on device '{self.device}'...")
                self.llm = LLM(self.model_name, tensor_parallel_size=1)
                print("vLLM model loaded successfully.")
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")

    def _setup_remote_api(self):
        """Set up remote API client."""
        if self.model_type == "openai":
            try:
                import openai
                openai.api_key = self.api_key
                if self.api_base:
                    openai.api_base = self.api_base
                self.client = openai
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")
        elif self.model_type == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Install with: pip install anthropic")

    def generate_response(self, prompt: str, num_copies: int = 2) -> List[str]:
        """Generate responses using the configured model or API."""
        if self.model_type == "hf":
            return self._generate_response_hf(prompt, num_copies)
        elif self.model_type == "vllm":
            return self._generate_response_vllm(prompt, num_copies)
        else:
            return self._generate_response_remote(prompt, num_copies)

    def _generate_response_hf(self, prompt: str, num_copies: int = 2) -> List[str]:
        """Generate responses using HuggingFace pipeline."""
        if self.pipe is None:
            raise ValueError("HuggingFace pipeline not initialized. Call start_service() first.")
        
        prompts = [prompt] * num_copies
        outputs = self.pipe(
            prompts,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        return [output[0]['generated_text'][len(prompt):].strip() for output in outputs]

    def _generate_response_vllm(self, prompt: str, num_copies: int = 2) -> List[str]:
        """Generate responses using vLLM."""
        if self.llm is None:
            raise ValueError("vLLM not initialized. Call start_service() first.")
        
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens
        )
        
        prompts = [prompt] * num_copies
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    def _generate_response_remote(self, prompt: str, num_copies: int = 2) -> List[str]:
        """Generate responses using remote API."""
        if self.client is None:
            raise ValueError(f"{self.model_type} client not initialized. Call start_service() first.")
        
        responses = []
        for _ in range(num_copies):
            if self.model_type == "openai":
                response = self._call_openai_api(prompt)
            elif self.model_type == "anthropic":
                response = self._call_anthropic_api(prompt)
            responses.append(response)
        return responses

    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API to generate response."""
        response = self.client.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        return response.choices[0].message.content

    def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API to generate response."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
