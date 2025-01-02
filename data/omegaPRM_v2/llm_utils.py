import os
import threading
from typing import List, Optional

# Import vllm if using vLLM backend
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM not installed. Install it if you wish to use it as a model backend.")

# Import OpenAI if using OpenAI backend
try:
    import openai
except ImportError:
    print("OpenAI not installed. Install it if you wish to use it as a model backend.")

# Set the environment variable for the endpoint
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

HF_BACKEND = False
if HF_BACKEND:
    from transformers import pipeline

class LLMService:
    """
    A class to manage a large language model (LLM) service using multiple backends including
    Hugging Face, vLLM, and OpenAI.
    """

    def __init__(self, 
                 model_name: str = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-Math-7B-Instruct",
                 device: str = "cuda", 
                 max_new_tokens: int = 2048,
                 temperature: float = 0.7, 
                 top_k: int = 30, 
                 top_p: float = 0.9, 
                 model_type: str = "hf",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None):
        """
        Initialize the LLMService with model parameters and sampling settings.

        Parameters:
        - model_name (str): Path, Hugging Face hub model name, or OpenAI model name.
        - device (str): Device for computation, e.g., 'cuda' or 'cpu'.
        - max_new_tokens (int): Maximum number of new tokens to generate.
        - temperature (float): Sampling temperature for response generation.
        - top_k (int): Top-K sampling parameter for response diversity.
        - top_p (float): Top-P sampling parameter for response diversity.
        - model_type (str): Type of model backend ('hf', 'vllm', or 'openai').
        - api_key (str, optional): API key for OpenAI.
        - api_base (str, optional): Custom API base URL for OpenAI.
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
        
        self.pipe = None
        self.llm = None
        self.client = None
        self.load_lock = threading.Lock()

        if self.model_type == "hf":
            if not HF_BACKEND:
                from transformers import pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype="auto",
                device_map=self.device
            )

    def start_service(self):
        """
        Start the LLM service by loading the model or setting up API client.
        Ensures thread-safe loading using a lock.
        """
        with self.load_lock:
            if self.model_type == "hf":
                if self.pipe is None:
                    print(f"Loading Hugging Face model '{self.model_name}' on device '{self.device}'...")
                    if not HF_BACKEND:
                        from transformers import pipeline
                    self.pipe = pipeline(
                        "text-generation",
                        model=self.model_name,
                        torch_dtype="auto",
                        device_map=self.device
                    )
                    print("Hugging Face model loaded successfully.")
            elif self.model_type == "vllm":
                if self.llm is None:
                    print(f"Loading vLLM model '{self.model_name}' on device '{self.device}'...")
                    self.llm = LLM(self.model_name, tensor_parallel_size=1)
                    print("vLLM model loaded successfully.")
            elif self.model_type == "openai":
                if self.client is None:
                    if not self.api_key:
                        raise ValueError("API key is required for OpenAI backend")
                    print("Setting up OpenAI client...")
                    self.client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=self.api_base or "https://api.openai.com/v1"  # 如果没有提供自定义URL，则使用默认值
                    )
                    print(f"OpenAI client setup with base URL: {self.api_base}")
            else:
                raise ValueError("Unsupported model_type. Choose 'hf', 'vllm', or 'openai'.")

    def generate_response(self, prompt: str, num_copies: int = 2) -> List[str]:
        """
        Generate responses from the model based on the provided prompt.

        Parameters:
        - prompt (str): The input prompt to generate responses for.
        - num_copies (int): The number of copies of the prompt to create for batch processing.

        Returns:
        - List[str]: A list of generated responses.
        """
        if self.model_type == "hf":
            return self._generate_response_hf(prompt, num_copies)
        elif self.model_type == "vllm":
            return self._generate_response_vllm(prompt, num_copies)
        elif self.model_type == "openai":
            return self._generate_response_openai(prompt, num_copies)
        else:
            raise ValueError("Unsupported model_type. Choose 'hf', 'vllm', or 'openai'.")

    def _generate_response_hf(self, prompt: str, num_copies: int = 2) -> List[str]:
        """
        Generate responses from the model based on the provided prompt, duplicated to form a batch.

        Parameters:
        - prompt (str): The input prompt to generate responses for.
        - num_copies (int): The number of copies of the prompt to create for batch processing (default is 16).

        Returns:
        - List[str]: A list of generated responses, each corresponding to a duplicate of the input prompt.
        """
        if self.pipe is None:
            raise ValueError("LLM service not started. Please call start_service() first.")

        # Create a batch of the same prompt
        prompts = [prompt] * num_copies

        # Generate responses from the model
        responses = self.pipe(
            prompts,
            max_new_tokens=self.max_new_tokens,
            batch_size=num_copies,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            return_full_text=False
        )
        response_message_batch = [result[0]["generated_text"] for result in responses]

        # Extract and return the generated text for each response
        return response_message_batch

    def _generate_response_vllm(self, prompt: str, num_copies: int) -> List[str]:
        """
        Generate responses using vLLM.
        """
        if self.llm is None:
            raise ValueError("LLM service not started for vLLM model. Please call start_service() first.")

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens
        )

        prompts = [prompt] * num_copies
        responses = self.llm.generate(prompts, sampling_params=sampling_params)

        return [response.outputs[0].text for response in responses]

    def _generate_response_openai(self, prompt: str, num_copies: int = 2) -> List[str]:
        """
        Generate responses using OpenAI API.

        Parameters:
        - prompt (str): The input prompt to generate responses for.
        - num_copies (int): The number of responses to generate.

        Returns:
        - List[str]: A list of generated responses.
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Please call start_service() first.")

        try:
            print(f"Sending request to: {self.api_base}")
            responses = []
            for _ in range(num_copies):
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                responses.append(response.choices[0].message.content.strip())
            return responses
        except Exception as e:
            print(f"Error generating response from OpenAI: {str(e)}")
            return [""] * num_copies

    def generate_batch(self, prompts: List[str], batch_size: int = 1) -> List[str]:
        """
        Generate responses for a batch of prompts.

        Parameters:
        - prompts (List[str]): A list of input prompts to generate responses for.
        - batch_size (int): The number of prompts to process in each batch.

        Returns:
        - List[str]: A list of generated responses.
        """
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_responses = self.generate_response(batch_prompts[0], num_copies=len(batch_prompts))
            responses.extend(batch_responses)
        return responses


if __name__ == "__main__":
    # Initialize the service for vLLM
    llm_service = LLMService(model_type="vllm")
    llm_service.start_service()

    prompt = "What is game theory?"
    responses = llm_service.generate_response(prompt, num_copies=3)

    print(responses)
