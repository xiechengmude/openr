import argparse
import json
import logging
import os
from typing import Dict
from pathlib import Path

from omegaprm import LanguageModel, OmegaPRM

logger: logging.Logger

def setup_logging(log_file_prefix: str):
    """Set up logging configuration."""
    log_dir = os.path.dirname(log_file_prefix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"{log_file_prefix}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)

def load_questions(filepath: str):
    """Load questions from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
    with open(filepath, "r", encoding='utf-8') as f:
        return json.load(f)

def should_process_question(question: Dict[str, str], llm: LanguageModel) -> bool:
    """Filter questions based on initial rollouts."""
    prompt = question["problem"]
    correct_answer = question["final_answer"]

    has_correct = False
    has_incorrect = False

    initial_batch_answers = llm.generate_rollout(prompt, 32)

    for answer in initial_batch_answers:
        if llm.evaluate_correctness(answer, correct_answer):
            has_correct = True
        else:
            has_incorrect = True

        if has_correct and has_incorrect:
            return True

    return False

def process_question(omega_prm: OmegaPRM, question: Dict[str, str]):
    """Process a single question using OmegaPRM."""
    return omega_prm.run(question["problem"], question["final_answer"])

def save_question_data(collected_data: Dict, index: int, output_path: str):
    """Save collected data for a question."""
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"question_{index}.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(collected_data, f, indent=2, ensure_ascii=False)

def main(args):
    # Set up logging
    logger = setup_logging(args.log_file_prefix)
    
    # Load questions from specified dataset
    questions = load_questions(args.dataset_path)
    logger.info(f"Loaded {len(questions)} questions from {args.dataset_path}")
    
    # Initialize language model with configurations
    llm_kwargs = {
        "model_name": args.model_name,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "model_type": args.model_type
    }
    
    # Add API configurations if using API-based models
    if args.model_type in ["openai", "anthropic"]:
        if not args.api_key:
            raise ValueError(f"API key is required for {args.model_type} backend")
        llm_kwargs.update({
            "api_key": args.api_key,
            "api_base": args.api_base
        })
    
    llm = LanguageModel(**llm_kwargs)
    logger.info(f"Initialized {args.model_type} model: {args.model_name}")

    # Initialize OmegaPRM
    omega_prm = OmegaPRM(
        LM=llm,
        c_puct=args.c_puct,
        alpha=args.alpha,
        beta=args.beta,
        L=args.length_scale,
        k=args.num_rollouts,
        N=args.max_search_count,
        rollout_budget=args.rollout_budget,
        save_data_tree=args.save_data_tree,
    )

    # Process questions
    for i, question in enumerate(questions):
        if should_process_question(question, llm):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            collected_data = process_question(omega_prm, question)
            save_question_data(collected_data, i, args.output_dir)
        else:
            logger.info(f"Skipping question {i+1} (did not pass initial filtering)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OmegaPRM on filtered questions")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, 
                       default="/root/.cache/modelscope/hub/Qwen/Qwen2___5-Math-7B-Instruct",
                       help="Name or path of the model to use")
    parser.add_argument("--model_type", type=str, default="hf", 
                       choices=["hf", "vllm", "openai", "anthropic"],
                       help="Type of model backend to use")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to run the model on")
    
    # API configuration
    parser.add_argument("--api_key", type=str, 
                       help="API key for OpenAI/Anthropic backend")
    parser.add_argument("--api_base", type=str, 
                       help="Custom API base URL")
    
    # Dataset configuration
    parser.add_argument("--dataset_path", type=str, 
                       default="../sources/test.json",
                       help="Path to the dataset JSON file")
    parser.add_argument("--output_dir", type=str, 
                       default="./output",
                       help="Directory to save output files")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=30,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    
    # OmegaPRM parameters
    parser.add_argument("--c_puct", type=float, default=0.125,
                       help="Exploration constant for OmegaPRM")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Weight for MC(s) in OmegaPRM")
    parser.add_argument("--beta", type=float, default=0.9,
                       help="Length penalty for OmegaPRM")
    parser.add_argument("--length_scale", type=int, default=500,
                       help="Length scale in OmegaPRM")
    parser.add_argument("--num_rollouts", type=int, default=8,
                       help="Number of rollouts for Monte Carlo estimation")
    parser.add_argument("--max_search_count", type=int, default=20,
                       help="Max search count in OmegaPRM")
    parser.add_argument("--rollout_budget", type=int, default=200,
                       help="Rollout budget for OmegaPRM")
    
    # Logging and output
    parser.add_argument("--log_file_prefix", type=str, default="./logs/run",
                       help="Prefix for log files")
    parser.add_argument("--save_data_tree", action="store_true",
                       help="Whether to save the full data tree")
    
    args = parser.parse_args()
    main(args)
