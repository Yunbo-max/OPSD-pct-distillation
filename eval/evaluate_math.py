import torch
import argparse
import inspect
import json
import re
from pathlib import Path
from datasets import load_dataset
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Use math_verify package directly
from math_verify import parse, verify


DATASET_CHOICES = ["math500", "amo-bench", "aime24", "aime25", "hmmt25", "minerva", "amc23", "aimo", "rrb-aime"]
METADATA_KEYS = (
    "perturbation",
    "perturbation_type",
    "transform",
    "variant",
    "source_problem_id",
    "original_problem_id",
    "source_dataset",
)


def extract_boxed_answer(text: str) -> str:
    """
    Extract answer from \\boxed{} command in the text.
    Returns the last boxed answer found.
    """
    # Find all \boxed{...} patterns
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    # Find the matching closing brace
    i = idx
    num_left_braces = 0
    right_brace_idx = None

    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    # Extract content inside \boxed{...}
    boxed_str = text[idx : right_brace_idx + 1]

    # Remove the \boxed{ and } wrapper
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        answer = boxed_str[7:-1]  # Remove "\boxed{" and "}"
        return answer.strip()

    return None


def grade_answer(predicted: str, ground_truth: str) -> bool:
    """
    Grade the predicted answer against ground truth using math_verify.

    Args:
        predicted: The predicted answer (already extracted from \\boxed{})
        ground_truth: The ground truth answer

    Returns:
        True if answers match, False otherwise
    """
    if predicted is None:
        return False

    try:
        # Ensure answers are wrapped in $ for latex parsing
        if not "$" in predicted:
            predicted = f"${predicted}$"
        if not "$" in ground_truth:
            ground_truth = f"${ground_truth}$"

        # Parse both answers
        pred_parsed = parse(predicted, fallback_mode="no_fallback")
        gt_parsed = parse(ground_truth, fallback_mode="no_fallback")

        # Verify equivalence
        return verify(gt_parsed, pred_parsed, timeout_seconds=5)
    except Exception as e:
        # If math_verify fails, try simple string comparison
        # Normalize by removing spaces, $, and converting to lowercase
        pred_norm = predicted.replace("$", "").replace(" ", "").lower().strip()
        gt_norm = ground_truth.replace("$", "").replace(" ", "").lower().strip()
        return pred_norm == gt_norm


def load_vllm_model(
    base_model_path: str,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int = None,
    enable_thinking: bool = True,
):
    """
    Load a model using vLLM for fast inference.

    Args:
        base_model_path: Path to the base model
        lora_adapter_path: Path to the LoRA adapters (checkpoint directory). If None, uses base model only.
        gpu_memory_utilization: GPU memory utilization (0.0 to 1.0)
        tensor_parallel_size: Number of GPUs to use for tensor parallelism
        max_model_len: Maximum model context length
        enable_thinking: Whether to enable thinking mode for Qwen3

    Returns:
        Tuple of (vLLM LLM instance, tokenizer)
    """
    if LLM is None:
        raise ImportError("vLLM is not installed. Use --backend transformers for local smoke tests.")
    print(f"Loading model with vLLM from: {base_model_path}")

    # Set max_model_len based on thinking mode if not specified
    if max_model_len is None:
        # For thinking mode, use larger context as recommended by Qwen3
        max_model_len = 40960 if enable_thinking else 32768
        print(
            f"Auto-setting max_model_len to {max_model_len} for {'thinking' if enable_thinking else 'non-thinking'} mode"
        )

    # Build LLM configuration
    llm_config = {
        "model": base_model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "distributed_executor_backend": "mp",
        "enforce_eager": True,
    }

    if lora_adapter_path is not None:
        print(f"LoRA adapter path provided: {lora_adapter_path}")

        # Check if LoRA weights exist
        adapter_path = Path(lora_adapter_path) / "adapter_model.safetensors"
        if not adapter_path.exists():
            # Try alternative name
            adapter_path = Path(lora_adapter_path) / "adapter_model.bin"

        if adapter_path.exists():
            print("LoRA weights found. Enabling LoRA support...")
            llm_config["enable_lora"] = True
            llm_config["max_lora_rank"] = 64  # Adjust based on your LoRA rank
            llm_config["max_loras"] = 1
            llm_config["max_cpu_loras"] = 1
        else:
            print(f"Warning: No LoRA weights found at {lora_adapter_path}")
            print("Continuing with base model only...")
            lora_adapter_path = None

    llm = LLM(**llm_config)

    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # Print dtype information
    print("\n" + "=" * 70)
    print("MODEL DTYPE INFORMATION")
    print("=" * 70)
    print(f"vLLM Model Config dtype: {llm.llm_engine.model_config.dtype}")
    print(f"vLLM Model quantization: {llm.llm_engine.model_config.quantization}")
    print(f"KV cache dtype: {llm.llm_engine.cache_config.cache_dtype}")
    print("=" * 70 + "\n")

    print("vLLM model loaded successfully!")
    return llm, tokenizer


def load_transformers_model(base_model_path: str, torch_dtype: str = "float16"):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_jsonl_or_json(path: str):
    raw_path = Path(path)
    if raw_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("data") or data.get("examples") or data.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"Could not load a list of examples from {path}")
    return rows


def load_eval_dataset(dataset_name: str, dataset_jsonl: str | None = None):
    if dataset_jsonl:
        dataset = load_jsonl_or_json(dataset_jsonl)
        print(f"Loaded local eval dataset {dataset_jsonl} with {len(dataset)} problems")
        return dataset
    if dataset_name.lower() == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        print(f"Loaded HuggingFaceH4/MATH-500 dataset with {len(dataset)} problems")
        return dataset
    if dataset_name.lower() == "amo-bench":
        dataset = load_dataset("meituan-longcat/AMO-Bench", split="test")
        print(f"Loaded meituan-longcat/AMO-Bench dataset with {len(dataset)} problems")
        return dataset
    if dataset_name.lower() == "minerva":
        dataset = load_dataset("math-ai/minervamath", split="test")
        print(f"Loaded minerva dataset with {len(dataset)} problems")
        return dataset
    if dataset_name.lower() == "amc23":
        dataset = load_dataset("math-ai/amc23", split="test")
        print(f"Loaded amc 23 dataset with {len(dataset)} problems")
        return dataset
    if dataset_name.lower() == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        print(f"Loaded HuggingFaceH4/aime_2024 dataset with {len(dataset)} problems")
        return dataset
    if dataset_name.lower() == "aime25":
        dataset = load_dataset("yentinglin/aime_2025", split="train", trust_remote_code=True)
        print(f"Loaded yentinglin/aime_2025 dataset with {len(dataset)} problems")
        return dataset
    if dataset_name.lower() == "hmmt25":
        dataset = load_dataset("MathArena/hmmt_feb_2025", split="train", trust_remote_code=True)
        print(f"Loaded MathArena/hmmt_feb_2025 dataset with {len(dataset)} problems")
        return dataset
    raise ValueError(
        f"Unknown dataset: {dataset_name}. Choose one of {DATASET_CHOICES}, or pass --dataset_jsonl for a local set."
    )


def normalize_eval_example(example: dict, dataset_name: str, idx: int) -> tuple[str, str, str | int]:
    name = dataset_name.lower()
    if name == "amo-bench":
        return example["prompt"], example["answer"], example.get("question_id", idx)
    if name == "aime24":
        return example["problem"], example["answer"], example.get("id", idx)
    if name in {"minerva", "amc23"}:
        return example["question"], example["answer"], example.get("id", idx)
    if name in {"aime25", "hmmt25"}:
        return example["problem"], str(example["answer"]), example.get("problem_idx", idx)

    problem = example.get("problem") or example.get("question") or example.get("prompt")
    if not problem:
        raise KeyError("Eval example must contain one of: problem, question, prompt")
    if "answer" in example:
        gt_answer = str(example["answer"])
    elif "ground_truth" in example:
        gt_answer = str(example["ground_truth"])
    elif "solution" in example:
        gt_answer = extract_boxed_answer(str(example["solution"])) or str(example["solution"])
    else:
        raise KeyError("Eval example must contain one of: answer, ground_truth, solution")
    question_id = example.get("id", example.get("problem_id", example.get("question_id", idx)))
    return str(problem), gt_answer, question_id


def eval_metadata(example: dict) -> dict:
    return {key: example[key] for key in METADATA_KEYS if key in example}


def evaluate_math500(
    llm,
    tokenizer,
    max_new_tokens: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    num_samples: int = None,
    output_file: str = None,
    lora_request=None,
    dataset_name: str = "math500",
    dataset_jsonl: str = None,
    base_model_name: str = None,
    enable_thinking: bool = True,
    val_n: int = 1,
    backend: str = "vllm",
    seed: int | None = None,
):
    """
    Evaluate model on MATH500 or other datasets using Qwen3 thinking mode with best practices.

    Args:
        llm: The vLLM LLM instance
        tokenizer: The tokenizer for chat template
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.6 for thinking, 0.7 for non-thinking)
        top_p: Top-p sampling parameter (0.95 for thinking, 0.8 for non-thinking)
        top_k: Top-k sampling parameter (20 recommended)
        min_p: Minimum probability threshold (0 recommended)
        presence_penalty: Presence penalty to reduce repetitions (0-2)
        num_samples: Number of samples to evaluate (None = all)
        output_file: Path to save detailed results
        lora_request: Optional LoRA request for inference
        dataset_name: Name of dataset to use
        base_model_name: Base model name for logging
        enable_thinking: Whether to use thinking mode
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION CONFIGURATION")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Thinking Mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Temperature: {temperature} (Qwen3 {'thinking' if enable_thinking else 'non-thinking'} mode)")
    print(f"Top-P: {top_p}")
    print(f"Top-K: {top_k}")
    print(f"Min-P: {min_p}")
    print(f"Presence Penalty: {presence_penalty}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Val-N (solutions per problem): {val_n}")
    print(f"Seed: {seed if seed is not None else 'unset'}")
    print(f"{'='*70}\n")

    print(f"Loading {dataset_name.upper()} dataset...")
    dataset = load_eval_dataset(dataset_name, dataset_jsonl=dataset_jsonl)

    # Limit to num_samples if specified
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # # Check if output file already exists with required samples
    # if output_file and Path(output_file).exists():
    #     print(f"\nFound existing results file: {output_file}")
    #     try:
    #         with open(output_file, 'r', encoding='utf-8') as f:
    #             existing_data = json.load(f)

    #         existing_results = existing_data.get('results', [])
    #         existing_count = len(existing_results)
    #         expected_count = len(dataset)

    #         if existing_count >= expected_count:
    #             print(f"✓ Existing results already have {existing_count} samples (expected: {expected_count})")
    #             print(f"Skipping generation and returning existing results.")
    #             print(f"Accuracy: {existing_data.get('accuracy', 0):.2f}%")
    #             print("=" * 70 + "\n")
    #             return existing_data.get('accuracy', 0), existing_results
    #         else:
    #             print(f"✗ Existing results have {existing_count} samples, but need {expected_count}")
    #             print(f"Proceeding with generation...")
    #     except Exception as e:
    #         print(f"Warning: Could not validate existing results: {e}")
    #         print(f"Proceeding with generation...")

    print(f"Evaluating on {len(dataset)} problems with {backend} inference...")

    if backend == "vllm":
        # Setup sampling parameters following Qwen3 best practices
        sampling_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_new_tokens,
            "presence_penalty": presence_penalty,
            "n": val_n,  # Generate val_n solutions per prompt
        }
        if seed is not None:
            if "seed" not in inspect.signature(SamplingParams).parameters:
                raise ValueError("This vLLM SamplingParams version does not support seed; unset --seed or upgrade vLLM.")
            sampling_kwargs["seed"] = seed
        sampling_params = SamplingParams(**sampling_kwargs)

    total = 0
    formatted_count = 0
    results = []

    # Metrics for val_n > 1
    pass_at_n = 0  # At least one correct
    total_correct_per_problem = 0  # Sum of correct solutions across all problems

    # Prepare all prompts/messages for batch inference
    all_prompts = []
    all_messages = []
    all_gt_answers = []
    all_problems = []
    all_question_ids = []
    all_metadata = []

    for idx, example in enumerate(dataset):
        example = dict(example)
        problem, gt_answer, question_id = normalize_eval_example(example, dataset_name, idx)

        # Format prompt following Qwen3 best practices for math problems
        user_message = (
            f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        )

        messages = [{"role": "user", "content": user_message}]

        all_messages.append(messages)
        all_gt_answers.append(gt_answer)
        all_problems.append(problem)
        all_question_ids.append(question_id)
        all_metadata.append(eval_metadata(example))

    # Run batch inference with vLLM using generate interface
    print(f"\nRunning vLLM batch inference on {len(all_messages)} problems...")
    print("Using generate interface with manual chat template...")

    # Apply chat template to all messages
    all_prompts = []
    for messages in all_messages:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        all_prompts.append(text)

    # Print dtype info before generation
    print("\n" + "=" * 70)
    print("GENERATION DTYPE CHECK")
    print("=" * 70)
    if backend == "vllm":
        print(f"Model dtype: {llm.llm_engine.model_config.dtype}")
        print(f"Quantization: {llm.llm_engine.model_config.quantization}")
        print(f"KV cache dtype: {llm.llm_engine.cache_config.cache_dtype}")
        print(f"Using LoRA: {lora_request is not None}")
        if lora_request is not None:
            if lora_request.lora_path is None:
                raise ValueError(
                    "LoRA request created but lora_local_path is None; lora weights are empty, might be issue with using zero3 + peft; try using zero2"
                )
            print(f"LoRA path: {lora_request.lora_path}")
    else:
        print(f"Model dtype: {next(llm.parameters()).dtype}")
        print("Using LoRA: False")
    print("=" * 70 + "\n")

    # Generate outputs
    if backend == "vllm":
        if lora_request is not None:
            outputs = llm.generate(all_prompts, sampling_params, lora_request=lora_request, use_tqdm=True)
        else:
            outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)
    else:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        if lora_request is not None:
            raise ValueError("LoRA checkpoint evaluation currently requires --backend vllm.")
        outputs = []
        for prompt in tqdm(all_prompts, desc="Generating"):
            encoded = tokenizer(prompt, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                generated = llm.generate(
                    **encoded,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p,
                    top_k=top_k if top_k > 0 else 0,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=val_n,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = encoded["input_ids"].shape[1]
            outputs.append([tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in generated])

    # Process results
    print("\nProcessing results...")
    for idx, (output, problem, gt_answer, question_id, metadata) in enumerate(
        zip(outputs, all_problems, all_gt_answers, all_question_ids, all_metadata)
    ):
        # Process all val_n generations for this problem
        generations = []
        predicted_answers = []
        is_correct_list = []
        is_formatted_list = []

        output_texts = [o.text for o in output.outputs] if backend == "vllm" else output
        for generated_text in output_texts:

            # Extract answer from generated text
            predicted_answer = extract_boxed_answer(generated_text)

            # Check if answer was properly formatted
            is_formatted = predicted_answer is not None

            # Grade the answer
            is_correct = grade_answer(predicted_answer, gt_answer)

            generations.append(generated_text)
            predicted_answers.append(predicted_answer if predicted_answer else "[No boxed answer found]")
            is_correct_list.append(is_correct)
            is_formatted_list.append(is_formatted)

        # Calculate metrics for this problem
        num_correct = sum(is_correct_list)
        num_formatted = sum(is_formatted_list)
        has_correct = any(is_correct_list)

        # Majority vote: find the most common answer among formatted predictions
        majority_vote_correct = False
        if num_formatted > 0:
            from collections import Counter

            formatted_predictions = [pred for pred, fmt in zip(predicted_answers, is_formatted_list) if fmt]
            if formatted_predictions:
                most_common_answer = Counter(formatted_predictions).most_common(1)[0][0]
                majority_vote_correct = grade_answer(most_common_answer, gt_answer)

        # Update global metrics
        if has_correct:
            pass_at_n += 1
        total_correct_per_problem += num_correct
        formatted_count += num_formatted
        total += val_n

        # Store result with all generations
        result = {
            "problem_id": question_id if question_id is not None else idx,
            "metadata": metadata,
            "problem": problem,
            "ground_truth": gt_answer,
            "val_n": val_n,
            "generations": [
                {"predicted_answer": pred, "full_generation": gen, "correct": corr, "formatted": fmt}
                for pred, gen, corr, fmt in zip(
                    predicted_answers, generations, is_correct_list, is_formatted_list
                )
            ],
            "num_correct": num_correct,
            "pass_at_n": has_correct,
            "majority_vote_correct": majority_vote_correct,
            # For backward compatibility
            "predicted_answer": predicted_answers[0],
            "full_generation": generations[0],
            "correct": is_correct_list[0],
            "formatted": is_formatted_list[0],
        }
        results.append(result)

        # Print progress for each problem
        format_rate = formatted_count / total * 100
        current_pass_at_n = pass_at_n / (idx + 1) * 100
        current_avg_at_n = total_correct_per_problem / total * 100

        # Print brief update for every problem
        status = "✓" if has_correct else "✗"
        print(
            f"{status} [{idx + 1}/{len(dataset)}] Pass@{val_n}: {current_pass_at_n:.1f}% | Avg@{val_n}: {current_avg_at_n:.1f}% | Formatted: {format_rate:.1f}%"
        )

        # Print detailed info every 10 problems
        if (idx + 1) % 10 == 0:
            print(f"\n{'='*70}")
            print(f"Progress: {idx + 1}/{len(dataset)}")
            print(f"Pass@{val_n}: {current_pass_at_n:.2f}%")
            print(f"Average@{val_n}: {current_avg_at_n:.2f}%")
            print(f"Format Rate: {format_rate:.2f}%")
            print(f"Last problem: {problem[:100]}...")
            print(f"Solutions correct: {num_correct}/{val_n}")
            print(f"Majority vote: {'✓' if majority_vote_correct else '✗'}")
            print(f"Ground truth: {gt_answer}")
            print(f"{'='*70}\n")

    # Calculate final metrics
    num_problems = len(dataset)
    format_rate = formatted_count / total * 100

    # Calculate pass@n, average@n, and majority vote metrics
    pass_at_n_pct = pass_at_n / num_problems * 100
    average_at_n_pct = total_correct_per_problem / total * 100

    # Calculate majority vote accuracy
    majority_vote_correct_count = sum(1 for r in results if r["majority_vote_correct"])
    majority_vote_at_n_pct = majority_vote_correct_count / num_problems * 100

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS")
    print("=" * 70)
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Thinking Mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Total problems: {num_problems}")
    print(f"Solutions per problem: {val_n}")
    print(f"Total solutions: {total}")
    print(f"\nMetrics:")
    print(f"  Pass@{val_n}: {pass_at_n_pct:.2f}% ({pass_at_n}/{num_problems})")
    print(f"  Average@{val_n}: {average_at_n_pct:.2f}% ({total_correct_per_problem}/{total})")
    print(
        f"  Majority Vote@{val_n}: {majority_vote_at_n_pct:.2f}% ({majority_vote_correct_count}/{num_problems})"
    )
    print(f"\nFormatting:")
    print(f"  Formatted (boxed) answers: {formatted_count}/{total}")
    print(f"  Format rate: {format_rate:.2f}%")
    print("=" * 70)

    # Save detailed results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "base_model": base_model_name,
            "dataset": dataset_name,
            "enable_thinking": enable_thinking,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "val_n": val_n,
            "num_problems": num_problems,
            "total_solutions": total,
            "pass_at_n": pass_at_n,
            "pass_at_n_pct": pass_at_n_pct,
            "average_at_n": total_correct_per_problem,
            "average_at_n_pct": average_at_n_pct,
            "majority_vote_at_n": majority_vote_correct_count,
            "majority_vote_at_n_pct": majority_vote_at_n_pct,
            "formatted_count": formatted_count,
            "format_rate": format_rate,
            "results": results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {output_file}")

    return average_at_n_pct, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on MATH tasks with Qwen3 thinking mode")
    parser.add_argument(
        "--base_model",
        type=str,
        default="/infra/old-home/home/siyanzhao/models/Qwen3-4B-Instruct-2507",
        help="Path to base model",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to checkpoint directory with LoRA adapters. If not provided, will use base model only.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        choices=DATASET_CHOICES,
        help="Dataset to use for evaluation (default: math500)",
    )
    parser.add_argument(
        "--dataset_jsonl",
        type=str,
        default=None,
        help="Optional local JSON/JSONL eval file. Use with --dataset as the output label.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=38912,
        help="Maximum tokens to generate (default: 32768, use 38912 for complex competition problems)",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=True,
        help="Enable Qwen3 thinking mode (default: True)",
    )
    parser.add_argument(
        "--no_thinking", dest="enable_thinking", action="store_false", help="Disable Qwen3 thinking mode"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (auto: 0.6 for thinking, 0.7 for non-thinking)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p sampling parameter (auto: 0.95 for thinking, 0.8 for non-thinking)",
    )
    parser.add_argument(
        "--top_k", type=int, default=-1, help="Top-k sampling parameter (default: -1, disabled)"
    )
    parser.add_argument(
        "--min_p", type=float, default=0.0, help="Minimum probability threshold (default: 0.0)"
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Presence penalty to reduce repetitions (0-2, default: 0.0)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional generation seed for reproducible eval sampling.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save detailed results JSON")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (0.0 to 1.0, default: 0.9)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model context length (auto: 40960 for thinking, 32768 for non-thinking)",
    )
    parser.add_argument(
        "--val_n", type=int, default=6, help="Number of solutions to sample per problem (default: 6)"
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers"],
        default="vllm",
        help="Generation backend. Use transformers for dependency-light smoke tests.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"],
        help="Torch dtype for --backend transformers.",
    )

    args = parser.parse_args()

    # Validate checkpoint directory exists if provided
    if args.checkpoint_dir is not None:
        checkpoint_path = Path(args.checkpoint_dir)
        if not checkpoint_path.exists():
            print(f"\n{'='*70}")
            print("ERROR: Checkpoint directory does not exist")
            print(f"{'='*70}")
            print(f"Provided checkpoint directory: {args.checkpoint_dir}")
            print("This directory does not exist.")
            print(
                "\nPlease provide a valid checkpoint directory or omit --checkpoint_dir to use the base model only."
            )
            print(f"{'='*70}\n")
            exit(1)

    if args.top_p is None:
        args.top_p = 0.95 if args.enable_thinking else 0.8
        print(
            f"Auto-setting top_p to {args.top_p} for {'thinking' if args.enable_thinking else 'non-thinking'} mode"
        )

    # Warn if using greedy decoding in thinking mode
    if args.enable_thinking and args.temperature == 0.0:
        print("\n" + "!" * 70)
        print("WARNING: Using greedy decoding (temperature=0.0) in thinking mode!")
        print("Qwen3 recommends temperature=0.6 for thinking mode to avoid")
        print("performance degradation and endless repetitions.")
        print("!" * 70 + "\n")

    # Auto-generate output file if not specified
    if args.output_file is None:
        parts = ["eval_results", args.dataset, Path(args.base_model).name]
        if args.checkpoint_dir:
            checkpoint_path = Path(args.checkpoint_dir)
            parts += [checkpoint_path.parent.name, checkpoint_path.name]
        parts += [
            "thinking" if args.enable_thinking else "nonthinking",
            f"temp{args.temperature}",
            f"valn{args.val_n}",
        ]
        args.output_file = str(Path("eval_results") / ("_".join(parts) + ".json"))

    print(f"Results will be saved to: {args.output_file}")

    print("\n" + "=" * 70)
    print("QWEN3 MATH EVALUATION WITH THINKING MODE")
    print("=" * 70)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint_dir or 'None (base model only)'}")
    print(f"Thinking Mode: {'ENABLED ✓' if args.enable_thinking else 'DISABLED'}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(
        f"Temperature: {args.temperature} (Qwen3 {'thinking' if args.enable_thinking else 'non-thinking'} mode)"
    )
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Min-p: {args.min_p}")
    print(f"Presence penalty: {args.presence_penalty}")
    print(f"Num samples: {args.num_samples or 'All'}")
    print(f"Val-N (solutions per problem): {args.val_n}")
    print(f"Seed: {args.seed if args.seed is not None else 'unset'}")
    print(f"Output file: {args.output_file}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print("=" * 70 + "\n")

    # Load model
    if args.backend == "vllm":
        llm, tokenizer = load_vllm_model(
            args.base_model,
            args.checkpoint_dir,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            enable_thinking=args.enable_thinking,
        )
    else:
        llm, tokenizer = load_transformers_model(args.base_model, torch_dtype=args.torch_dtype)

    # Setup LoRA request if checkpoint is provided
    lora_request = None
    if args.backend == "vllm" and args.checkpoint_dir is not None:
        try:
            from vllm.lora.request import LoRARequest

            # Verify LoRA weights exist
            adapter_safetensors = Path(args.checkpoint_dir) / "adapter_model.safetensors"
            adapter_bin = Path(args.checkpoint_dir) / "adapter_model.bin"

            if adapter_safetensors.exists() or adapter_bin.exists():
                lora_request = LoRARequest("checkpoint_lora", 1, args.checkpoint_dir)
                print(f"✓ Successfully created LoRA request for: {args.checkpoint_dir}")
            else:
                print(f"Warning: No LoRA adapter weights found at {args.checkpoint_dir}")
                print("Expected 'adapter_model.safetensors' or 'adapter_model.bin'")
                print("Continuing with base model only...")
        except ImportError:
            print("Warning: Could not import LoRARequest. Running without LoRA.")
        except Exception as e:
            print(f"Warning: Could not create LoRA request: {e}")
            print("Continuing without LoRA.")

    # Run evaluation
    average_at_n_pct, results = evaluate_math500(
        llm,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        num_samples=args.num_samples,
        output_file=args.output_file,
        lora_request=lora_request,
        dataset_name=args.dataset,
        dataset_jsonl=args.dataset_jsonl,
        base_model_name=args.base_model,
        enable_thinking=args.enable_thinking,
        val_n=args.val_n,
        backend=args.backend,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"Final Average@{args.val_n}: {average_at_n_pct:.2f}%")
    print(f"Results saved to: {args.output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
