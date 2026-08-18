import os
try:
    import wandb
except ImportError:
    wandb = None

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from opsd_trainer import OPSDTrainer
from dataclasses import dataclass, field

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )

    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    student_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the student during rollout. "
            "Default False (matches the main OPSD setup: student rolls out without <think>)."
        },
    )
    teacher_thinking: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the teacher when scoring student tokens. "
            "Default True. Set to False for the matched non-thinking ablation (both nonthink)."
        },
    )
    pct_method: str = field(
        default="none",
        metadata={
            "help": "Privileged Computation Transport method. Use none, phf_single, phf_random, "
            "phf_mean, phf_medoid, phf_grassmann, phf_set, set_ot, set_fgw, or set_uot."
        },
    )
    pct_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for the hidden-flow/transport auxiliary loss."},
    )
    pct_num_references: int = field(
        default=1,
        metadata={"help": "Number of correct references per example for multi-reference PCT."},
    )
    pct_layers: str = field(
        default="last",
        metadata={"help": "Hidden layers for PCT: last, all, even, or odd."},
    )
    pct_tau: float = field(
        default=0.05,
        metadata={"help": "Soft-min temperature for set-style PCT losses."},
    )
    pct_geometry_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for PHF-style trajectory Gram geometry term."},
    )
    pct_max_atoms: int = field(
        default=64,
        metadata={"help": "Maximum hidden-flow atoms per sample for OT losses."},
    )
    pct_sinkhorn_epsilon: float = field(
        default=0.05,
        metadata={"help": "Entropic regularization for Sinkhorn OT/UOT."},
    )
    pct_sinkhorn_iters: int = field(
        default=40,
        metadata={"help": "Number of Sinkhorn iterations for OT/UOT."},
    )
    pct_uot_rho: float = field(
        default=0.5,
        metadata={"help": "Mass relaxation strength for unbalanced OT."},
    )
    pct_fgw_outer: int = field(
        default=4,
        metadata={"help": "Outer updates for approximate fused Gromov-Wasserstein PCT."},
    )
    pct_fgw_feature_weight: float = field(
        default=0.5,
        metadata={"help": "FGW weight on trajectory structure; 0 is feature OT, 1 is structure-only GW."},
    )
    pct_grassmann_rank: int = field(
        default=2,
        metadata={"help": "Subspace rank for the phf_grassmann multi-reference ablation."},
    )
    pct_dataset_name: str = field(
        default="siyanzhao/Openthoughts_math_30k_opsd",
        metadata={
            "help": "Training dataset name or local data file. Local .json/.jsonl files are loaded with datasets' json loader."
        },
    )
    pct_dataset_split: str = field(
        default="train",
        metadata={"help": "Training split to use."},
    )
    pct_train_num_samples: int = field(
        default=0,
        metadata={"help": "Limit training to the first N examples. Set 0 for the full split."},
    )


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path (e.g., "Qwen3-1.7B" from "/home/siyanzhao/models/Qwen3-1.7B")
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_config = (
            f"opsd_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

        # Add fixed_teacher to wandb name if enabled
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Validate fixed_teacher argument
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )

    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    report_to = training_args.report_to or []
    if isinstance(report_to, str):
        report_to = [report_to]
    use_wandb = "wandb" in report_to
    if use_wandb and wandb is None:
        raise ImportError("report_to includes wandb, but wandb is not installed.")

    if os.environ.get("LOCAL_RANK", "0") == "0" and use_wandb:
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "use_tinker_loss": script_args.use_tinker_loss,
                "fixed_teacher": script_args.fixed_teacher,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
                "pct_method": script_args.pct_method,
                "pct_loss_weight": script_args.pct_loss_weight,
                "pct_num_references": script_args.pct_num_references,
                "pct_layers": script_args.pct_layers,
                "pct_tau": script_args.pct_tau,
                "pct_geometry_weight": script_args.pct_geometry_weight,
                "pct_max_atoms": script_args.pct_max_atoms,
                "pct_sinkhorn_epsilon": script_args.pct_sinkhorn_epsilon,
                "pct_sinkhorn_iters": script_args.pct_sinkhorn_iters,
                "pct_uot_rho": script_args.pct_uot_rho,
                "pct_fgw_outer": script_args.pct_fgw_outer,
                "pct_fgw_feature_weight": script_args.pct_fgw_feature_weight,
                "pct_grassmann_rank": script_args.pct_grassmann_rank,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    # No separate teacher model needed - we use the same model with privileged info

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty

    if script_args.pct_dataset_name.endswith((".json", ".jsonl")):
        dataset = load_dataset("json", data_files=script_args.pct_dataset_name)
        train_dataset = dataset["train"]
    else:
        dataset = load_dataset(script_args.pct_dataset_name)
        train_dataset = dataset[script_args.pct_dataset_split]
    if script_args.pct_train_num_samples > 0:
        train_dataset = train_dataset.select(range(min(script_args.pct_train_num_samples, len(train_dataset))))

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        pct_method=script_args.pct_method,
        pct_loss_weight=script_args.pct_loss_weight,
        pct_num_references=script_args.pct_num_references,
        pct_layers=script_args.pct_layers,
        pct_tau=script_args.pct_tau,
        pct_geometry_weight=script_args.pct_geometry_weight,
        pct_max_atoms=script_args.pct_max_atoms,
        pct_sinkhorn_epsilon=script_args.pct_sinkhorn_epsilon,
        pct_sinkhorn_iters=script_args.pct_sinkhorn_iters,
        pct_uot_rho=script_args.pct_uot_rho,
        pct_fgw_outer=script_args.pct_fgw_outer,
        pct_fgw_feature_weight=script_args.pct_fgw_feature_weight,
        pct_grassmann_rank=script_args.pct_grassmann_rank,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    trainer.save_model(training_args.output_dir)
