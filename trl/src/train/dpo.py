"""
# Full training
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns

# LoRA:
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpo_utils import DataCollatorForPreferenceAndLength, DPOTrainer

from trl import (
    DPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

import os
import wandb
import logging
logger = logging.getLogger(__name__)

@dataclass
class ExtendedScriptArguments(ScriptArguments):
    dataset_size: int = field(default=-1)
    gradient_checkpointing_use_reentrant: bool = field(default=False)
    test: Optional[bool] = field(default=False, metadata={"help": "Use limited samples for testing"})
    # max_rejected_length: int = field(default=None, metadata={"help": "Maximum length of the rejected completion"})
    wandb_run_name: str = field(default="dpo_filter_user_match", metadata={"help": "Wandb run name"})

def setup_wandb(script_args, training_args):
    if ("wandb" in training_args.report_to) and (
        training_args.local_rank == 0 or training_args.local_rank == -1
    ):
        run_name = os.path.basename(training_args.output_dir)
        project_name = os.environ.get("WANDB_PROJECT", script_args.wandb_run_name)
        
        wandb.init(
            project=project_name,
            name=run_name,
            dir=training_args.output_dir,
        )
        logger.info(f"Wandb initialized: {run_name}")

def main(script_args, training_args, model_args):
    ################
    # Model & Tokenizer
    ###################
    # if script_args.test is True:
    #     training_args.report_to = []
    
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    peft_config = get_peft_config(model_args) # 用于微调模型的一部分参数
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        ) # ref model == base model
    else:
        ref_model = None # 微调在基础模型上进行
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({
            "pad_token": "<|pad|>"
        })
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_dataset = dataset[script_args.dataset_eval_split]

    if script_args.dataset_size > 0:
        train_dataset = train_dataset.select(range(script_args.dataset_size))
    
    if script_args.test is True:
        train_dataset = train_dataset.select(range(1024))
        if eval_dataset is not None:
            eval_dataset = eval_dataset.select(range(256))

    print(f"train dataset size: {len(train_dataset)}")
    
    # max_prompt_length = None
    # max_completion_length = None
    # max_rejected_length = None
    # data_collator = None
    # if script_args.test is True:
    #     max_prompt_length = training_args.max_prompt_length
    #     max_completion_length = training_args.max_completion_length
    #     data_collator = DataCollatorForPreferenceAndLength(max_prompt_length=max_prompt_length, max_completion_length=max_completion_length, max_rejected_length=max_rejected_length)
    
    print(f"max_prompt_length: {training_args.max_prompt_length}, max_completion_length: {training_args.max_completion_length}, max_length: {training_args.max_length}")

    if training_args.report_to and "wandb" in training_args.report_to:
        setup_wandb(script_args, training_args)
    
    ##########
    # Training
    ################
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=peft_config,
        # data_collator=data_collator
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ExtendedScriptArguments, DPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)