import argparse

from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Union, List, Optional, Any, Dict

import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    PreTrainedTokenizerBase,
    enable_full_determinism, 
    set_seed
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

@dataclass
class ExtendedScriptArguments(ScriptArguments):
    dataset_format: str = field(default="parquet", metadata={"help": "json、csv、parquet..."})
    dataset_size: int = field(default=-1)
    gradient_checkpointing_use_reentrant: bool = field(default=False)
    test: Optional[bool] = field(default=False, metadata={"help": "Use limited samples for testing"})
    

def main(script_args, training_args, model_args):
    if hasattr(script_args, "gradient_checkpointing_use_reentrant"):
        training_args.gradient_checkpointing_kwargs = dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant)

    if script_args.test is True:
        training_args.report_to = []

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({
            'pad_token': '<|pad|>'
        })
        model.generation_config.pad_token_id = model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    training_args.model_init_kwargs = model_kwargs

    ################
    # Dataset
    # add test
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_dataset = dataset[script_args.dataset_eval_split]
    
    # 控制dataset
    if script_args.dataset_size > 0:
        train_dataset = train_dataset.select(range(script_args.dataset_size))
    
    if script_args.test is True:
        train_dataset = train_dataset.select(range(1024))
        if eval_dataset is not None:
            eval_dataset = eval_dataset.select(range(256))

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)

def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ExtendedScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser

if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)    