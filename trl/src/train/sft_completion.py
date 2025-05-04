import argparse

from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
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

from sft_utils import get_response_template_ids, DataCollatorForCompletionOnlyLM

import logging
logger = logging.getLogger(__name__)
import os
import wandb



@dataclass
class ExtendedScriptArguments(ScriptArguments):
    response_template: str = None
    instruction_template: str = None
    dataset_format: str = field(default="parquet", metadata={"help": "json、csv、parquet..."})
    dataset_size: int = field(default=-1)
    gradient_checkpointing_use_reentrant: bool = field(default=False)
    test: Optional[bool] = field(default=False, metadata={"help": "Use limited samples for testing"})
    wandb_run_name: str = field(default="sft_filter_user_match", metadata={"help": "Wandb run name"})

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
    if hasattr(script_args, "gradient_checkpointing_use_reentrant"):
        training_args.gradient_checkpointing_kwargs = dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant)

    if script_args.test is True:
        training_args.report_to = []
    
    print(f'training_args: {training_args}')
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

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

    print(f'max_length: {training_args.max_seq_length}')

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code, 
        use_fast=True,
        max_length=training_args.max_seq_length,
        truncation_side="left",
    )


    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({
            'pad_token': '<|pad|>'
        })
        model.generation_config.pad_token_id = model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_dataset = dataset[script_args.dataset_eval_split]
    
    def preprocess_function(examples):
        formatted_chats = [
            tokenizer.apply_chat_template(messages, tokenize=False) 
            for messages in examples["messages"]
        ]
        tokenized = tokenizer(
            formatted_chats,
            padding=False,
            truncation=True,
            max_length=training_args.max_seq_length,
            return_tensors=None,
        )
        return {"input_ids": tokenized["input_ids"]}
    
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing training data with chat template and truncation",
    )
    
    # 控制dataset
    if script_args.dataset_size > 0:
        train_dataset = train_dataset.select(range(script_args.dataset_size))
    
    if script_args.test is True:
        train_dataset = train_dataset.select(range(1024))
        if eval_dataset is not None:
            eval_dataset = eval_dataset.select(range(256))
    
    padding = True
    max_length = training_args.max_seq_length

    if script_args.test is True:
        padding = 'max_length'
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=get_response_template_ids(model_args.model_name_or_path, tokenizer, script_args.response_template),
        # instruction_template=get_instruction_template_ids(model_args.model_name_or_path, tokenizer, script_args.instruction_template),
        padding=padding,
        max_length=max_length
    )

    if training_args.report_to and "wandb" in training_args.report_to:
        setup_wandb(script_args, training_args)

    ################
    # Training
    ################

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        data_collator=data_collator,
        # dataset_kwargs={"skip_prepare_dataset": True},
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