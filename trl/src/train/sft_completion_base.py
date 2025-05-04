import argparse

from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Union, List, Optional, Any, Dict

import numpy as np
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
    setup_chat_format
)

import warnings
from typing import Union, List, Optional, Any, Dict

import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin, pad_without_fast_tokenizer_warning

from sft_utils import get_response_template_ids, get_instruction_template_ids, DataCollatorForCompletionOnlyLM


@dataclass
class ExtendedScriptArguments(ScriptArguments):
    response_template: str = None
    instruction_template: str = None
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

    # tokenizer.chat_template = None
    # model, tokenizer = setup_chat_format(model, tokenizer)

    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({
            'eos_token': '<|endoftext|>',
            'pad_token': '<|im_start|>'
        })
        model.generation_config.pad_token_id = model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = model.config.eos_token_id = tokenizer.eos_token_id
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)


    print(f"Final setting - pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    print(f"Final setting - eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    print(f"Final setting - bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    
    ##############################################################################################
    # if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
    #     tokenizer.add_special_tokens({
    #         'eos_token': '<|im_end|>'
    #     })
    #     model.generation_config.pad_token_id = model.config.pad_token_id = tokenizer.pad_token_id
    #     model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    # training_args.model_init_kwargs = model_kwargs
    ###############################################################################################

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
    
    padding = True
    # max_length = None
    # if script_args.test is True:
    #     padding = 'max_length'
    #     max_length = training_args.max_seq_length
    max_length = training_args.max_seq_length
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=get_response_template_ids(model_args.model_name_or_path, tokenizer, script_args.response_template),
        instruction_template=get_instruction_template_ids(model_args.model_name_or_path, tokenizer, script_args.instruction_template),
        padding=padding,
        max_length=max_length
    )

    ################
    # Training
    ################

    trainer = SFTTrainer(
        # model=model_args.model_name_or_path,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        data_collator=data_collator,
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