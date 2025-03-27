import os
import sys
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers import is_wandb_available

from model.classify_model import Qwen2ForClassification
from utils.metrics import compute_recommendation_metrics

from datasets import load_dataset
import pandas as pd

if is_wandb_available():
    import wandb

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="/mnt/wangxiaolei/model/Qwen/gte-Qwen2-7B-instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    loss_type: Optional[str] = field(
        default="ForSequenceClassification",
        metadata={
            "help": "Which loss function to use. You can run `--loss_type=ForCausalLM`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_file: str = field(
        default=None, metadata={"help": "The path to the training file. format: jsonl"}
    )
    validation_file: str = field(
        default=None, metadata={"help": "The path to the validation file. format: jsonl"}
    )
    test_file: str = field(
        default=None, metadata={"help": "The path to the test file. format: jsonl"}
    )
    max_seq_length: int = field(
        default=1536,
        metadata={"help": "The maximum sequence length of the input."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch."}
    )

    # 输入文本列名
    text_column_name: str = field(
        default="cot_result",
        metadata={"help": "COT text for matching"}
    )
    label_column_name: str = field(
        default="target_movie",
        metadata={"help": "label for matching"}
    )
    
    # 覆盖缓存
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Whether to overwrite the cached preprocessed datasets or not."}
    )

    # 预处理进程数
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum num of training samples"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum num of validation samples"}
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum num of prediction samples"}
    )
    shuffle_train_dataset: bool = field(
        default=True, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    movie_name_path: str = field(
        default="dataset/movies_with_mentions.csv",
        metadata={"help": "Path to the movie name file."}
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    metrics = compute_recommendation_metrics(
        predictions=predictions,
        labels=labels,
        k_values=(1, 5, 10, 20, 50)
    )
    return metrics

def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

def setup_wandb(training_args):
    if ("wandb" in training_args.report_to) and (
        training_args.local_rank == 0 or training_args.local_rank == -1
    ):
        run_name = os.path.basename(training_args.output_dir)
        project_name = os.environ.get("WANDB_PROJECT", "cot_result_1_epoch")
        
        wandb.init(
            project=project_name,
            name=run_name,
            dir=training_args.output_dir,
        )
        logger.info(f"Wandb initialized: {run_name}")

def get_torch_dtype(dtype_str):
    if dtype_str == "auto":
        return "auto"
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    return None

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1])) # 提供json配置文件
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    setup_logging(training_args)

    print(f"Model Arguments:\n {model_args}\n")
    print(f"Data Arguments:\n {data_args}\n")
    print(f"Training Arguments:\n {training_args}\n")

    set_seed(training_args.seed)

    # 检测检查点
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"The output directory ({training_args.output_dir}) already exists and is not empty. Overwrite with --overwrite_output_dir."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, continue training from {last_checkpoint}. If you are retraining, change '--output_dir' or add '--overwrite_output_dir'."
            )
            training_args.resume_from_checkpoint = last_checkpoint
    
    
    data_files = {}
    if data_args.train_file is not None and training_args.do_train:
        data_files["train"] = data_args.train_file
        logger.info(f"Training data file: {data_args.train_file}")
    if data_args.validation_file is not None and training_args.do_eval:
        data_files["validation"] = data_args.validation_file
        logger.info(f"Validation data file: {data_args.validation_file}")
    if data_args.test_file is not None and training_args.do_predict:
        data_files["test"] = data_args.test_file
        logger.info(f"Test data file: {data_args.test_file}")

    raw_datasets = load_dataset("json", data_files=data_files)
    
    movie_df = pd.read_csv(data_args.movie_name_path)
    movie_list_raw = [name.strip() for name in movie_df["movieName"]]
    movie_set = set()
    duplicates = []
    for movie in movie_list_raw:
        if movie in movie_set:
            duplicates.append(movie)
        else:
            movie_set.add(movie)
    if duplicates:
        print(f'Found {len(duplicates)} duplicate movies: {duplicates}')
    movie_list = list(dict.fromkeys(movie_list_raw))
    print(f"{len(movie_list)} movies has been loaded")
    
    num_labels = len(movie_list)
    print(f"label nums: {num_labels}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        use_cache=False if training_args.gradient_checkpointing else True,
        num_labels=num_labels,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        # Qwen Classification需指定pad_token_id
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        # 指定loss_type
        loss_type="ForSequenceClassification",
    )

    torch_dtype = get_torch_dtype(model_args.torch_dtype) if model_args.torch_dtype else None

    model = Qwen2ForClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
    )

    movie_name_to_id = {name: i for i, name in enumerate(movie_list)}

    def preprocess_function(examples):
        texts = examples[data_args.text_column_name]
        result = tokenizer(
            texts,
            padding="max_length" if data_args.pad_to_max_length else False,
            max_length=data_args.max_seq_length,
            truncation=True,
        )
        
        if data_args.label_column_name in examples:
            result["labels"] = [
                movie_name_to_id.get(label.strip(), -100) # 未知标签映射为-100，理论无 
                for label in examples[data_args.label_column_name]
            ]
        
        return result
    
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names if "train" in raw_datasets else None,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Tokenizing dataset",
    )

    train_dataset = None
    eval_dataset = None
    predict_dataset = None

    if training_args.do_train:
        train_dataset = processed_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        if data_args.shuffle_train_dataset:
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
    
    if training_args.do_eval:
        eval_dataset = processed_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    
    if training_args.do_predict:
        predict_dataset = processed_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # wandb
    if training_args.report_to and "wandb" in training_args.report_to:
        setup_wandb(training_args)

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("Finished training")
    
    if training_args.do_eval:
        logger.info("*** Evaluation ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Prediction ***")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
    
    if is_wandb_available() and wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()