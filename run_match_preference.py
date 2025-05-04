import os
import sys
import logging
import torch
import json
from dataclasses import dataclass, field
from typing import Optional
import random

import transformers
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers import is_wandb_available

from model.dssm_model_preference import Qwen2DSSMPreference
from trainer.dssm_trainer_preference import DSSMTrainerPreference
from utils.metrics import compute_recommendation_metrics
from utils.utils_like_dislike import create_movie_text
from datacollator.data_collator_preference import DSSMDataCollatorPreference

from datasets import load_dataset, DatasetDict


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
    similarity_temperature: float = field(
        default=0.07,
        metadata={"help": "The temperature for the similarity, between 0.05 - 0.2"}
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
    preference_column_name: str = field(
        default="preference",
        metadata={"help": "Preference field for matching"}
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
        default=4,
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
    movie_info_path: str = field(
        default=None,
        metadata={"help": "Path to the movie info file."}
    )
    item_max_length: int = field(
        default=192,
        metadata={"help": "The maximum length of the item text."}
    )
    item_batch_size: int = field(
        default=32,
        metadata={"help": "The batch size for the item text."}
    )
    num_negative_samples: int = field(
        default=32,
        metadata={"help": "The number of negative samples for each positive sample."}
    )
    wandb_run_name: str = field(
        default="match_filter_user_preference",
        metadata={"help": "The name of the wandbrun."}
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

def setup_wandb(training_args, data_args):
    if ("wandb" in training_args.report_to) and (
        training_args.local_rank == 0 or training_args.local_rank == -1
    ):
        run_name = os.path.basename(training_args.output_dir)
        project_name = os.environ.get("WANDB_PROJECT", data_args.wandb_run_name)
        
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

    ###############获取全局样本，用于全局随机负采样##################
    # movie_df = pd.read_csv(data_args.movie_name_path)
    # movie_list_raw = [name.strip() for name in movie_df["movieName"]]
    # movie_list = list(dict.fromkeys(movie_list_raw))
    # print(f"{len(movie_list)} movies has been loaded")
    ################################################################

    with open(data_args.movie_info_path, 'r', encoding='utf-8') as f:
        movie_info_dict = json.load(f)
    print(f"Finish loading movie info")

    train_movie_set = set([movie.strip() for movie in raw_datasets["train"]["target_movie"]])
    valid_movie_set = set([movie.strip() for movie in raw_datasets["validation"]["target_movie"]])
    test_movie_set = set([movie.strip() for movie in raw_datasets["test"]["target_movie"]])

    train_movie_list = sorted(list(train_movie_set))
    all_movie_set = train_movie_set.union(valid_movie_set).union(test_movie_set)
    all_movie_list = sorted(list(all_movie_set))

    print(f"train_movie_list_items: {len(train_movie_list)}")
    print(f"all_movie_list_items: {len(all_movie_list)}")

    print(f"There are {len(valid_movie_set - train_movie_set)} movies in validation that are not in train")
    print(f"There are {len(test_movie_set - train_movie_set)} movies in test that are not in train")

    
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
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        similarity_temperature=model_args.similarity_temperature,
        model_name_or_path=model_args.model_name_or_path,
    )

    if not hasattr(config, "similarity_temperature"):
        config.similarity_temperature = model_args.similarity_temperature

    print(f"Using similarity_temperature: {getattr(config, 'similarity_temperature', None)}")
    torch_dtype = get_torch_dtype(model_args.torch_dtype) if model_args.torch_dtype else None
    print(f"Using torch_dtype: {torch_dtype}")

    model = Qwen2DSSMPreference.from_pretrained( 
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
    )

    movie_name_to_idx = {name.strip(): i for i, name in enumerate(all_movie_list)}
    all_movie_indices = set([movie_name_to_idx[movie.strip()] for movie in all_movie_set])
    train_movie_indices = set([movie_name_to_idx[movie.strip()] for movie in train_movie_set if movie.strip() in movie_name_to_idx]) # 应用于训练集的负采样

    def preprocess_function(examples, split_name="train"):
        # local_random = random.Random(hash(f"{split_name}_{data_args.shuffle_seed}"))
        # 对preference文本进行编码
        preference_texts = examples[data_args.preference_column_name]
        preference_texts_with_prompt = [
            f"Given a user's movie preferences, retrieve relevant movies that match these preferences\nQuery: {text}" 
            if text.strip() else "" for text in preference_texts
        ]

        preference_encodings = tokenizer(
            preference_texts_with_prompt,
            padding="longest",
            max_length=data_args.max_seq_length,
            truncation=True,
        )

        num_negatives = data_args.num_negative_samples
        batch_size = len(examples[data_args.label_column_name])

        # ========== 物品侧 ==========
        all_item_texts = []
        movie_indices = []

        for i in range(batch_size):
            # ========== 正样本处理 ==========
            movie_name = examples[data_args.label_column_name][i].strip()
            pos_idx = movie_name_to_idx[movie_name]
            movie_indices.append(pos_idx)
        
            # ========== 负样本采样 ==========
            if split_name == "train":
                available_indices = train_movie_indices - {pos_idx}
                # neg_indices = local_random.sample(list(available_indices), num_negatives)
                neg_indices = random.sample(list(available_indices), num_negatives)
            else:
                all_indices = all_movie_indices - {pos_idx} 
                # neg_indices = local_random.sample(list(all_indices), num_negatives)
                neg_indices = random.sample(list(all_indices), num_negatives)

            # ========== 生成正负样本文本 ==========
            pos_text = create_movie_text(movie_name, movie_info_dict)
            neg_texts = [create_movie_text(all_movie_list[idx], movie_info_dict) for idx in neg_indices]
            all_item_texts.extend([pos_text] + neg_texts)  # 先展平为二维结构

        # ========== 逐样本分词 ==========
        item_encodings = tokenizer(
            all_item_texts,
            padding="max_length",  # 提前填充
            # padding="longest",
            truncation=True,
            max_length=data_args.item_max_length,
            return_tensors="pt",
        )

        num_items = 1 + num_negatives
        return {
            "preference_input_ids": preference_encodings["input_ids"],
            "preference_attention_mask": preference_encodings["attention_mask"],
            "item_input_ids": item_encodings["input_ids"].view(batch_size, num_items, -1),
            "item_attention_mask": item_encodings["attention_mask"].view(batch_size, num_items, -1),
            "labels": movie_indices,
        }

    processed_datasets = DatasetDict()
    for split, dataset in raw_datasets.items():
        processed_datasets[split] = dataset.map(
            lambda examples: preprocess_function(examples, split_name=split),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Tokenizing {split} dataset",
        )


    if training_args.do_train:
        train_dataset = processed_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # if data_args.shuffle_train_dataset:
        #     train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
    
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


    data_collator = DSSMDataCollatorPreference(
        tokenizer=tokenizer,
        padding='longest',
        max_user_length=data_args.max_seq_length,
        max_item_length=data_args.item_max_length,
        num_negatives=data_args.num_negative_samples,
    )
    
    trainer = DSSMTrainerPreference(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        similarity_temperature=model_args.similarity_temperature,
        movie_list=all_movie_list,
        movie_info_dict=movie_info_dict,
        data_args=data_args,
        data_collator=data_collator,
    )

    if training_args.report_to and "wandb" in training_args.report_to:
        setup_wandb(training_args, data_args)
    
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
        predict_results = trainer.predict(predict_dataset)
        metrics = predict_results.metrics
        
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
    
    if is_wandb_available() and wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()