import json
import yaml
import asyncio
import argparse

from typing import Any, Sequence, TypeVar, Union
from dataclasses import dataclass

from src.utils.logger import RunLogger, clean_old_run_directories, setup_logger
from src.utils.gpu import find_free_gpu
from src.utils.ConfigBase import ConfigBase


@dataclass
class DataConfig(ConfigBase):
    dataset: str = None
    valid_size: int = None
    no_shuffle: bool = False


@dataclass
class EvalConfig(ConfigBase):
    dataset: str = None
    method: str = "score"
    num_threads: int = 8
    max_concurrent_requests: int = 40
    output_file: str = None


def _init_method_evalconfig(method):
    config = EvalConfig()
    return config


T = TypeVar("T")


def parse_configs(args: Any, *configs: T) -> Union[T, Sequence[T]]:
    if not configs:
        raise ValueError("At least one config object is required")

    arg_dict = args.__dict__ if hasattr(args, "__dict__") else {}

    for key, value in arg_dict.items():
        if value is not None:  # Only update values that are not None
            for config in configs:
                if hasattr(config, key):
                    setattr(config, key, value)

    # If there is only one config object, return it directly
    # Otherwise, return a tuple of all config objects
    return configs[0] if len(configs) == 1 else configs


def init_logger():
    parser = argparse.ArgumentParser(description="This is an example program")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--method", type=str, default="rec")
    parser.add_argument("--description", type=str, default="", help="description")
    parser.add_argument(
        "--num_threads", type=int, default=8, help="Number of threads to use"
    )
    parser.add_argument("--model", type=str, default="qwen", help="model name")
    parser.add_argument(
        "--model_list_path",
        type=str,
        default=None,
        help="The path to llm_config",
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="the temperature of llm"
    )
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=120,
        help="the max concurrent running task",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="the output file",
    )

    args = parser.parse_args()

    if args.model_list_path:
        args.max_concurrent_requests = 80
    else:
        args.max_concurrent_requests = 40

    eval_config = _init_method_evalconfig(args.method)

    data_config, eval_config = parse_configs(args, DataConfig(), eval_config)

    suffix_str = f"{args.model}_{args.method}"
    if args.description:
        suffix_str += f"_{args.description}"

    RunLogger.initialize(suffix=suffix_str)

    return data_config, eval_config, args


def parse_yaml(path, config):
    with open(path, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yaml_config.items():
            if hasattr(config, k):
                setattr(config, k, v)
    return config


async def async_main():
    data_config, eval_config, args = init_logger()

    from score import AsyncCRS

    logger = setup_logger(__name__, "main.log")
    from src.model.client import AsyncLLMClient, LLMConfig

    llm_config = parse_configs(args, LLMConfig())

    input_file = data_config.dataset
    data_list = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            data = json.loads(line)
            data_list.append(data)


    total_test_data = data_list

    if args.model.lower() in ["llama", 
                      "qwen", 
                      "gemma", 
                      "qwen2.5-32b-instruct-gptq-int8", 
                      "deepseek-r1-distill-qwen-32b-quantized",
                      "qwen3-8b"
                     ]:
        if args.model_list_path:
            llm_config = llm_config.update(
                MODEL_NAME=args.model,
                api_max_retries=20,
                valid_max_retries=10,
            )
        else:
            model_path = {
                "llama": "/media/public/models/huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct",
                "qwen": "/media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct",
                "gemma": "/media/public/models/huggingface/google/gemma-2-9b-it",
                "qwen2.5-32b-instruct-gptq-int8": "/mnt/wangxiaolei/model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
                "deepseek-r1-distill-qwen-32b-quantized": "/mnt/wangxiaolei/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8",
                "qwen3-8b": "/mnt/wangxiaolei/model/Qwen/Qwen3-8B",
            }[args.model]
            llm_config = llm_config.update(
                OPENAI_API_KEY="EMPTY",
                OPENAI_API_BASE="http://localhost:8006/v1",
                MODEL_NAME=model_path,
                api_max_retries=20,
                valid_max_retries=10,
                max_concurrent_requests=128,
            )
    else:
        pass

    llm_client = await AsyncLLMClient.create(llm_config)

    logger.info(f"\n{llm_config.format()}")
    logger.info(f"\n{eval_config.format()}")
    logger.info(f"\n{data_config.format()}")
    args_str = "\n".join([f"{k}: {v}" for k, v in args.__dict__.items()])
    logger.info(f"\nargs_str\n{args_str}")

    CRS = AsyncCRS(
        eval_config=eval_config,
        llm_client=llm_client,
        **args.__dict__,
    )

    await CRS.parallel_pipeline_async(total_test_data)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()