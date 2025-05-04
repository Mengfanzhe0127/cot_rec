import json
import time
import copy
import yaml

from openai import AsyncOpenAI
import asyncio
import aiofiles
from litellm import Router
from typing import Optional, Any, Callable
from dataclasses import dataclass
from src.utils.logger import setup_logger
from src.utils.ConfigBase import ConfigBase
from src.utils.data import (
    parse_md_dict,
    parse_md_list,
    parse_indented_list,
    hash_string, parse_json_from_response,
)

logger = setup_logger(__name__, log_file="log/llm_client.log")
cost_logger = setup_logger("cost_logger", log_file="cost_logger.log")


@dataclass
class LLMConfig(ConfigBase):
    OPENAI_API_KEY: str = "EMPTY"
    OPENAI_API_BASE: str = "http://localhost:8000/v1"
    MODEL_NAME: str = (
        "/media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct"
    )
    model_list_path: str = None
    temperature: float = 0.7
    max_tokens: int = 4096

    api_max_retries: int = 20
    valid_max_retries: int = 20
    retry_delay: int = 1

    request_timeout: int = 600
    cache_filepath: str = "cache/llm_response.json"
    max_concurrent_requests: int = 156


class AsyncLLMClient:
    def __init__(
        self,
        llm_config: LLMConfig,
        **kwargs,
    ):
        self.llm_config = llm_config
        self.model_name = llm_config.MODEL_NAME
        self.request_semaphore = asyncio.Semaphore(
            llm_config.max_concurrent_requests
        )  # Add semaphore

        if self.llm_config.model_list_path:
            with open(self.llm_config.model_list_path) as f:
                model_list = yaml.safe_load(f)
            self._client = Router(
                model_list=model_list,
                disable_cooldowns=True,
                routing_strategy="least-busy",
            )
            self.acompletion = self._client.acompletion
        else:
            self._client = AsyncOpenAI(
                api_key=llm_config.OPENAI_API_KEY,
                base_url=llm_config.OPENAI_API_BASE,
                max_retries=llm_config.api_max_retries,
            )
            self.acompletion = self._client.chat.completions.create
        self.cache_path = self.llm_config.cache_filepath
        self.update_time = 0
        self.cache = {}
        self.cache_lock = asyncio.Lock()

    @classmethod
    async def create(cls, llm_config: LLMConfig, **kwargs):
        self = cls(llm_config, **kwargs)
        await self._load_cache_from_file()
        return self

    async def _load_cache_from_file(self) -> None:
        async with self.cache_lock:
            try:
                async with aiofiles.open(self.cache_path, "r") as f:
                    content = await f.read()
                    self.cache = json.loads(content)
                    logger.info(f"Loaded cache from {self.cache_path}")
            except FileNotFoundError:
                logger.warning(f"Cache file not found at {self.cache_path}")
                self.cache = {}
            except Exception as e:
                logger.error(f"Failed to load cache: {str(e)}")
                self.cache = {}

    async def _save_cache_to_file(self):
        cache_copy = copy.deepcopy(self.cache)

        if cache_copy is not None:
            try:
                async with aiofiles.open(self.cache_path, "w") as f:
                    await f.write(json.dumps(cache_copy, indent=2))
            except Exception as e:
                logger.error(f"Failed to save cache: {str(e)}")

    def _save_cache_to_file_sync(self):
        cache_copy = copy.deepcopy(self.cache)

        if cache_copy is not None:
            try:
                with open(self.cache_path, "w") as f:
                    f.write(json.dumps(cache_copy, indent=2))
            except Exception as e:
                logger.error(f"Failed to save cache: {str(e)}")

    async def generate(
        self,
        prompt: str,
        return_format: str = None,
        valid_func: Optional[Callable] = lambda x: True,
        format_func: Optional[Callable] = None,
        return_response=False,
        **kwargs,
    ) -> Optional[Any]:
        """
        Async version of generate method.
        """
        if return_format == "str":
            logger.debug("return str response")
            format_func = format_func or (lambda x: x)
        elif return_format == "list":
            format_func = format_func or parse_md_list
        elif return_format == "dict":
            format_func = format_func or parse_md_dict
        elif return_format == "indent_list":
            format_func = format_func or parse_indented_list
        elif return_format == "json":
            format_func = format_func or parse_json_from_response
        elif format_func is None:
            raise ValueError(
                "format_func must be provided if return_format is not provided"
            )

        prompt_hash = hash_string(prompt)
        # async with self.cache_lock:
        if prompt_hash in self.cache:
            response = self.cache[prompt_hash]
            total_res = format_func(response)
            if return_response:
                return total_res, response
            return total_res

        begin_time = time.time()
        if self.model_name in ["qwen3-8b", "deepseek-r1-distill-qwen-32b-quantized"]:
            result = await self._generate_response_reasoning(prompt, **kwargs)
            reasoning_content = result["reasoning_content"]
        else:
            result = await self._generate_response_no_reasoning(prompt, **kwargs)
        
        response = result["content"]

        max_retries = self.llm_config.valid_max_retries + 1
        total_res = None
        item_buffer = None
        for i in range(max_retries):
            cleaned_response = format_func(response)

            if not total_res:
                item_buffer = copy.deepcopy(cleaned_response)

            if return_format == "list":
                item_buffer = cleaned_response
                total_res = item_buffer
            elif return_format == "dict":
                item_buffer.update(cleaned_response)
                total_res = item_buffer
            elif return_format == "json":
                total_res = cleaned_response
            else:
                total_res = cleaned_response

            if cleaned_response and valid_func(total_res):
                end_time = time.time()
                if i > 0:
                    logger.debug(
                        f"return response, using {i+1} attempts, costing {end_time - begin_time} seconds"
                    )
                else:
                    logger.debug(
                        f"return response, costing {end_time - begin_time} seconds"
                    )
                
                self.update_time += 1
                if self.update_time % 100 == 0:
                    self.update_time = 0
                    self._save_cache_to_file_sync()

                if return_response:
                    if self.model_name in ["qwen3-8b", "deepseek-r1-distill-qwen-32b-quantized"]:
                        return total_res or item_buffer, response, reasoning_content
                    else:
                        return total_res or item_buffer, response
                return total_res or item_buffer
            else:
                kwargs.pop("temperature", None)
                response = await self._generate_response(
                    prompt, temperature=1, **kwargs
                )
                logger.debug(
                    "valid failed, retry with temperature=1\n"
                    f"prompt:\n{prompt}\n"
                    f"raw_response:\n{response}"
                )

        if not cleaned_response:
            logger.error(f"prompt: {prompt}")
            logger.error(f"response: {response}")
            logger.error(f"Failed to clean response after {max_retries} attempts.")
            if return_response:
                if self.model_name in ["qwen3-8b", "deepseek-r1-distill-qwen-32b-quantized"]:
                    return total_res, response, reasoning_content
                else:
                    return total_res, response
            return None

        logger.error(f"prompt: {prompt}")
        logger.error(f"response: {response}")
        logger.error(f"Failed to clean response after {max_retries} attempts.")
        if return_response:
            if self.model_name in ["qwen3-8b", "deepseek-r1-distill-qwen-32b-quantized"]:
                return total_res, response, reasoning_content
            else:
                return total_res, response
        return total_res

    async def _generate_response(self, prompt: str, **kwargs) -> str:
        """
        Async version of _generate_response method.
        """
        max_tokens = (
            kwargs.get("max_tokens")
            or kwargs.get("max_completion_tokens")
            or self.llm_config.max_tokens
        )
        temperature = kwargs.get("temperature") or self.llm_config.temperature
        backoff_time = 1

        while True:
            try:
                messages = [{"role": "user", "content": prompt}]
                async with self.request_semaphore:
                    if "gemma" in self.model_name:
                        response = await self.acompletion(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stop=["<eos>", "<end of turn>"],
                        )
                    else:
                        response = await self.acompletion(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            timeout=self.llm_config.request_timeout,
                        )

                logger.debug(
                    "prompt:\n"
                    + "#" * 80
                    + "\n"
                    + f"{prompt}\n"
                    + "#" * 80
                    + "\n"
                    + f"response (temperature: {temperature}):\n"
                    + f"{response.choices[0].message.content}\n"
                    + "#" * 80
                    + "\n"
                )
                cost_logger.debug(
                    f"completion_tokens: {response.usage.completion_tokens}"
                )
                cost_logger.debug(f"prompt_tokens: {response.usage.prompt_tokens}")
                cost_logger.debug(f"total_tokens: {response.usage.total_tokens}")
                break
            except Exception as e:
                logger.error(f"API error: {str(e)}\nWaiting for {backoff_time} seconds")
                logger.debug(
                    "current args:\n"
                    f"model: {self.model_name}\n"
                    f"max_tokens: {max_tokens}\n"
                    f"temperature: {temperature}\n"
                    f"timeout: {self.llm_config.request_timeout}\n"
                )
                await asyncio.sleep(backoff_time)
                backoff_time = min(3, backoff_time * 1.5)
        return response.choices[0].message.content
    
    async def _generate_response_sample(self, prompt: str, **kwargs) -> str:
        """
        Async version of _generate_response method.
        """
        max_tokens = (
            kwargs.get("max_tokens")
            or kwargs.get("max_completion_tokens")
            or self.llm_config.max_tokens
        )
        temperature = kwargs.get("temperature") or self.llm_config.temperature
        n = kwargs.get("n", 1)
        backoff_time = 1

        while True:
            try:
                messages = [{"role": "user", "content": prompt}]
                async with self.request_semaphore:
                    if "gemma" in self.model_name:
                        response = await self.acompletion(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stop=["<eos>", "<end of turn>"],
                            n=n,
                        )
                    else:
                        response = await self.acompletion(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            timeout=self.llm_config.request_timeout,
                            n=n,
                        )
                if n == 1:
                    return response.choices[0].message.content
                else:
                    # 返回所有样本的结果
                    return [choice.message.content for choice in response.choices]
            except Exception as e:
                logger.error(f"API error: {str(e)}\nWaiting for {backoff_time} seconds")
                await asyncio.sleep(backoff_time)
                backoff_time = min(3, backoff_time * 1.5)

    async def _generate_response_reasoning(self, prompt: str, **kwargs) -> dict:
        """
        Async version of _generate_response method.
        """
        max_tokens = (
            kwargs.get("max_tokens")
            or kwargs.get("max_completion_tokens")
            or self.llm_config.max_tokens
        )
        temperature = kwargs.get("temperature") or self.llm_config.temperature
        backoff_time = 1

        while True:
            try:
                messages = [{"role": "user", "content": prompt}]
                async with self.request_semaphore:
                    if "gemma" in self.model_name:
                        response = await self.acompletion(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stop=["<eos>", "<end of turn>"],
                        )
                    else:
                        response = await self.acompletion(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            timeout=self.llm_config.request_timeout,
                        )

                logger.debug(
                    "prompt:\n"
                    + "#" * 80
                    + "\n"
                    + f"{prompt}\n"
                    + "#" * 80
                    + "\n"
                    + f"response (temperature: {temperature}):\n"
                    + f"{response.choices[0].message.content}\n"
                    + f"reasoning_content:\n"
                    + f"{response.choices[0].message.reasoning_content}\n"
                    + "#" * 80
                    + "\n"
                )
                cost_logger.debug(
                    f"completion_tokens: {response.usage.completion_tokens}"
                )
                cost_logger.debug(f"prompt_tokens: {response.usage.prompt_tokens}")
                cost_logger.debug(f"total_tokens: {response.usage.total_tokens}")
                break
            except Exception as e:
                logger.error(f"API error: {str(e)}\nWaiting for {backoff_time} seconds")
                logger.debug(
                    "current args:\n"
                    f"model: {self.model_name}\n"
                    f"max_tokens: {max_tokens}\n"
                    f"temperature: {temperature}\n"
                    f"timeout: {self.llm_config.request_timeout}\n"
                )
                await asyncio.sleep(backoff_time)
                backoff_time = min(3, backoff_time * 1.5)
        result = dict()
        result["content"] = response.choices[0].message.content
        result["reasoning_content"] = response.choices[0].message.reasoning_content
        return result
    
    async def _generate_response_no_reasoning(self, prompt: str, **kwargs) -> dict:
        """
        Async version of _generate_response method.
        """
        max_tokens = (
            kwargs.get("max_tokens")
            or kwargs.get("max_completion_tokens")
            or self.llm_config.max_tokens
        )
        temperature = kwargs.get("temperature") or self.llm_config.temperature
        backoff_time = 1

        while True:
            try:
                messages = [{"role": "user", "content": prompt}]
                async with self.request_semaphore:
                    if "gemma" in self.model_name:
                        response = await self.acompletion(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stop=["<eos>", "<end of turn>"],
                        )
                    else:
                        response = await self.acompletion(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            timeout=self.llm_config.request_timeout,
                        )

                logger.debug(
                    "prompt:\n"
                    + "#" * 80
                    + "\n"
                    + f"{prompt}\n"
                    + "#" * 80
                    + "\n"
                    + f"response (temperature: {temperature}):\n"
                    + f"{response.choices[0].message.content}\n"
                    + "#" * 80
                    + "\n"
                )
                cost_logger.debug(
                    f"completion_tokens: {response.usage.completion_tokens}"
                )
                cost_logger.debug(f"prompt_tokens: {response.usage.prompt_tokens}")
                cost_logger.debug(f"total_tokens: {response.usage.total_tokens}")
                break
            except Exception as e:
                logger.error(f"API error: {str(e)}\nWaiting for {backoff_time} seconds")
                logger.debug(
                    "current args:\n"
                    f"model: {self.model_name}\n"
                    f"max_tokens: {max_tokens}\n"
                    f"temperature: {temperature}\n"
                    f"timeout: {self.llm_config.request_timeout}\n"
                )
                await asyncio.sleep(backoff_time)
                backoff_time = min(3, backoff_time * 1.5)
        result = dict()
        result["content"] = response.choices[0].message.content
        return result