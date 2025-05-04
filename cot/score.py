import os
import re
from os import path
import json
import hashlib
import numpy as np
import asyncio
from typing import List, Dict, Any, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from numba import jit
from tqdm import tqdm
from rapidfuzz import process, fuzz, utils

from src.prompts.prompt_manager import PromptManager
from src.model.client import AsyncLLMClient

from src.utils.logger import setup_logger

logger = setup_logger(__name__, "scoreCRS.log")
predict_logger = setup_logger("predict_logger", "predict.log")


@jit(nopython=True)
def hash_string_numba(s: bytes) -> int:
    h = 0
    for b in s:
        h = (h * 31 + b) & 0xFFFFFFFF
    return h


def hash_string(s: str) -> str:
    num_hash = hash_string_numba(s.encode())
    return hashlib.sha256(str(num_hash).encode()).hexdigest()


def save_res(res: Dict, name: str = "res"):
    with open(f"result/{name}.json", "w") as f:
        json.dump(res, f, indent=2)


class AsyncCRS:
    def __init__(
        self,
        eval_config,
        llm_client: AsyncLLMClient,
        **kwargs,
    ):
        self.llm_client = llm_client
        self.cache_for_clean_prediction = dict()

        # Keep ThreadPoolExecutor for CPU-bound tasks
        self.max_workers = eval_config.num_threads
        self.eval_config = eval_config
        self.kwargs = kwargs
        self.method = self.eval_config.method

        self.prompt_manager = PromptManager(method=self.method)


    async def _process_single_data_async(self, data):
        """Process single data asynchronously"""
        initiator_worker_id = data["initiatorWorkerId"]
        conversation_id = data["conversationId"]
        messages = data["messages"]
        target_movie = data["target_movie"]
        id = data["id"]
        cot_results = data["cot_results"]
    
        # processing interaction history
        conversation_groups = defaultdict(list)
        for msg in messages:
            conv_id = msg.get("conversationId")
            conversation_groups[conv_id].append(msg)

        sorted_conv_ids_int = sorted(int(conv_id) for conv_id in conversation_groups.keys())
    
        dialog_blocks = []
        for i, conv_id in enumerate(sorted_conv_ids_int, 1):
            conv_id_str = conv_id
            messages_in_group = conversation_groups[conv_id_str]
    
            # split by action type (conversation, click)
            action_groups = {}
            for msg in messages_in_group:
                action = msg.get("type")
                if action not in action_groups:
                    action_groups[action] = []
                action_groups[action].append(msg)
            
            for action, msgs in action_groups.items():
                if action == "conversation":
                    conv_lines = [f"Interaction {i}. "]
                    for msg in msgs:
                        sender = msg.get("sender")
                        text = msg.get("text")
                        conv_lines.append(f"{sender}: {text}")
                    dialog_blocks.append("\n".join(conv_lines))
        
                elif action == "click":
                    click_lines = [f"Interaction {i}. "]
                    for msg in msgs:
                        text = msg.get("text")
                        click_lines.append(text)
                    dialog_blocks.append("\n".join(click_lines))

        formatted_interactions = "\n\n".join(dialog_blocks)
    
        # format analyses
        formatted_analyses = []
        for i, cot_result in enumerate(cot_results, 1):
            formatted_analyses.append(f"Preference Analysis {i}:\n\n{cot_result}")
    
        formatted_preference_analyses = "\n\n".join(formatted_analyses)
    
        # final instruction (without label)
        complete_prompt = f"{formatted_interactions}\n\n## 2. Multiple User Preference Analyses:\n\n{formatted_preference_analyses}"

        prompt = self.prompt_manager.construct_prompt(
            dialog_str=complete_prompt,
        )

        def custom_validation_func(response_text):
            if not isinstance(response_text, str):
                return False
        
            reason_marker = "### Reason:"
            ranked_marker = "### Ranked Preference Analyses:"
        
            has_reason = reason_marker in response_text
            has_ranked = ranked_marker in response_text
        
            if not (has_reason and has_ranked):
                predict_logger.warning(f"Response missing necessary parts: Reason part exists={has_reason}, Ranked part exists={has_ranked}")
                return False

            ranked_position = response_text.find(ranked_marker)
            ranked_text = response_text[ranked_position + len(ranked_marker):]

            analyses_count = len(cot_results)
            score_pattern = r"Preference Analysis \d+, Score: \d+"
            matches = re.findall(score_pattern, ranked_text)
        
            if len(matches) != analyses_count:
                predict_logger.warning(f"expect {analyses_count} scores, but found {len(matches)}")
                return False
        
            return True

        _, response = await self.llm_client.generate(
            prompt,
            return_format="str",
            return_response=True,
            valid_func=custom_validation_func,
        )

        predict_logger.debug(
            f"Original Prompt:\n{prompt}\n\n"
            f"LLM Response:\n{response}\n"
            f"Conversation ID: {conversation_id}\n"
            f"ID: {id}\n"
            f"target_movie: {target_movie}\n"
        )

        output_file = self.eval_config.output_file
    
        final_json = {
            "initiatorWorkerId": initiator_worker_id,
            "conversationId": conversation_id,
            "target_movie": target_movie,
            "id": id,
            "response": response
        }
    
        with open(output_file, "a") as f:
            f.write(json.dumps(final_json) + "\n")
        
        return final_json

    async def parallel_pipeline_async(self, valid_data):
        """Main async pipeline with controlled concurrency"""

        semaphore = asyncio.Semaphore(self.eval_config.max_concurrent_requests)

        async def process_with_semaphore(data):
            async with semaphore:
                return await self._process_single_data_async(data)

        tasks = [
            asyncio.create_task(process_with_semaphore(data)) for data in valid_data
        ]

        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Processing conversations",
        ):
            await task
        logger.info("All tasks completed. Results stored in JSONL file.")