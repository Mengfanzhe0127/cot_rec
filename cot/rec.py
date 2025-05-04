import json
import hashlib
import asyncio
from typing import Dict
from collections import defaultdict
from numba import jit
from tqdm import tqdm

from src.prompts.prompt_manager import PromptManager
from src.model.client import AsyncLLMClient

from src.utils.logger import setup_logger

logger = setup_logger(__name__, "recCRS.log")
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
        final_json = {"initiatorWorkerId": initiator_worker_id, "conversationId": conversation_id, "target_movie": target_movie, "id": id}

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

        dialog_str = "\n\n".join(dialog_blocks)

        prompt = self.prompt_manager.construct_prompt(
            dialog_str=dialog_str,
        )

        def custom_validation_func(response_text):
            if not isinstance(response_text, str):
                return False
            
            analysis_marker = "### Analysis:"
            preferences_marker = "### User Preferences:"
    
            has_analysis = analysis_marker in response_text
            has_preferences = preferences_marker in response_text

            if not (has_analysis and has_preferences):
                predict_logger.warning(f"The response is missing the necessary part: **Analysis** part exists={has_analysis}, **User Preferences** part exists={has_preferences}")
                return False

            preferences_position = response_text.find(preferences_marker)
            preferences_text = response_text[preferences_position + len(preferences_marker):]

            required_elements = [
                "- movies:", 
                "  - like:", 
                "  - dislike:", 
                "- attributes:", 
                "  - like:", 
                "  - dislike:"
            ]
    
            missing_elements = []
            for element in required_elements:
                if element not in preferences_text:
                    missing_elements.append(element)
    
            if missing_elements:
                predict_logger.warning(f"User Preferences lacks necessary structure: {', '.join(missing_elements)}")
                return False
    
            # 检查是否按正确顺序包含这些元素
            movies_pos = preferences_text.find("- movies:")
            movies_like_pos = preferences_text.find("  - like:", movies_pos)
            movies_dislike_pos = preferences_text.find("  - dislike:", movies_like_pos)
    
            attributes_pos = preferences_text.find("- attributes:")
            attributes_like_pos = preferences_text.find("  - like:", attributes_pos)
            attributes_dislike_pos = preferences_text.find("  - dislike:", attributes_like_pos)
    
            # 验证所有位置都大于-1（即找到了）且顺序正确
            if not (movies_pos > -1 and movies_like_pos > movies_pos and 
            movies_dislike_pos > movies_like_pos and attributes_pos > -1 and
            attributes_like_pos > attributes_pos and attributes_dislike_pos > attributes_like_pos):
                predict_logger.warning("The order of User Preferences is incorrect")
                predict_logger.warning(f"positions: movies={movies_pos}, m_like={movies_like_pos}, m_dislike={movies_dislike_pos}, "
                              f"attrs={attributes_pos}, a_like={attributes_like_pos}, a_dislike={attributes_dislike_pos}")
                return False
            
            return True
        
        def custom_validation_func_qwen3(response_text):
            if not isinstance(response_text, str):
                return False
            
            preferences_marker = "### User Preferences:"

            has_preferences = preferences_marker in response_text

            if not (has_preferences):
                predict_logger.warning(f"The response is missing the necessary part: **User Preferences** part exists={has_preferences}")
                return False

            preferences_position = response_text.find(preferences_marker)
            preferences_text = response_text[preferences_position + len(preferences_marker):]

            required_elements = [
                "- movies:", 
                "  - like:", 
                "  - dislike:", 
                "- attributes:", 
                "  - like:", 
                "  - dislike:"
            ]
    
            missing_elements = []
            for element in required_elements:
                if element not in preferences_text:
                    missing_elements.append(element)
    
            if missing_elements:
                predict_logger.warning(f"User Preferences lacks necessary structure: {', '.join(missing_elements)}")
                return False

            movies_pos = preferences_text.find("- movies:")
            movies_like_pos = preferences_text.find("  - like:", movies_pos)
            movies_dislike_pos = preferences_text.find("  - dislike:", movies_like_pos)
    
            attributes_pos = preferences_text.find("- attributes:")
            attributes_like_pos = preferences_text.find("  - like:", attributes_pos)
            attributes_dislike_pos = preferences_text.find("  - dislike:", attributes_like_pos)
    
            # 验证所有位置都大于-1（即找到了）且顺序正确
            if not (movies_pos > -1 and movies_like_pos > movies_pos and 
            movies_dislike_pos > movies_like_pos and attributes_pos > -1 and
            attributes_like_pos > attributes_pos and attributes_dislike_pos > attributes_like_pos):
                predict_logger.warning("The order of User Preferences is incorrect")
                predict_logger.warning(f"positions: movies={movies_pos}, m_like={movies_like_pos}, m_dislike={movies_dislike_pos}, "
                              f"attrs={attributes_pos}, a_like={attributes_like_pos}, a_dislike={attributes_dislike_pos}")
                return False
            
            return True

        if self.method.lower() == "rec":
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
        elif self.method.lower() == "reason_rec":
            _, response, reasoning_content = await self.llm_client.generate(
                prompt,
                return_format="str",
                return_response=True,
                valid_func=custom_validation_func_qwen3,
            )

            predict_logger.debug(
                f"Original Prompt:\n{prompt}\n\n"
                f"Reasoning Content:\n{reasoning_content}\n"
                f"LLM Response:\n{response}\n"
                f"Conversation ID: {conversation_id}\n"
                f"ID: {id}\n"
                f"target_movie: {target_movie}\n"
            )

        output_file = self.eval_config.output_file

        final_json["cot_result"] = response
        if self.method.lower() == "reason_rec":
            final_json["reasoning_content"] = reasoning_content
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