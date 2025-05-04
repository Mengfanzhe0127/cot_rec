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
from src.model.memory import (
    MemoryConfig,
    AsyncUserMemory,
    AsyncGeneralMemory,
)
from src.model.expert import (
    GlobalSearchExpertModel,
    UccrExpertModel,
)
from src.model.embedding import EmbeddingModel
from src.model.info_manager import InfoManager
from src.task.data import split_test
from src.metric import calculate_metrics

from src.utils.logger import setup_logger
from src.utils.data import parse_md_list

logger = setup_logger(__name__, "memoCRS.log")
memory_bank_logger = setup_logger("memory_bank_logger", "memory_bank.log")
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


class AsyncMemoCRS:
    def __init__(
        self,
        eval_config: "EvalConfig",
        llm_client: AsyncLLMClient,
        memory_config: MemoryConfig,
        history_conversations: List = None,
        user_history: Dict[str, List] = None,
        id2name: Dict = None,
        item_pool: List = None,
        movie_info: Dict = None,
        **kwargs,
    ):
        self.llm_client = llm_client
        self.cache_for_clean_prediction = dict()

        # Keep ThreadPoolExecutor for CPU-bound tasks
        self.max_workers = eval_config.num_threads

        self.memory_config = memory_config
        self.eval_config = eval_config

        self.user_history = user_history
        self.history_conversations = history_conversations
        self.id2name = id2name
        self.item_pool = item_pool
        # self.info_manager = InfoManager(movie_info) if movie_info else None
        self.kwargs = kwargs

        language = "zh" if "tg" in self.eval_config.dataset else "en"
        prompt_method = kwargs.get("prompt", language)
        self.language = "zh" if "tg" in self.eval_config.dataset else "en"
        self.info_manager = InfoManager(movie_info, language) if movie_info else None
        self.prompt_manager = PromptManager(method=prompt_method)

        # 合并记忆
        # self.memory_path = "/home/wangxiaolei/mengfanzhe/MCTS_CRS_CHECK/MCTS_CRS-cold/data/crslab_redial/process_all/processed_results_top_1.json"
        self.memory_path = "/home/wangxiaolei/mengfanzhe/MCTS_CRS_CHECK/MCTS_CRS-cold/data/crslab_redial/process_all_key/processed_results_top_10.json"
        # self.memory_path = "/home/wangxiaolei/mengfanzhe/MCTS_CRS_CHECK/MCTS_CRS-cold/data/crslab_redial/process_all_new/new_merge_results_top_10.json"

        self.init_modules()

    def init_modules(self):
        self.user_memory = None
        if self.eval_config.use_memory:
            if not hasattr(self, "embed_model"):
                self.embed_model = EmbeddingModel(
                    gpu_id=self.eval_config.embedding_gpu_id
                )
            self.user_memory = AsyncUserMemory(
                self.llm_client, embed_model=self.embed_model, config=self.memory_config
            )

        self.general_memory = None
        if self.eval_config.collaborative_knowledge or self.eval_config.guide_line:
            if self.eval_config.expert_model == "global":
                if not hasattr(self, "embed_model"):
                    self.embed_model = EmbeddingModel(
                        gpu_id=self.eval_config.embedding_gpu_id
                    )
                expert_model = GlobalSearchExpertModel(
                    embedding_model=self.embed_model, user_history=self.user_history
                )
            else:
                expert_model = UccrExpertModel(
                    dataset=self.eval_config.dataset, id2name=self.id2name
                )

            self.general_memory = AsyncGeneralMemory(
                expert_model=expert_model,
                llm_client=self.llm_client,
                item_pool=self.item_pool,
                user_history=self.user_history,
                config=self.memory_config,
            )

    async def _process_user_memory_async(
        self, user_id: str, user_convs: List[Dict], cache_path: str
    ) -> None:
        """Process memory for a single user asynchronously"""
        try:
            # Create a new UserMemory instance for this user
            user_specific_memory = AsyncUserMemory(
                llm_client=self.llm_client,
                embed_model=self.embed_model,
                config=self.user_memory.config,
            )

            # Load existing memory for this user if exists
            user_cache_path = path.join(cache_path, f"{user_id}.json")
            if path.exists(user_cache_path):
                await user_specific_memory.load_memory(user_cache_path)

            # Initialize or update memory for this user
            if not user_specific_memory.memory_banks:
                # Run CPU-intensive tasks in thread pool
                await user_specific_memory.init_memory(user_convs)
                logger.info(
                    f"Initialized memory for user {user_id}, convs: {len(user_convs)}"
                )
                await user_specific_memory.save_memory(user_cache_path)

            _memory = user_specific_memory.get_memory(user_id)
            await self.user_memory.set_memory(user_id, _memory)

        except Exception as e:
            logger.error(f"Error processing memory for user {user_id}: {str(e)}")
            raise

    async def update_guideline(self, datas) -> None:
        for d in tqdm(datas, desc="Updating guideline"):
            tasks = []
            samples = split_test(d, self.id2name)
            for sample in samples:
                tasks.append(
                    self._update_guideline_task(sample, messages=d["messages"])
                )

            await asyncio.gather(*tasks)
            # logger.info(f"Updated guideline for {len(samples)} samples")

    async def _update_guideline_task(self, data, messages) -> None:
        user_id = data["user_id"]
        dialog_str = self.prompt_manager.messages2str(data["messages"])

        mentioned_movies = []
        for message in data["messages"]:
            mentioned_movies.extend(message["movie_list"])
        mentioned_movies = list(set(mentioned_movies))
        mentioned_movie_info = self.info_manager.get_info(mentioned_movies)

        entity_attitude_dict = {}
        if self.eval_config.use_memory:
            if "tg" in self.eval_config.dataset:
                entity_attitude_dict = await self.user_memory.retrieve_memories_zh(
                    str(user_id), data, 3
                )
            else:
                entity_attitude_dict = await self.user_memory.retrieve_memories(
                str(user_id), data, 3, mentioned_movies_info=mentioned_movie_info
                )
            if not entity_attitude_dict:
                logger.debug(f"cold_start: {user_id}")
            else:
                logger.debug(f"entity_attitude_dict: {entity_attitude_dict}")

        collaborative_item_list = []
        origin_collaborative_item_list = []
        if self.eval_config.collaborative_knowledge:
            collaborative_item_list = (
                await self.general_memory.get_collaborative_knowledge(data)
            )
            origin_collaborative_item_list = collaborative_item_list.copy()

        if self.eval_config.label_engaging:
            collaborative_item_list.extend([k for k in data["rec_label_list"]])

        if self.info_manager:
            collaborative_item_list = self.info_manager.get_info(
                collaborative_item_list,
                info_type=self.eval_config.info_type,
            )

        if self.eval_config.candidate_num:
            collaborative_item_list = dict(
                list(collaborative_item_list.items())[: self.eval_config.candidate_num]
            )

        reasoning_guidelines = None
        if self.eval_config.guide_line:
            reasoning_guidelines = self.general_memory.reasoning_guidelines

        prompt = self.prompt_manager.construct_prompt(
            entity_attitude_dict=entity_attitude_dict,
            collaborative_item_list=collaborative_item_list,
            reasoning_guidelines=reasoning_guidelines,
            dialog_str=dialog_str,
            mentioned_movie_info=mentioned_movie_info,
        )

        if self.kwargs.get("raw_uccr", None):
            # testing uccr
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            item_list = (
                collaborative_item_list.keys()
                if isinstance(collaborative_item_list, dict)
                else collaborative_item_list
            )
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        else:
            # Run LLM generation in thread pool
            def valid_func(res):
                if isinstance(res, list) and len(res) >= 20:
                    return True
                return False

            item_list, response = await self.llm_client.generate(
                prompt,
                return_format="list",
                valid_func=valid_func,
                return_response=True,
            )

        # Clean predictions in thread pool
        cleaned_item_list = await asyncio.to_thread(self.clean_prediction, item_list)
        cleaned_item_list = cleaned_item_list[:20]

        results = {}
        for label in data["rec_label_list"]:
            if label not in cleaned_item_list:
                continue
            current_tid = data["messages"][-1]["turn_id"]
            if len(messages) <= int(current_tid) + 2:
                continue
            recommender_res = messages[int(current_tid) + 1]["text"]
            user_res = (
                messages[int(current_tid) + 2]["text"] or "Thanks, I will check it out."
            )
            # reasoning_trace = (
            #     f"{prompt}\n\n" f"Recommender: {recommender_res}\n" f"User: {user_res}"
            # )
            if 'tg' in self.memory_config.dataset:
                reasoning_trace = (
                    f"{prompt}\n\n" f"推荐系统: {recommender_res}\n" f"用户: {user_res}"
                )
            else:
                reasoning_trace = (
                    f"{prompt}\n\n" f"Recommender: {recommender_res}\n" f"User: {user_res}"
                )

            await self.general_memory.update_guidelines(reasoning_trace=reasoning_trace)

            predict_logger.debug(
                "#" * 80
                + "\n"
                + f"prompt:\n{prompt}\n"
                + "#" * 80
                + "\n"
                + f"response:\n{response}\n"
                + "#" * 80
                + "\n"
                + f"\ncollaborative_item_list: {collaborative_item_list if isinstance(collaborative_item_list, list) else collaborative_item_list.keys()}\n"
                f"uccr_pred: {origin_collaborative_item_list}\n"
                f"cleaned_item_list: {cleaned_item_list}\n"
                f"label: {label}\n"
                f"conv_id: {data['conv_id']}\n"
                f"turn_id: {data['messages'][-1]['turn_id']}"
            )

        logger.debug(f"results: {results}")

        return results

    async def update_history_async(self, dataset: str) -> None:
        """Update history using async approach"""
        cache_base_path = f"cache/user_memory_{dataset}/"
        if path.exists(path.join(cache_base_path, "complete.json")):
            logger.info("Memory already updated")
            await self.user_memory.load_memory(
                path.join(cache_base_path, "complete.json")
            )
            memory_bank_logger.debug(
                "\n################################################################################\n".join(
                    [
                        f"User {u} memory:\n"
                        + "\n".join(
                            [
                                f"{e}: {v['attitude']}, {v['timestamp']}"
                                for e, v in self.user_memory.memory_banks[u].items()
                            ]
                        )
                        for u in self.user_memory.memory_banks.keys()
                    ]
                )
            )
            return

        os.makedirs(cache_base_path, exist_ok=True)

        # 同步debug用
        #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # for user_id, user_convs in tqdm(
        #     self.user_history.items(),
        #     desc="Processing user memories",
        #     total=len(self.user_history),
        # ):
        #     try:
        #         # Call the synchronous version of memory processing
        #         # Note: You'll need to create a non-async version of _process_user_memory_async
        #         await self._process_user_memory_async(user_id, user_convs, cache_base_path)
        #         logger.debug(f"Processed memory for user {user_id}")
        #     except Exception as e:
        #         logger.error(f"Error processing user {user_id}: {str(e)}")
        #         continue
        # # Create tasks for each user's memory processing
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        tasks = []
        for user_id, user_convs in self.user_history.items():
            task = asyncio.create_task(
                self._process_user_memory_async(user_id, user_convs, cache_base_path)
            )
            tasks.append(task)
        # Wait for all tasks to complete with progress tracking
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Processing user memories",
        ):
            await task

        # Save the complete memory banks
        await self.user_memory.save_memory(path.join(cache_base_path, "complete.json"))
        memory_bank_logger.debug(
            "\n################################################################################\n".join(
                [
                    f"User {u} memory:\n"
                    + "\n".join(
                        [
                            f"{e}: {v['attitude']}, {v['timestamp']}"
                            for e, v in self.user_memory.memory_banks[u].items()
                        ]
                    )
                    for u in self.user_memory.memory_banks
                ]
            )
        )
        logger.info(f"Completed memory update for {len(self.user_history)} users")

    async def _get_rec_prompt_parts(
        self, data, eval_config=None, user_preference: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Get recommendation prompt"""
        if not eval_config:
            eval_config = self.eval_config
        user_id = data["user_id"]
        dialog_str = self.prompt_manager.messages2str(data["messages"])

        mentioned_movies = []
        for message in data["messages"]:
            mentioned_movies.extend(message["movie_list"])
        mentioned_movies = list(set(mentioned_movies))
        mentioned_movie_info = self.info_manager.get_info(mentioned_movies)

        entity_attitude_dict = {}
        if eval_config.use_memory:
            entity_attitude_dict = await self.user_memory.retrieve_memories(
                str(user_id), data, 3, mentioned_movies_info=mentioned_movie_info
            )
            if not entity_attitude_dict:
                logger.debug(f"cold_start: {user_id}")
            else:
                logger.debug(f"entity_attitude_dict: {entity_attitude_dict}")
            entity_attitude_dict = await self.user_memory.retrieve_memories(
                str(user_id), data, 3, mentioned_movies_info=mentioned_movie_info
            )
            if not entity_attitude_dict:
                logger.debug(f"cold_start: {user_id}")
            else:
                logger.debug(f"entity_attitude_dict: {entity_attitude_dict}")

        collaborative_item_list = []
        origin_collaborative_item_list = []
        if eval_config.collaborative_knowledge:
            collaborative_item_list = (
                await self.general_memory.get_collaborative_knowledge(data)
            )
            origin_collaborative_item_list = collaborative_item_list.copy()

        if eval_config.label_engaging:
            collaborative_item_list.extend([k for k in data["rec_label_list"]])

        if self.info_manager:
            collaborative_item_list = self.info_manager.get_info(
                collaborative_item_list,
                info_type=eval_config.info_type,
            )

        if eval_config.candidate_num:
            collaborative_item_list = dict(
                list(collaborative_item_list.items())[: eval_config.candidate_num]
            )

        reasoning_guidelines = None
        if eval_config.guide_line:
            reasoning_guidelines = self.general_memory.reasoning_guidelines

        return {
            "entity_attitude_dict": entity_attitude_dict,
            "collaborative_item_list": collaborative_item_list,
            "reasoning_guidelines": reasoning_guidelines,
            "dialog_str": dialog_str,
            "mentioned_movie_info": mentioned_movie_info,
        }

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
            # 按action类型分组
            action_groups = {}
            for msg in messages_in_group:

                action = msg.get("type")
                if action not in action_groups:
                    action_groups[action] = []
                action_groups[action].append(msg)
        
            # 处理每种action类型
            for action, msgs in action_groups.items():
                if action == "conversation":
                    # 构建对话格式
                    conv_lines = [f"{i}. Conversation"]
                    for msg in msgs:
                        sender = msg.get("sender")
                        text = msg.get("text")
                        conv_lines.append(f"{sender}: {text}")
                    dialog_blocks.append("\n".join(conv_lines))
            
                elif action == "click":
                    click_lines = [f"{i}. Click"]
                    for msg in msgs:
                        text = msg.get("text")
                        click_lines.append(text)
                    dialog_blocks.append("\n".join(click_lines))

        dialog_str = "\n\n".join(dialog_blocks)

        # dialog_str += f"\n\n**Accepted Movie**: {target_movie}"

        prompt = self.prompt_manager.construct_prompt(
            entity_attitude_dict=None,
            collaborative_item_list=None,
            reasoning_guidelines=None,
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
                predict_logger.warning(f"响应缺少必要部分: Analysis部分存在={has_analysis}, User Preferences部分存在={has_preferences}")
                return False
            
             # 获取User Preferences部分的文本
            preferences_position = response_text.find(preferences_marker)
            preferences_text = response_text[preferences_position + len(preferences_marker):]
    
            # 检查User Preferences中的结构
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
                predict_logger.warning(f"User Preferences缺少必要结构: {', '.join(missing_elements)}")
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
                predict_logger.warning("User Preferences中结构顺序不正确")
                predict_logger.warning(f"positions: movies={movies_pos}, m_like={movies_like_pos}, m_dislike={movies_dislike_pos}, "
                              f"attrs={attributes_pos}, a_like={attributes_like_pos}, a_dislike={attributes_dislike_pos}")
                return False
            
            return True


        _, response = await self.llm_client.generate(
            prompt,
            return_format="str",  # 解析模型返回的json
            return_response=True,
            valid_func=custom_validation_func,
        )

        predict_logger.debug(
            # f"Annotated JSON:\n{json.dumps(annotated_json, indent=4)}\n"
            f"Original Prompt:\n{prompt}\n"
            f"LLM Response:\n{response}\n"
            # f"Reasoning Content:\n{reasoning_content}\n"
            f"Conversation ID: {conversation_id}\n"
            f"ID: {id}\n"
            f"target_movie: {target_movie}\n"
            # f"Message ID: {messageId}\n"
            # f"Initiator Worker ID: {user_id}"
        )

        output_file = "/home/wangxiaolei/mengfanzhe/COT2/data/filter_user_v_t/new_fvt_res_2_gas-2.jsonl"

        final_json["cot_result"] = response
        with open(output_file, "a") as f:
            f.write(json.dumps(final_json) + "\n")
        return final_json

    def metric_label(self, item_list, labels):
        cleaned_item_list = self.clean_prediction(item_list)
        results = {}
        for label in labels:
            res = calculate_metrics(cleaned_item_list, label)
            for key, value in res.items():
                if key not in results:
                    results[key] = []
                results[key].append(value)
        return results

    async def parallel_pipeline_async(self, valid_data):
        """Main async pipeline with controlled concurrency"""
        if self.eval_config.use_memory and self.history_conversations:
            await self.update_history_async(dataset=self.eval_config.dataset)

        # total_res = defaultdict(list)

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
            # results = await task
            # for key, values in results.items():
            #     total_res[key].extend(values)
            await task

                # logger.error(f"Error in task: {str(e)}")
                # exit(0)
        logger.info("All tasks completed. Results stored in JSONL file.")
        # avg_res = {f"{key}_avg": np.mean(values) for key, values in total_res.items()}
        #
        # return total_res, avg_res

    def clean_prediction(self, pred_item_list) -> List[str]:
        """Keep the original clean_prediction method for CPU-bound operations"""
        cleaned_pred_item_list = []
        for item in pred_item_list:
            if item in self.cache_for_clean_prediction:
                cleaned_item = self.cache_for_clean_prediction[item]
            else:
                if item in self.item_pool:
                    cleaned_item = item
                else:
                    stripped_item = re.sub(r"\s*\(\d{4}\)$", "", item)
                    if stripped_item in self.item_pool:
                        cleaned_item = stripped_item
                    else:
                        cleaned_item = process.extractOne(
                            item,
                            self.item_pool,
                            scorer=fuzz.ratio,
                            processor=utils.default_process,
                        )[0]

            cleaned_pred_item_set = set(cleaned_pred_item_list)
            if cleaned_item in cleaned_pred_item_set:
                item_set = set(self.item_pool) - cleaned_pred_item_set
                cleaned_item = process.extractOne(
                    item,
                    item_set,
                    scorer=fuzz.ratio,
                    processor=utils.default_process,
                )[0]
            else:
                self.cache_for_clean_prediction[item] = cleaned_item

            cleaned_pred_item_list.append(cleaned_item)
        return cleaned_pred_item_list

    @staticmethod
    def user_liked_movies(convs: List) -> List[str]:
        movie_name = []
        for c in convs:
            for m in reversed(c["messages"]):
                if m["role"] == "recommender" and m["movie_list"]:
                    movie_name.extend(m["movie_list"])
        return list(set(movie_name))

    @staticmethod
    def get_movies_from_convs(convs: List) -> List[str]:
        movie_name = []
        for c in convs:
            for m in c["messages"]:
                movie_name.extend(m["movie_list"])
        return list(set(movie_name))
