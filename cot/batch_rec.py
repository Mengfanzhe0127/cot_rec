import re
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

logger = setup_logger(__name__, "batch_recCRS.log")
predict_logger = setup_logger("predict_logger", "batch_predict.log")


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


class BatchAsyncCRS:
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
        
        self.mode = self.eval_config.mode

    async def parallel_pipeline_async(self, valid_data, n_samples=10):
        """Main async pipeline with controlled concurrency"""

        semaphore = asyncio.Semaphore(self.eval_config.max_concurrent_requests)

        async def process_with_semaphore(data):
            async with semaphore:
                return await self._process_single_data_batch(data, n_samples)

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

    def _validate_response_format(self, response_text, sample_idx):
        if not isinstance(response_text, str):
            predict_logger.warning(f"Sample {sample_idx+1} is not a string")
            return False
    
        analysis_marker = "### Analysis:"
        preferences_marker = "### User Preferences:"

        has_analysis = analysis_marker in response_text
        has_preferences = preferences_marker in response_text

        if not (has_analysis and has_preferences):
            predict_logger.warning(f"sample {sample_idx+1} response missing necessary parts: Analysis part exists={has_analysis}, User Preferences part exists={has_preferences}")
            return False

        analysis_position = response_text.find(analysis_marker)
        preferences_position = response_text.find(preferences_marker)
    
        if analysis_position >= preferences_position:
            predict_logger.warning(f"Sample {sample_idx+1} Analysis and User Preferences's order is incorrect")
            return False

        preferences_text = response_text[preferences_position + len(preferences_marker):].strip()

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
            predict_logger.warning(f"Sample {sample_idx+1} User Preferences lack necessary structure: {', '.join(missing_elements)}")
            return False

        movies_pos = preferences_text.find("- movies:")
        movies_like_pos = preferences_text.find("  - like:", movies_pos)
        movies_dislike_pos = preferences_text.find("  - dislike:", movies_like_pos)

        attributes_pos = preferences_text.find("- attributes:")
        attributes_like_pos = preferences_text.find("  - like:", attributes_pos)
        attributes_dislike_pos = preferences_text.find("  - dislike:", attributes_like_pos)

        if not (movies_pos > -1 and movies_like_pos > movies_pos and 
        movies_dislike_pos > movies_like_pos and attributes_pos > -1 and
        attributes_like_pos > attributes_pos and attributes_dislike_pos > attributes_like_pos):
            predict_logger.warning(f"Sample {sample_idx+1} User Preferences's order is incorrect'")
            predict_logger.warning(f"positions: movies={movies_pos}, m_like={movies_like_pos}, m_dislike={movies_dislike_pos}, "
                      f"attrs={attributes_pos}, a_like={attributes_like_pos}, a_dislike={attributes_dislike_pos}")
            return False
    
        return True
    
    def _extract_all_generated_features(self, user_preferences_text, matched_movies):
        movies_section = re.search(r'- movies:(.*?)(?=- attributes:|$)', user_preferences_text, re.DOTALL)
        attributes_section = re.search(r'- attributes:(.*?)$', user_preferences_text, re.DOTALL)
        
        generated_movies = []
        generated_attributes = []
        all_generated_features = set()

        if movies_section:
            movies_text = movies_section.group(1)

            all_movies_pattern = r'- (?:like|dislike):\s+(.*?)(?=\s+- (?:like|dislike):|$)'
            all_movies_matches = re.finditer(all_movies_pattern, movies_text, re.DOTALL)
            
            for match in all_movies_matches:
                if match.group(1).strip().lower() != "none":
                    movie_text = match.group(1).strip()
                    movie_names = [name.strip() for name in re.split(r',|\n', movie_text) if name.strip()]
                    generated_movies.extend(movie_names)

                    all_generated_features.update([movie.lower() for movie in movie_names])

                    for movie_name in movie_names:
                        movie_info = matched_movies.get(movie_name)
                        if movie_info:
                            if isinstance(movie_info.get('genre'), list):
                                all_generated_features.update([genre.lower() for genre in movie_info['genre']])
                            if movie_info.get('year'):
                                all_generated_features.add(movie_info['year'])
                            if isinstance(movie_info.get('director'), list):
                                all_generated_features.update([director.lower() for director in movie_info['director']])
                            if isinstance(movie_info.get('writer'), list):
                                all_generated_features.update([writer.lower() for writer in movie_info['writer']])
                            if isinstance(movie_info.get('star'), list):
                                all_generated_features.update([star.lower() for star in movie_info['star']])

        if attributes_section:
            attrs_text = attributes_section.group(1)

            all_attrs_pattern = r'- (?:like|dislike):\s+(.*?)(?=\s+- (?:like|dislike):|$)'
            all_attrs_matches = re.finditer(all_attrs_pattern, attrs_text, re.DOTALL)
            
            for match in all_attrs_matches:
                if match.group(1).strip().lower() != "none":
                    attrs_text = match.group(1).strip()
                    attrs = [attr.strip().lower() for attr in re.split(r',|\n', attrs_text) if attr.strip()]
                    generated_attributes.extend(attrs)
                    all_generated_features.update(attrs)
        
        return generated_movies, generated_attributes, list(all_generated_features)
    
    def _extract_combined_features(self, user_preferences_text, matched_movies):
        movies_section = re.search(r'- movies:(.*?)(?=- attributes:|$)', user_preferences_text, re.DOTALL)
        attributes_section = re.search(r'- attributes:(.*?)$', user_preferences_text, re.DOTALL)
        
        all_features = set()
        all_movies = []
        all_attributes = []

        if movies_section:
            movies_text = movies_section.group(1)

            like_movies_match = re.search(r'- like:\s+(.*?)(?=\s+- dislike:|$)', movies_text, re.DOTALL)
            if like_movies_match and like_movies_match.group(1).strip().lower() != "none":
                liked_movies_text = like_movies_match.group(1).strip()
                movie_names = [name.strip() for name in re.split(r',|\n', liked_movies_text) if name.strip()]
                all_movies.extend(movie_names)
                
                for movie_name in movie_names:
                    movie_info = matched_movies.get(movie_name)
                    if movie_info:
                        if isinstance(movie_info.get('genre'), list):
                            all_features.update([genre.lower() for genre in movie_info['genre']])

                        if movie_info.get('year'):
                            all_features.add(movie_info['year'])

                        if isinstance(movie_info.get('director'), list):
                            all_features.update([director.lower() for director in movie_info['director']])

                        if isinstance(movie_info.get('writer'), list):
                            all_features.update([writer.lower() for writer in movie_info['writer']])

                        if isinstance(movie_info.get('star'), list):
                            all_features.update([star.lower() for star in movie_info['star']])

            dislike_movies_match = re.search(r'- dislike:\s+(.*?)(?=$)', movies_text, re.DOTALL)
            if dislike_movies_match and dislike_movies_match.group(1).strip().lower() != "none":
                disliked_movies_text = dislike_movies_match.group(1).strip()
                movie_names = [name.strip() for name in re.split(r',|\n', disliked_movies_text) if name.strip()]
                all_movies.extend(movie_names)
                
                for movie_name in movie_names:
                    movie_info = matched_movies.get(movie_name)
                    if movie_info:
                        if isinstance(movie_info.get('genre'), list):
                            all_features.update([genre.lower() for genre in movie_info['genre']])

                        if movie_info.get('year'):
                            all_features.add(movie_info['year'])

                        if isinstance(movie_info.get('director'), list):
                            all_features.update([director.lower() for director in movie_info['director']])

                        if isinstance(movie_info.get('writer'), list):
                            all_features.update([writer.lower() for writer in movie_info['writer']])

                        if isinstance(movie_info.get('star'), list):
                            all_features.update([star.lower() for star in movie_info['star']])
        
        if attributes_section:
            attrs_text = attributes_section.group(1)

            like_attrs_match = re.search(r'- like:\s+(.*?)(?=\s+- dislike:|$)', attrs_text, re.DOTALL)
            if like_attrs_match and like_attrs_match.group(1).strip().lower() != "none":
                liked_attrs = like_attrs_match.group(1).strip()
                attrs = [attr.strip().lower() for attr in re.split(r',|\n', liked_attrs) if attr.strip()]
                all_attributes.extend(attrs)
                all_features.update(attrs)

            dislike_attrs_match = re.search(r'- dislike:\s+(.*?)(?=$)', attrs_text, re.DOTALL)
            if dislike_attrs_match and dislike_attrs_match.group(1).strip().lower() != "none":
                disliked_attrs = dislike_attrs_match.group(1).strip()
                attrs = [attr.strip().lower() for attr in re.split(r',|\n', disliked_attrs) if attr.strip()]
                all_attributes.extend(attrs)
                all_features.update(attrs)

        formatted_parts = []
        if all_movies:
            formatted_parts.append(f"movies: {', '.join(all_movies)}")
        if all_attributes:
            formatted_parts.append(f"attributes: {', '.join(all_attributes)}")
        
        formatted_text = "\n".join(formatted_parts)
        
        return list(all_features), formatted_text
    
    async def _process_single_data_batch(self, data, n_samples=5):
        initiator_worker_id = data["initiatorWorkerId"]
        conversation_id = data["conversationId"]
        messages = data["messages"]
        target_movie = data["target_movie"].strip()
        target_movie = re.sub(r'\s+', ' ', target_movie)
        id = data["id"]
        all_results = []

        try:
            matched_movies_path = self.eval_config.matched_movies_path
            with open(matched_movies_path, 'r', encoding='utf-8') as f:
                matched_movies = json.load(f)
        except Exception as e:
            predict_logger.error(f"Error loading movies database: {str(e)}")
            matched_movies = {}

        conversation_groups = defaultdict(list)
        for msg in messages:
            conv_id = msg.get("conversationId")
            conversation_groups[conv_id].append(msg)

        sorted_conv_ids_int = sorted(int(conv_id) for conv_id in conversation_groups.keys())
        
        dialog_blocks = []
        for i, conv_id in enumerate(sorted_conv_ids_int, 1):
            conv_id_str = conv_id
            messages_in_group = conversation_groups[conv_id_str]

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

        try:
            responses = await self.llm_client._generate_response_sample(
                prompt,
                temperature=0.2 if self.mode == "pos" else 1.0, # 1.0 for neg
                n=n_samples
            )
            
            if not isinstance(responses, list):
                responses = [responses]

            for sample_idx, response in enumerate(responses):
                try:
                    if not self._validate_response_format(response, sample_idx):
                        continue

                    preferences_marker = "### User Preferences:"
                    preferences_position = response.find(preferences_marker)
                    user_preferences_text = response[preferences_position + len(preferences_marker):].strip()

                    generated_movies, generated_attributes, all_generated_features = self._extract_all_generated_features(user_preferences_text, matched_movies)

                    target_movie_info = matched_movies.get(target_movie, {})

                    target_features = []

                    target_features.append(target_movie)

                    if 'genre' in target_movie_info and isinstance(target_movie_info['genre'], list):
                        target_features.extend([genre.lower() for genre in target_movie_info['genre']])
                    if 'year' in target_movie_info and target_movie_info['year']:
                        target_features.append(target_movie_info['year'])
                    if 'director' in target_movie_info and isinstance(target_movie_info['director'], list):
                        target_features.extend([director.lower() for director in target_movie_info['director']])
                    if 'writer' in target_movie_info and isinstance(target_movie_info['writer'], list):
                        target_features.extend([writer.lower() for writer in target_movie_info['writer']])
                    if 'star' in target_movie_info and isinstance(target_movie_info['star'], list):
                        target_features.extend([star.lower() for star in target_movie_info['star']])

                    matching_features = [f for f in target_features if f.lower() in [af.lower() for af in all_generated_features]]

                    generated_features_count = len(generated_movies) + len(generated_attributes)
                    if generated_features_count > 0:
                        score = len(matching_features) / generated_features_count
                    else:
                        score = 0.0

                    predict_logger.debug(
                        f"Original Prompt:\n{prompt}\n\n"
                        f"LLM Response:\n{response}\n\n"
                        f"Target Movie: {target_movie}\n"
                        f"Target Features: {', '.join(target_features)}\n"
                        f"Matching Features: {', '.join(matching_features)} ({len(matching_features)}/{len(target_features)})\n"
                        f"Generated Features Count: {generated_features_count}\n"
                        f"Score (matching/generated): {score:.4f}\n"
                        f"Sample Index: {sample_idx+1}/{len(responses)}\n"
                        f"Conversation ID: {conversation_id}\n"
                        f"ID: {id}\n"
                    )

                    sample_json = {
                        "initiatorWorkerId": initiator_worker_id, 
                        "conversationId": conversation_id, 
                        "target_movie": target_movie,
                        "id": id, 
                        "batch_id": f"{id}_{sample_idx}",
                        "cot_result": response,
                        "score": score,
                        "matching_features_count": len(matching_features),
                        "generated_features_count": generated_features_count,
                        "target_features_count": len(target_features),
                        "generated_movies_count": len(generated_movies),
                        "generated_attributes_count": len(generated_attributes)
                    }

                    all_results.append(sample_json)
                    
                except Exception as e:
                    predict_logger.error(f"An error occurred with sample {sample_idx+1}: {str(e)}")

            if all_results:
                combined_output_file = self.eval_config.output_file
                with open(combined_output_file, "a") as f:
                    for result in all_results:
                        f.write(json.dumps(result) + "\n")
                
                predict_logger.info(f"{len(all_results)} / {len(responses)} samples with ID={id} have been saved")
            else:
                predict_logger.info(f"ID={id} generated no valid samples")
        
        except Exception as e:
            predict_logger.error(f"An error occurred when generating samples: {str(e)}")
        
        return all_results