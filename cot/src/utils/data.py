import re
import json
import hashlib

from typing import List, Dict
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def robust_json_loads(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON due to {e}: {text}")
        _append_json_failed(text)
        return None


def _append_json_failed(text):
    with open("log/failed_responses.txt", "a") as f:
        f.write(text + "\n" + "#" * 80 + "\n")


def parse_json(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    json_code_block = re.findall(r"```json(.*?)```", response_text, re.DOTALL)
    ans = {}
    for code_block in json_code_block:
        if robust_json_loads(code_block):
            ans.update(json.loads(code_block))
        else:
            code_block = re.sub(r"//.*$", "", code_block, flags=re.MULTILINE)
            code_block = re.sub(r"'(\w+)':", r'"\1":', code_block)
            code_block = re.sub(r",\s*([\]}])", r"\1", code_block)
        code_block = robust_json_loads(code_block)
        if code_block:
            ans.update(code_block)
    if ans:
        return ans
    return None


def parse_list(response_text: str) -> list:
    list_items = re.findall(r"\[.*?\]", response_text)
    ans = []
    for item in list_items:
        if "movie1" in item or "moviename1" in item or "movie_1" in item:
            continue
        item = item.strip("[]")
        item = item.replace("\\'", "'")
        item = re.sub(r"^\s*\d+\.\s*", "", item)
        ans.extend([s.strip(" \"'") for s in item.split(",")])
    return ans


def messages2str(messages: list) -> str:
    return "\n".join(
        f"{msg['role']}: {msg['text']}"
        for msg in (messages + [{"role": "recommender", "text": ""}])
    )


def dict2md(d: dict) -> str:
    return "\n".join(f"{i+1}. **{k}**: {d[k]}" for i, k in enumerate(d))


def list2md(ls: list) -> str:
    return "\n".join(f"{i+1}. **{k}**" for i, k in enumerate(ls))


def hash_string(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def parse_json_from_response(response_text: str) -> dict:
    pattern = r'\{.*\}'
    match = re.search(pattern, response_text, re.DOTALL)

    if match:
        json_str = match.group(0)
        try:
            json_data = json.loads(json_str)
            required_keys = ["text", "senderWorkerId", "conversationId", "timeOffSet", "mapping", "sentiment"]
            missing_keys = [key for key in required_keys if key not in json_data]
            if missing_keys:
                logger.error(
                    f"Parsed JSON is missing required keys: {missing_keys}\nResponse: {response_text}"
                )
                return None
            return json_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}\nResponse: {response_text}")
    else:
        logger.error(f"No valid JSON found in response: {response_text}")

    return None

def parse_md_dict(response_text: str) -> Dict[str, str]:
    pattern = r"(\d+)\. \*\*(.*?)\*\*[:：] (.*)"
    matches = re.findall(pattern, response_text, re.MULTILINE)

    res = {}
    for match in matches:
        entity = match[1]
        attitude = match[2]
        res[entity] = attitude
    return res


def parse_md_list(response_text: str) -> List[str]:
    pattern = r"(\d+)\. \*\*(.*?)\*\*"
    matches = re.findall(pattern, response_text, re.MULTILINE)

    res = []
    for match in matches:
        entity = match[1]
        res.append(entity)
    return res


def parse_indented_list(text) -> Dict[str, List[str]]:
    pattern = r"^- (.*?)(?=^- |\Z)"
    sub_pattern = r"^\s+- (.*?)$"
    result = {}

    for block in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
        block_text = block.group(0)

        main_item = re.match(r"- (.*?)$", block_text, re.MULTILINE).group(1)

        sub_items = re.finditer(sub_pattern, block_text, re.MULTILINE)
        result[main_item] = [item.group(1) for item in sub_items]

    return result


def parse_merged_list(text) -> Dict[str, Dict[str, str]]:
    pattern = r"^- [Ee]ntity:\s?(.*?)(?=^- |\Z)"
    sub_pattern = r"^\s+- (.*?)$"

    result = {}

    for block in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
        block_text = block.group(0)

        main_item = re.match(r"- [Ee]ntity: (.*?)$", block_text, re.MULTILINE).group(1)

        sub_items = re.finditer(sub_pattern, block_text, re.MULTILINE)
        sub_items = [item.group(1) for item in sub_items]
        result[main_item] = {}
        for t in sub_items:
            colon_index = t.find(":")
            entity = t[:colon_index].strip()
            attitude = t[colon_index + 1 :].strip()
            result[main_item][entity] = attitude

    return result


def parse_preferences(text: str) -> Dict[str, List[str]]:
    preferences = {}

    lines = text.strip().split("\n")

    current_entity = None
    current_descriptions = []

    for line in lines:
        line = re.sub(r"^\d+\.\s*", "", line.strip())

        entity_match = re.match(r"\*\*(.*?)\*\*:\s*(.*)", line)
        if entity_match:
            if current_entity:
                preferences[current_entity] = current_descriptions

            current_entity = entity_match.group(1)
            description = entity_match.group(2)
            current_descriptions = []

            if description:
                current_descriptions.append(description.strip())

        # Check if line is part of a bullet point list for current entity
        elif line.strip().startswith("-"):
            description = line.strip("- ").strip()
            if description:
                current_descriptions.append(description)

    # Add final entity
    if current_entity and current_descriptions:
        preferences[current_entity] = current_descriptions

    return preferences
