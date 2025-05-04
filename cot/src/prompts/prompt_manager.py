from typing import Dict
from src.prompts.base import BasePromptManager

def import_rec_prompts():
    from src.prompts.prompt_repo import (
        CONVERSATION_TEMPLATE,
        RECOMMEND_INSTRUCTION_QWEN_MATCH_FINAL,
    )

    return {
        "conversation": CONVERSATION_TEMPLATE,
        "recommend": RECOMMEND_INSTRUCTION_QWEN_MATCH_FINAL,
    }

def import_reason_rec_prompts():
    from src.prompts.prompt_repo import (
        CONVERSATION_TEMPLATE,
        RECOMMEND_INSTRUCTION_QWEN3_MATCH_FINAL,
    )

    return {
        "conversation": CONVERSATION_TEMPLATE,
        "recommend": RECOMMEND_INSTRUCTION_QWEN3_MATCH_FINAL,
    }

def import_score_prompts():
    """Import MemoCRS prompt templates."""
    from src.prompts.prompt_repo import (
        CONVERSATION_TEMPLATE_SCORE,
        RECOMMEND_INSTRUCTION_SCORE_FINAL,
    )

    return {
        "conversation": CONVERSATION_TEMPLATE_SCORE,
        "recommend": RECOMMEND_INSTRUCTION_SCORE_FINAL,
    }


def list2str(ls):
    return ", ".join(ls) if isinstance(ls, list) else ls


def drop_none(x):
    return [i for i in x if i]


def dict2md(d):
    return "\n".join([f"- **{k}**: {v}" for k, v in d.items()])


def list2md_bold(ls, prefix=""):
    return "\n".join([f"{prefix}{i+1}. **{item}**" for i, item in enumerate(ls)])


def list2md_normal(ls, prefix=""):
    return "\n".join([f"{prefix}{i+1}. {item}" for i, item in enumerate(ls)])


def list2md_disorder(ls, prefix=""):
    return "\n".join([f"{prefix}- {item}" for i, item in enumerate(ls)])


from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="memory.log")


class PromptManager(BasePromptManager):
    """Manage prompts for different tasks"""

    def __init__(self, method: str = "rec"):
        self.method = method
        if method.lower() == "rec":
            self.prompts = import_rec_prompts()
        elif method.lower() == "reason_rec":
            self.prompts = import_reason_rec_prompts()
        elif method.lower() == "score":
            self.prompts = import_score_prompts()

    def get_prompt(self, prompt_type: str) -> str:
        """Get prompt template based on language and prompt type"""
        if prompt_type not in self.prompts:
            logger.warning(
                f"Prompt type {prompt_type} not found in {self.method} templates"
            )
            return ""
        return self.prompts[prompt_type]

    def messages2str(self, messages):
        role_mapping = {
            "user": "User",
            "recommender": "Recommender",
        }
        return "\n".join([f"{role_mapping[m['role']]}: {m['text']}" for m in messages])

    def construct_prompt(
        self,
        dialog_str,
    ):  
        conditions = ", ".join(
            drop_none(
                [
                    "user's conversation"
                ]
            )
        )

        if conditions:
            conditions = " and ".join(conditions.rsplit(", ", 1))

        
        conversation_str = self.get_prompt("conversation").format(
            dialog_str=dialog_str
        )

        prompt = "\n\n".join(
            drop_none(
                [
                    self.get_prompt("recommend"),
                    conversation_str,
                ]
            )
        )

        return prompt
