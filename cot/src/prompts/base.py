from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BasePromptManager(ABC):
    """Abstract base class for prompt managers."""

    @abstractmethod
    def get_prompt(self, prompt_type: str) -> str:
        """Get prompt template for given type."""
        pass

    # @abstractmethod
    # def format_prompt(self, prompt_type: str, **kwargs) -> str:
    #     """Format prompt template with given arguments."""
    #     pass

    @abstractmethod
    def messages2str(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to string format."""
        pass

    @abstractmethod
    def construct_prompt(
        self,
        entity_attitude_dict: Optional[Dict[str, str]],
        collaborative_item_list: Optional[Dict[str, str]],
        reasoning_guidelines: Optional[List[str]],
        dialog_str: str,
    ) -> str:
        """Construct complete prompt from components."""
        pass
