from typing import Dict, List


class PromptFormatter:
    """Utility class for formatting different types of data into prompts."""

    @staticmethod
    def dict2md(d: Dict[str, str]) -> str:
        """Convert dictionary to markdown format."""
        return "\n".join([f"**{k}**: {v}" for k, v in d.items()])

    @staticmethod
    def list2md_bold(ls: List[str], prefix: str = "") -> str:
        """Convert list to markdown with bold items."""
        return "\n".join([f"{prefix}{i+1}. **{item}**" for i, item in enumerate(ls)])

    @staticmethod
    def list2str(ls: List[str]) -> str:
        """Convert list to comma-separated string."""
        return ", ".join(ls) if isinstance(ls, list) else ls

    @staticmethod
    def collaborate_md(collaborate: Dict) -> str:
        """Format collaborative items to markdown."""
        res = []
        for idx, (k, v) in enumerate(collaborate.items()):
            format_str = f"{idx+1}. **{k}**"
            if isinstance(v, list):
                format_str += ": " + PromptFormatter.list2str(v)
            elif isinstance(v, dict):
                info_str = "\n".join(
                    [f"  -{k}: {PromptFormatter.list2str(v)}" for k, v in v.items()]
                )
                format_str += "\n" + info_str
            res.append(format_str)
        return "\n".join(res)
