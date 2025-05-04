from dataclasses import dataclass


@dataclass
class ConfigBase:
    def __repr__(self) -> str:
        max_key_length = max(len(k) for k in self.__dict__)
        return "\n".join(
            [f" {self.__class__.__name__} "]
            + [f"{k:<{max_key_length}} : {v}" for k, v in self.__dict__.items()]
        )

    def format(self) -> str:
        max_key_length = max(len(k) for k in self.__dict__)
        return "\n".join(
            f"{k:<{max_key_length}} : {v}" for k, v in self.__dict__.items()
        )

    def v_format(self) -> str:
        max_key_length = max(len(k) for k in self.__dict__)
        lines = [f"{k:<{max_key_length}} : {v}" for k, v in self.__dict__.items()]
        title = f" {self.__class__.__name__} "
        width = max(len(line) for line in lines)
        width = max(width, len(title))

        border = "─" * width
        return (
            f"╭{border}╮\n"
            f"│{title:^{width}}│\n"
            f"├{border}┤\n"
            + "\n".join(f"│ {line:<{width}} │" for line in lines)
            + f"\n╰{border}╯"
        )

    def update(self, **kwargs) -> "ConfigBase":
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")
        return self
