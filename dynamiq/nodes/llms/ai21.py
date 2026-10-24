from pydantic import field_validator

from dynamiq.connections import AI21 as AI21Connection
from dynamiq.nodes.llms.base import BaseLLM


class AI21(BaseLLM):
    """AI21 LLM node.

    This class provides an implementation for the AI21 Language Model node.

    Attributes:
        connection (AI21Connection): The connection to use for the AI21 LLM.
        MODEL_PREFIX (str): The prefix for the AI21 model name.
    """
    connection: AI21Connection
    MODEL_PREFIX = "ai21_chat/"

    @field_validator("model", mode="before")
    @classmethod
    def _route_legacy_prefix(cls, value: str | None) -> str | None:
        """Move a model saved with the "ai21/" prefix to "ai21_chat/".

        litellm sends "ai21/" models to OpenAI's API, with the AI21 key as the bearer token.
        """
        if isinstance(value, str) and value.startswith("ai21/"):
            return cls.MODEL_PREFIX + value.removeprefix("ai21/")
        return value

    def __init__(self, **kwargs):
        """Initialize the AI21 LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AI21Connection()
        super().__init__(**kwargs)
