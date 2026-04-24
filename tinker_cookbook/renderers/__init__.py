"""
Renderers for converting message lists into training and sampling prompts.

Use viz_sft_dataset to visualize the output of different renderers. E.g.,
    python -m tinker_cookbook.supervised.viz_sft_dataset dataset_path=Tulu3Builder renderer_name=role_colon
"""

from collections.abc import Callable
from typing import Any

from tinker_cookbook.image_processing_utils import ImageProcessor

# Types and utilities used by external code
from tinker_cookbook.renderers.base import (
    # Content part types
    ContentPart,
    ImagePart,
    Message,
    # Streaming types
    MessageDelta,
    # Renderer base
    RenderContext,
    Renderer,
    Role,
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    Utf8TokenDecoder,
    # Utility functions
    ensure_text,
    format_content_as_string,
    get_text_content,
    parse_content_blocks,
)

# Renderer classes used directly by tests
from tinker_cookbook.renderers.deepseek_v3 import DeepSeekV3ThinkingRenderer
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer

# Global registry for custom renderer factories
_CUSTOM_RENDERER_REGISTRY: dict[str, Callable[[Tokenizer, Any], Renderer]] = {}


def register_renderer(
    name: str,
    factory: Callable[[Tokenizer, Any], Renderer],
) -> None:
    """Register a custom renderer factory."""
    _CUSTOM_RENDERER_REGISTRY[name] = factory


def get_registered_renderer_names() -> list[str]:
    """Return a list of all registered custom renderer names."""
    return list(_CUSTOM_RENDERER_REGISTRY.keys())


def is_renderer_registered(name: str) -> bool:
    """Check if a renderer name is registered."""
    return name in _CUSTOM_RENDERER_REGISTRY


def unregister_renderer(name: str) -> bool:
    """Unregister a custom renderer factory."""
    if name in _CUSTOM_RENDERER_REGISTRY:
        del _CUSTOM_RENDERER_REGISTRY[name]
        return True
    return False


def get_renderer(
    name: str, tokenizer: Tokenizer, image_processor: ImageProcessor | None = None
) -> Renderer:
    """Factory function to create renderers by name."""
    if (renderer := _CUSTOM_RENDERER_REGISTRY.get(name)) is not None:
        return renderer(tokenizer, image_processor)

    from tinker_cookbook.renderers.deepseek_v3 import DeepSeekV3DisableThinkingRenderer
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
    from tinker_cookbook.renderers.kimi_k2 import KimiK2Renderer
    from tinker_cookbook.renderers.kimi_k25 import KimiK25DisableThinkingRenderer, KimiK25Renderer
    from tinker_cookbook.renderers.llama3 import Llama3Renderer
    from tinker_cookbook.renderers.qwen3 import (
        Qwen3DisableThinkingRenderer,
        Qwen3InstructRenderer,
        Qwen3VLInstructRenderer,
        Qwen3VLRenderer,
    )
    from tinker_cookbook.renderers.role_colon import RoleColonRenderer

    if name == "role_colon":
        return RoleColonRenderer(tokenizer)
    elif name == "llama3":
        return Llama3Renderer(tokenizer)
    elif name == "qwen3":
        return Qwen3Renderer(tokenizer)
    elif name == "qwen3_vl":
        return Qwen3VLRenderer(tokenizer, image_processor)
    elif name == "qwen3_vl_instruct":
        return Qwen3VLInstructRenderer(tokenizer, image_processor)
    elif name == "qwen3_disable_thinking":
        return Qwen3DisableThinkingRenderer(tokenizer)
    elif name == "qwen3_instruct":
        return Qwen3InstructRenderer(tokenizer)
    elif name == "deepseekv3":
        return DeepSeekV3DisableThinkingRenderer(tokenizer)
    elif name == "deepseekv3_disable_thinking":
        return DeepSeekV3DisableThinkingRenderer(tokenizer)
    elif name == "deepseekv3_thinking":
        return DeepSeekV3ThinkingRenderer(tokenizer)
    elif name == "kimi_k2":
        return KimiK2Renderer(tokenizer)
    elif name == "kimi_k25":
        return KimiK25Renderer(tokenizer, image_processor=image_processor)
    elif name == "kimi_k25_disable_thinking":
        return KimiK25DisableThinkingRenderer(tokenizer, image_processor=image_processor)
    else:
        raise ValueError(f"Unknown renderer: {name}")


__all__ = [
    "ContentPart",
    "ImagePart",
    "Message",
    "Role",
    "TextPart",
    "ThinkingPart",
    "ToolCall",
    "ToolSpec",
    "MessageDelta",
    "StreamingMessageHeader",
    "StreamingTextDelta",
    "StreamingThinkingDelta",
    "Utf8TokenDecoder",
    "RenderContext",
    "Renderer",
    "TrainOnWhat",
    "ensure_text",
    "format_content_as_string",
    "get_text_content",
    "parse_content_blocks",
    "register_renderer",
    "unregister_renderer",
    "get_registered_renderer_names",
    "is_renderer_registered",
    "get_renderer",
    "DeepSeekV3ThinkingRenderer",
    "GptOssRenderer",
    "Qwen3Renderer",
]