import json
import types
from typing import cast

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import (
    ImageProcessorProtocol,
    Message,
    RenderContext,
    RenderedMessage,
    TextPart,
    _tool_call_payload,
    image_to_chunk,
    remove_thinking,
)
from tinker_cookbook.renderers.qwen3 import (
    Qwen3Renderer,
    Qwen3VLRenderer,
    _merge_consecutive_text_parts,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

from oai_utils.tinker.litellm_model import TinkerLLM
from oai_utils.tinker.model_with_logprob import LogprobLitellmModel


def render_message_qwen3vl(
    self, message: Message, ctx: RenderContext
) -> RenderedMessage:
    maybe_newline = "\n" if ctx.idx > 0 else ""

    role = self._get_qwen_role_for_message(message)
    header_str = f"{maybe_newline}<|im_start|>{role}\n"

    # Strip thinking from history for non-last assistant messages (matching non-VL behavior)
    strip_thinking = (
        self.strip_thinking_from_history
        and message["role"] == "assistant"
        and not ctx.is_last
    )
    output_chunks = self._preprocess_message_parts(
        message, strip_thinking=strip_thinking
    )

    # Handle tool response wrapping
    if message["role"] == "tool":
        output_chunks = self._wrap_qwen_tool_response_chunks(output_chunks)

    if "tool_calls" in message:
        # we removed additional new line from tinker impl.
        output_chunks += [
            TextPart(
                type="text",
                text="\n".join(
                    [
                        f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                        for tool_call in message["tool_calls"]
                    ]
                ),
            )
        ]
    output_chunks += [TextPart(type="text", text="<|im_end|>")]

    if self.merge_text_chunks:
        output_chunks = _merge_consecutive_text_parts(output_chunks)

    output_chunks_encoded: list[tinker.ModelInputChunk] = [
        image_to_chunk(
            image_or_str=x["image"],
            image_processor=cast(ImageProcessorProtocol, self.image_processor),
        )
        if x["type"] == "image"
        else tinker.EncodedTextChunk(
            tokens=self.tokenizer.encode(x["text"], add_special_tokens=False)
        )
        for x in output_chunks
    ]

    header = tinker.types.EncodedTextChunk(
        tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
    )
    return RenderedMessage(header=header, output=output_chunks_encoded)


def render_message_qwen3(self, message: Message, ctx: RenderContext) -> RenderedMessage:
    maybe_newline = "\n" if ctx.idx > 0 else ""

    role = self._get_qwen_role_for_message(message)
    header_str = f"{maybe_newline}<|im_start|>{role}\n"

    content = message["content"]

    if isinstance(content, list):
        # Structured content - handle with list operations
        parts = content
        if (
            self.strip_thinking_from_history
            and message["role"] == "assistant"
            and not ctx.is_last
        ):
            # Remove thinking parts for historical messages
            parts = remove_thinking(parts)
        # Render parts in order, preserving interleaved thinking/text structure.
        # No separator needed - whitespace is preserved in TextPart for roundtrip identity.
        rendered_parts = []
        for p in parts:
            if p["type"] == "thinking":
                rendered_parts.append(f"<think>{p['thinking']}</think>")
            elif p["type"] == "text":
                rendered_parts.append(p["text"])
        output_content = "".join(rendered_parts)
    else:
        # String content - pass through as-is.
        # Note: strip_thinking_from_history only works with list-based content.
        # For stripping to work on historical messages, use structured content
        # with ThinkingPart separated from text (as returned by parse_response).
        output_content = content

    # Handle tool response wrapping
    if message["role"] == "tool":
        output_content = self._wrap_qwen_tool_response(output_content)

    # Handle tool_calls field
    if "tool_calls" in message:
        # Add leading newline to match HF template behavior
        # we removed additional new line from tinker impl.
        output_content += "\n".join(
            [
                f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                for tool_call in message["tool_calls"]
            ]
        )
    output_content += "<|im_end|>"
    header = tinker.types.EncodedTextChunk(
        tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
    )
    output: list[tinker.ModelInputChunk] = [
        tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(output_content, add_special_tokens=False)
        )
    ]
    return RenderedMessage(header=header, output=output)


def setup_tinkermodel(
    service_client: tinker.ServiceClient, model_name: str, path: str | None = None
) -> tuple[LogprobLitellmModel, Tokenizer, Renderer]:
    sampling_client = service_client.create_sampling_client(
        base_model=model_name, model_path=path
    )
    tokenizer = sampling_client.get_tokenizer()
    image_processor = get_image_processor(model_name)

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer, image_processor)
    # Monkey patch renderer
    if isinstance(renderer, Qwen3Renderer):
        renderer.render_message = types.MethodType(render_message_qwen3, renderer)
    elif isinstance(renderer, Qwen3VLRenderer):
        renderer.render_message = types.MethodType(render_message_qwen3vl, renderer)
    tinker_llm = TinkerLLM(
        model_name=model_name, renderer=renderer, tokenizer=tokenizer
    )
    tinker_llm.rewrite_litellm_custom_providers()
    litellm_model_name = f"tinker/{model_name}"
    model = LogprobLitellmModel(
        model=litellm_model_name, sampling_client=sampling_client
    )

    return model, tokenizer, renderer
