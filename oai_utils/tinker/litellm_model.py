from litellm import AsyncHTTPHandler
import httpx
from litellm import ModelConfig
from typing import Self
import logging
import uuid
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Type,
    TypeGuard,
    TypeVar,
)

import litellm
import tinker
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    ChatCompletionTokenLogprob,
)
from litellm.types.utils import ChoiceLogprobs as LitellmChoiceLogprobs
from litellm.types.utils import Choices
from litellm.types.utils import Message as LitellmMessage
from litellm.types.utils import ModelResponse
from litellm.types.utils import TopLogprob as LitellmTopLogprob
from litellm.utils import custom_llm_setup
from pydantic import TypeAdapter
from tinker.types import ModelInput, SampleResponse, SamplingParams
from tinker_cookbook.renderers import Message as TinkerMessage
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers import ToolCall as TinkerToolCall
from tinker_cookbook.renderers.base import ToolSpec
from transformers import PreTrainedTokenizer

from oai_utils.runresult import RunResultWrapper
from tinker_cookbook.rl.types import Trajectory, Transition
from agents.items import MessageOutputItem
from oai_utils.tinker.model_with_logprob import (
    LogprobResponseOutputMessage,
    LogprobResponseFunctionToolCall,
    TinkerModelResponse,
)
import tinker_cookbook.completers


logger = logging.getLogger(__name__)

T = TypeVar("T")


def generate_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix.

    Args:
        prefix: String prefix for the generated ID.

    Returns:
        A unique identifier string.
    """
    return prefix + str(uuid.uuid4())


class TinkerLLM(CustomLLM):
    """LiteLLM provider that proxies Tinker's sampling client.

    The cookbook exposes fine-tuned models through `TinkerTokenCompleter` (a
    lightweight callable). Agent-lightning needs a persistent LiteLLM endpoint,
    so that agent developers can still reuse the same agent code without changes.

    This class rewraps the sampling client to satisfy LiteLLM's `CustomLLM`
    protocol while keeping Tinker's renderer/tokenizer pipeline intact.

    Attributes:
        model_name: The HuggingFace model identifier.
        renderer: Prompt renderer for formatting messages.
        tokenizer: Tokenizer for the model.
        sampling_client: Tinker sampling client for generation.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling parameter.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        model_name: str,
        renderer: Renderer,
        tokenizer: PreTrainedTokenizer,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        seed: int = 42,
    ) -> None:
        """Initialize the TinkerLLM."""
        self.model_name = model_name
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed

    def update_sampling_client(self, sampling_client: tinker.SamplingClient) -> None:
        """Update the sampling client used for generation.

        Args:
            sampling_client: New Tinker sampling client to use.
        """
        self.sampling_client = sampling_client

    def _canonicalize_messages(self, messages: Any) -> List[TinkerMessage]:
        return TypeAdapter(
            List[TinkerMessage], config={"arbitrary_types_allowed": True}
        ).validate_python(messages)
        # Exception will be raised if validation fails

    def _validate_role(
        self, role: str
    ) -> TypeGuard[Literal["assistant", "user", "system", "tool", "function"]]:
        if role not in ["assistant", "user", "system", "tool", "function"]:
            raise ValueError(f"Invalid role: {role}")
        return True

    def _parse_tool_call(
        self, tool_call: TinkerToolCall
    ) -> ChatCompletionMessageToolCall:
        return ChatCompletionMessageToolCall(
            id=tool_call.id or generate_id("tinker-tool-call-"),
            function={
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
            type="function",
        )

    def _get_optional_params(
        self,
        optional_params: Dict[str, Any],
        keys: List[str],
        expected_type: Type[T],
        validate_fn: Callable[[T], bool],
        default_value: T,
    ) -> T:
        for key in keys:
            if key in optional_params:
                value = optional_params[key]
                if not isinstance(value, expected_type):
                    raise ValueError(f"Invalid {key} type: {type(value)}")
                if not validate_fn(value):
                    raise ValueError(f"Invalid {key}. Did not pass validation: {value}")
                return value
        return default_value

    def _prepare_model_input(
        self,
        messages: list,
        tools: list | None,
    ) -> ModelInput:
        """LiteLLM messages -> Tinker ModelInput."""

        final_messages = []
        if messages:
            final_messages = list(messages)

        # Handle Tools if present
        if tools:
            # 1. Extract system instruction
            system_instructions = ""
            system_msg_index = -1
            for i, msg in enumerate(final_messages):
                if msg.get("role") == "system":
                    system_instructions = msg.get("content", "")
                    system_msg_index = i
                    break

            # 2. Convert to ToolSpecs
            tool_specs: List[ToolSpec] = []
            for t in tools:
                if t.get("type") == "function":
                    f = t["function"]
                    tool_specs.append(
                        {
                            "name": f["name"],
                            "description": f.get("description", ""),
                            "parameters": f.get("parameters", {}),
                        }
                    )

            # 3. Create prefix messages
            try:
                prefix_messages = self.renderer.create_conversation_prefix_with_tools(
                    tool_specs, system_instructions
                )

                # Replace system message with prefix messages
                new_messages = []
                # Add prefix
                # We need to cast prefix_messages (TinkerMessage) to dicts if needed,
                # but valid dictionaries should pass canonicalization.
                new_messages.extend(prefix_messages)

                # Add rest of messages, skipping original system message
                for i, msg in enumerate(final_messages):
                    if i == system_msg_index:
                        continue
                    new_messages.append(msg)

                final_messages = new_messages

            except NotImplementedError:
                # Renderer doesn't support tools, ignore.
                logger.warning(
                    f"Renderer {type(self.renderer)} does not support 'create_conversation_prefix_with_tools'. Tools will be ignored."
                )
                pass

        canonical_messages = self._canonicalize_messages(final_messages)
        return self.renderer.build_generation_prompt(canonical_messages)

    def _parse_response(
        self, model_input: ModelInput, response: SampleResponse
    ) -> ModelResponse:
        """Tinker Response -> LiteLLM Response.

        Extract log probabilities as well.
        """
        choices: List[Choices] = []
        for seq in response.sequences:
            if seq.logprobs is not None:
                token_strings: List[str] = self.tokenizer.batch_decode(
                    [token]  # type: ignore
                    for token in seq.tokens
                )
                bytes_list: List[List[int]] = [
                    list(token.encode("utf-8")) for token in token_strings
                ]
                logprobs = LitellmChoiceLogprobs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token=token,
                            bytes=bytes,
                            logprob=logprob,
                            # Note: This top logprob is not accurate but satisfies validation
                            top_logprobs=[
                                LitellmTopLogprob(
                                    token=token, bytes=bytes, logprob=logprob
                                )
                            ],
                        )
                        for token, bytes, logprob in zip(
                            token_strings, bytes_list, seq.logprobs
                        )
                    ]
                )
            else:
                logprobs = None

            parsed_response, parse_success = self.renderer.parse_response(seq.tokens)
            if parse_success:
                role = parsed_response["role"]
                if not self._validate_role(role):
                    assert False, "This should never happen"

                content = parsed_response["content"]
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if part["type"] == "text":
                            text_parts.append(part["text"])
                        elif part["type"] == "thinking":
                            text_parts.append(f"<think>{part['thinking']}</think>")
                        elif part["type"] == "tool_call":
                            continue
                        else:
                            raise ValueError(
                                f"Unexpected content part type: {part['type']}"
                            )
                    content = "".join(text_parts)

                if content is not None and not isinstance(content, str):
                    raise ValueError(
                        f"Content must be str or None, got {type(content)}"
                    )

                # Legacy content check
                if not content:
                    logger.warning(
                        "Parsed content is empty. Original response: " + str(response)
                    )
                tool_calls = parsed_response.get("tool_calls", None)
                if tool_calls:
                    tool_calls = [
                        self._parse_tool_call(tool_call) for tool_call in tool_calls
                    ]
                choices.append(
                    Choices(
                        message=LitellmMessage(
                            role=role, content=content, tool_calls=tool_calls
                        ),
                        finish_reason=seq.stop_reason,
                        logprobs=logprobs,
                        token_ids=seq.tokens,
                    )
                )
            else:
                logger.warning(
                    "Failed to parse response, likely due to truncated response, submiting partial response."
                )
                # Go with the default path
                content = parsed_response["content"]
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if part["type"] == "text":
                            text_parts.append(part["text"])
                        elif part["type"] == "thinking":
                            text_parts.append(f"<think>{part['thinking']}</think>")
                        elif part["type"] == "tool_call":
                            continue
                        else:
                            raise ValueError(
                                f"Unexpected content part type: {part['type']}"
                            )
                    content = "".join(text_parts)

                if content is not None and not isinstance(content, str):
                    raise ValueError(
                        f"Content must be str or None, got {type(content)}"
                    )
                choices.append(
                    Choices(
                        message=LitellmMessage(role="assistant", content=content),
                        finish_reason=seq.stop_reason,
                        logprobs=logprobs,
                        token_ids=seq.tokens,
                    )
                )
        return TinkerModelResponse(
            id=generate_id("tinker-sampling-"),
            choices=choices,
            prompt_token_ids=model_input.to_ints(),
            tinker_model_input=model_input,
            created=int(time.time()),
            model=self.model_name,
            object="chat.completion",
        )

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: float | httpx.Timeout | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> ModelResponse:
        """Main entrypoint for LiteLLM to call."""
        tools = optional_params.get("tools", None)
        sampling_client = optional_params.get("sampling_client")
        if sampling_client is None:
            raise ValueError(
                "Sampling client in optional_params is required for TinkerLLM."
            )
        max_tokens = self._get_optional_params(
            optional_params,
            ["max_completion_tokens", "max_tokens"],
            int | None,
            lambda x: x >= 0 if x is not None else True,
            None,
        )
        temperature = self._get_optional_params(
            optional_params,
            ["temperature"],
            float,
            lambda x: 0.0 <= x <= 2.0,
            self.temperature,
        )
        top_k = self._get_optional_params(
            optional_params, ["top_k"], int, lambda x: True, self.top_k
        )
        top_p = self._get_optional_params(
            optional_params, ["top_p"], float, lambda x: 0.0 <= x <= 1.0, self.top_p
        )
        seed = self._get_optional_params(
            optional_params, ["seed"], int, lambda _: True, self.seed
        )

        model_input = self._prepare_model_input(messages=messages, tools=tools)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            stop=self.renderer.get_stop_sequences(),
        )
        result = await sampling_client.sample_async(
            prompt=model_input, sampling_params=params, num_samples=1
        )
        final_response = self._parse_response(model_input, result)
        return final_response

    def as_model_list(self) -> List[ModelConfig]:
        """Generate model configuration for LiteLLM proxy.

        Returns:
            List containing model configuration dict for LiteLLM.
        """
        return [
            ModelConfig(
                model_name=self.model_name,
                litellm_params={
                    "model": f"agl-tinker/{self.model_name}",
                },
                tpm=1_000_000,
                rpm=1_000,
            ),
        ]

    def rewrite_litellm_custom_providers(self) -> Self:
        """Register this TinkerLLM as a custom provider in LiteLLM.

        !!! warning
            This method modifies the global LiteLLM state, which could interfere with other tests in the
            same process.

        Returns:
            Self for method chaining.
        """
        litellm.custom_provider_map = [
            {"provider": "agl-tinker", "custom_handler": self}
        ]
        custom_llm_setup()
        return self


def result_to_trajectory(result_wrapper: RunResultWrapper[Any]) -> Trajectory:
    """Convert a RunResultWrapper to a Trajectory."""
    transitions: list[Transition] = []

    # We iterate over new_items to find actions taken by the agent
    # and reconstruct the sequence of transitions from LogprobResponseOutputMessage.
    # Each LogprobResponseOutputMessage contains:
    # - model_input (Observation BEFORE this action)
    # - tokens_with_logprobs (The Action taken)

    last_model_input: tinker.ModelInput | None = None
    last_action: tinker_cookbook.completers.TokensWithLogprobs | None = None

    for item in result_wrapper.result.new_items:
        if isinstance(item, MessageOutputItem):
            # Inspect raw_item
            if hasattr(item, "raw_item") and isinstance(
                item.raw_item, LogprobResponseOutputMessage
            ):
                traj_data = item.raw_item.tinker_trajectory_data
                if traj_data:
                    # Found a transition data point
                    obs = traj_data.model_input
                    action = traj_data.tokens_with_logprobs

                    transition = Transition(
                        ob=obs,
                        ac=action,
                        reward=0.0,
                        episode_done=False,
                    )
                    transitions.append(transition)

                    last_model_input = obs
                    last_action = action
        elif isinstance(item, LogprobResponseFunctionToolCall):
            traj_data = item.tinker_trajectory_data
            if traj_data:
                obs = traj_data.model_input
                action = traj_data.tokens_with_logprobs

                transition = Transition(
                    ob=obs,
                    ac=action,
                    reward=0.0,
                    episode_done=False,
                )
                transitions.append(transition)

                last_model_input = obs
                last_action = action

    if not transitions:
        # Fallback or empty?
        # If no transitions found (maybe no tool calls or no model response with logs), return empty?
        # But Trajectory requires final_ob.
        raise ValueError("No trajectory data found in result wrapper.")

    # Mark last transition as done
    transitions[-1].episode_done = True

    # Construct final_ob
    # We need to append the last action to the last observation to get the state AFTER the last action.
    # tinker.ModelInput.append(...) takes a Chunk.
    # We need to convert action (tokens) to a Chunk.
    # EncodedTextChunk is likely what we want.

    assert last_model_input is not None
    assert last_action is not None

    # We append the action to get final state.
    # Note: validation of append might require knowing if it's text or image, but tokens suggest text.
    final_ob = last_model_input.append(
        tinker.types.EncodedTextChunk(tokens=last_action.tokens)
    )

    return Trajectory(
        transitions=transitions,
        final_ob=final_ob,
    )
