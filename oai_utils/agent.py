import asyncio
import logging
from asyncio import timeout
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Self

import openai
from agents import (
    Agent,
    AgentsException,
    ItemHelpers,
    MaxTurnsExceeded,
    Model,
    ModelBehaviorError,
    ModelSettings,
    Runner,
    StopAtTools,
    Tool,
    ToolsToFinalOutputFunction,
    TResponseInputItem,
    UserError,
)
from agents.mcp.server import MCPServer
from agents.run import DEFAULT_MAX_TURNS
from litellm import ContextWindowExceededError
from openai._exceptions import BadRequestError
from pydantic import BaseModel
from agents.models.default_models import get_default_model_settings
from oai_utils.runresult import RunResultWrapper

logger = logging.getLogger(__name__)
type AgentsSDKModel = str | Model


class AgentRunFailure(Exception):
    def __init__(
        self,
        message: str,
        cause: Literal[
            "ModelBehaviourError",
            "Timeout",
            "MaxTurnsExceeded",
            "UserError",
            "ContextWindowExceededError",
            "BadRequestError",
        ],
        original: AgentsException
        | ContextWindowExceededError
        | TimeoutError
        | BadRequestError
        | openai.APITimeoutError,
    ):
        super().__init__(message)
        self.cause = cause
        self.original = original

    def to_input_list(self) -> list[TResponseInputItem] | None:
        if not isinstance(self.original, AgentsException):
            return None
        run_data = self.original.run_data
        if run_data is None:
            return None

        input_items = ItemHelpers.input_to_new_input_list(run_data.input)
        new_items = [item.to_input_item() for item in run_data.new_items]
        return input_items + new_items


@dataclass
class AgentWrapper[TOutput: BaseModel | str]:
    agent: Agent

    @classmethod
    def create(
        cls,
        name: str,
        instructions: str,
        model: AgentsSDKModel,
        model_settings: ModelSettings | None = None,
        mcp_servers: list[MCPServer] | None = None,
        output_type: type[TOutput] | None = None,
        tools: Iterable[Tool] | None = None,
        tool_use_behavior: Literal["run_llm_again", "stop_on_first_tool"]
        | StopAtTools
        | ToolsToFinalOutputFunction = "run_llm_again",
        reset_tool_choice: bool = True,
    ) -> Self:
        if isinstance(model, (str, Model)):
            agents_sdk_model = model
        else:
            raise ValueError("Unsupported model type")
        if model_settings is None:
            model_settings = get_default_model_settings()
        agent = Agent(
            name=name,
            instructions=instructions,
            model=agents_sdk_model,
            output_type=output_type,
            tools=list(tools) if tools is not None else [],
            mcp_servers=mcp_servers if mcp_servers is not None else [],
            tool_use_behavior=tool_use_behavior,
            reset_tool_choice=reset_tool_choice,
            model_settings=model_settings,
        )
        return cls(agent=agent)

    async def run(
        self,
        input: str | Iterable[TResponseInputItem],
        *,
        context: Any | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        time_out_seconds: float | None = None,
    ) -> RunResultWrapper[TOutput]:
        try:
            input_ = input if isinstance(input, str) else list(input)

            # Temporarily inject turn limit into instructions
            original_instructions = self.agent.instructions
            self.agent.instructions = (
                f"{original_instructions}\n\nTURN LIMIT: {max_turns}"
            )

            try:
                async with timeout(time_out_seconds):
                    result = await Runner.run(
                        self.agent,
                        input=input_,
                        context=context,
                        max_turns=max_turns,
                    )
            finally:
                self.agent.instructions = original_instructions
        except asyncio.TimeoutError as e:
            raise AgentRunFailure(
                str(e),
                cause="Timeout",
                original=e,
            ) from e
        except openai.APITimeoutError as e:
            raise AgentRunFailure(
                str(e),
                cause="Timeout",
                original=e,
            ) from e
        except ModelBehaviorError as e:
            raise AgentRunFailure(
                str(e),
                cause="ModelBehaviourError",
                original=e,
            ) from e
        except MaxTurnsExceeded as e:
            raise AgentRunFailure(
                str(e),
                cause="MaxTurnsExceeded",
                original=e,
            ) from e
        except UserError as e:
            raise AgentRunFailure(
                str(e),
                cause="UserError",
                original=e,
            ) from e
        except ContextWindowExceededError as e:
            raise AgentRunFailure(
                str(e),
                cause="ContextWindowExceededError",
                original=e,
            ) from e
        except BadRequestError as e:
            raise AgentRunFailure(str(e), cause="BadRequestError", original=e) from e
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise e

        return RunResultWrapper[type(result.final_output)](result=result)
