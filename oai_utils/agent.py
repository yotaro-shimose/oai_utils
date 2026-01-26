import asyncio
from asyncio import timeout
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Self

from agents import (
    Agent,
    AgentsException,
    ItemHelpers,
    MaxTurnsExceeded,
    Model,
    ModelBehaviorError,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    StopAtTools,
    Tool,
    ToolsToFinalOutputFunction,
    TResponseInputItem,
    UserError,
)
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp.server import MCPServer
from agents.models.openai_responses import OpenAIResponsesModel
from agents.run import DEFAULT_MAX_TURNS
from litellm import ContextWindowExceededError
from openai._exceptions import BadRequestError
from pydantic import BaseModel

from oai_utils.runresult import RunResultWrapper

type AgentsSDKModel = str | Model


class AgentRunFailure(BaseException):
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
        | BadRequestError,
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
        if isinstance(
            model, (str, OpenAIChatCompletionsModel, LitellmModel, OpenAIResponsesModel)
        ):
            agents_sdk_model = model
        else:
            raise ValueError("Unsupported model type")
        kwargs = {}
        if model_settings is not None:
            kwargs["model_settings"] = model_settings
        agent = Agent(
            name=name,
            instructions=instructions,
            model=agents_sdk_model,
            output_type=output_type,
            tools=list(tools) if tools is not None else [],
            mcp_servers=mcp_servers if mcp_servers is not None else [],
            tool_use_behavior=tool_use_behavior,
            reset_tool_choice=reset_tool_choice,
            **kwargs,  # type: ignore
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
            async with timeout(time_out_seconds):
                result = await Runner.run(
                    self.agent,
                    input=input_,
                    context=context,
                    max_turns=max_turns,
                )
        except asyncio.TimeoutError as e:
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
        return RunResultWrapper[type(result.final_output)](result=result)
