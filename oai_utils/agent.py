import asyncio
from asyncio import timeout
from dataclasses import dataclass
from typing import Any, Literal, Self, Sequence

from agents import (
    Agent,
    MaxTurnsExceeded,
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
from agents.mcp.server import MCPServer
from agents.run import DEFAULT_MAX_TURNS
from litellm import ContextWindowExceededError
from pydantic import BaseModel

from oai_wrapper.runresult import RunResultWrapper

type AgentsSDKModel = str | OpenAIChatCompletionsModel


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
        ],
    ):
        super().__init__(message)
        self.cause = cause


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
        tools: Sequence[Tool] | None = None,
        tool_use_behavior: Literal["run_llm_again", "stop_on_first_tool"]
        | StopAtTools
        | ToolsToFinalOutputFunction = "run_llm_again",
    ) -> Self:
        if isinstance(model, (str, OpenAIChatCompletionsModel)):
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
            **kwargs,  # type: ignore
        )
        return cls(agent=agent)

    async def run(
        self,
        input: str | list[TResponseInputItem],
        *,
        context: Any | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        time_out_seconds: float = 120.0,
    ) -> RunResultWrapper[TOutput]:
        try:
            async with timeout(time_out_seconds):
                result = await Runner.run(
                    self.agent,
                    input=input,
                    context=context,
                    max_turns=max_turns,
                )
        except asyncio.TimeoutError as e:
            raise AgentRunFailure(
                str(e),
                cause="Timeout",
            )
        except ModelBehaviorError as e:
            raise AgentRunFailure(
                str(e),
                cause="ModelBehaviourError",
            )
        except MaxTurnsExceeded as e:
            raise AgentRunFailure(
                str(e),
                cause="MaxTurnsExceeded",
            )
        except UserError as e:
            raise AgentRunFailure(
                str(e),
                cause="UserError",
            )
        except ContextWindowExceededError as e:
            raise AgentRunFailure(
                str(e),
                cause="ContextWindowExceededError",
            )
        return RunResultWrapper[type(result.final_output)](result=result)
