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
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp.server import MCPServer
from agents.models.openai_responses import OpenAIResponsesModel
from agents.run import DEFAULT_MAX_TURNS
from litellm import ContextWindowExceededError
from openai._exceptions import BadRequestError
from pydantic import BaseModel
from agents import Model

from oai_utils.runresult import RunResultWrapper
from oai_utils.vllm import VLLMSetup

type AgentsSDKModel = str | Model | VLLMSetup


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
        if isinstance(
            model, (str, OpenAIChatCompletionsModel, LitellmModel, OpenAIResponsesModel)
        ):
            agents_sdk_model = model
        elif isinstance(model, VLLMSetup):
            agents_sdk_model = model.litellm_agentssdk_name().model_name
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
        time_out_seconds: float | None = None,
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
            ) from e
        except ModelBehaviorError as e:
            raise AgentRunFailure(
                str(e),
                cause="ModelBehaviourError",
            ) from e
        except MaxTurnsExceeded as e:
            raise AgentRunFailure(
                str(e),
                cause="MaxTurnsExceeded",
            ) from e
        except UserError as e:
            raise AgentRunFailure(
                str(e),
                cause="UserError",
            ) from e
        except ContextWindowExceededError as e:
            raise AgentRunFailure(
                str(e),
                cause="ContextWindowExceededError",
            ) from e
        except BadRequestError as e:
            raise AgentRunFailure(str(e), cause="BadRequestError") from e
        return RunResultWrapper[type(result.final_output)](result=result)
