
# oai_utils

`oai_utils` is a utility library to simplify the usage of `openai-agents-sdk` ("The Agents SDK"), providing wrappers and helpers for common agent patterns.

## Features

- **AgentWrapper**: A high-level wrapper around the Agents SDK `Agent` and `Runner` to streamline agent creation and execution.
- **Structured Outputs**: Easy integration with Pydantic models for type-safe structured responses.
- **Function Calling**: Support for Python functions as tools for agents.
- **LiteLLM Integration**: Seamless support for non-OpenAI models (like Gemini, Anthropic, etc.) via LiteLLM.

## Usage

### 1. Basic Agent Execution

Create a simple agent with a system prompt and a model.

```python
import asyncio
from dotenv import load_dotenv
from oai_utils.agent import AgentWrapper

load_dotenv()

async def main():
    agent = AgentWrapper[str].create(
        name="SimpleAgent",
        instructions="You are a helpful assistant. Keep it short.",
        model="gpt-5-mini",
    )
    result = await agent.run("Say hello!")
    print(f"Output: {result.final_output()}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Structured Outputs

Define a Pydantic model and pass it as `output_type` to enforce structured responses.

```python
import asyncio
from dotenv import load_dotenv
from oai_utils.agent import AgentWrapper
from pydantic import BaseModel

load_dotenv()

class MathResult(BaseModel):
    ans: int
    explanation: str

async def main():
    agent = AgentWrapper[MathResult].create(
        name="MathAgent",
        instructions="Solve the math problem.",
        model="gpt-5-mini",
        output_type=MathResult,
    )
    result = await agent.run("What is 10 + 5?")
    # result.final_output() returns an instance of MathResult
    print(f"Output: {result.final_output()}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Function Calling (Tools)

Decorate python functions with `@function_tool` from the `agents` package and pass them to the agent.

```python
import asyncio
from dotenv import load_dotenv
from oai_utils.agent import AgentWrapper
from agents import function_tool

load_dotenv()

async def main():
    @function_tool
    def add(a: int, b: int) -> int:
        """Adds two numbers."""
        return a + b

    agent = AgentWrapper[str].create(
        name="ToolAgent",
        instructions="Use the tool to add numbers.",
        model="gpt-5-mini",
        tools=[add],
    )
    result = await agent.run("Calculate 20 + 30")
    print(f"Output: {result.final_output()}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 4. LiteLLM Integration

Use `LitellmModel` to use models from other providers like Gemini.

```python
import asyncio
import os
from dotenv import load_dotenv
from oai_utils.agent import AgentWrapper
from agents.extensions.models.litellm_model import LitellmModel

load_dotenv()

async def main():
    # Example using Gemini-3 Flash via LiteLLM
    # Ensure GEMINI_API_KEY is set in your environment
    model_name = "gemini/gemini-3-flash-preview"
    
    litellm_model = LitellmModel(model=model_name)
    agent = AgentWrapper[str].create(
        name="LiteAgent",
        instructions="You are a helpful assistant.",
        model=litellm_model,
    )
    result = await agent.run("Hello from Gemini!")
    print(f"Output: {result.final_output()}")

if __name__ == "__main__":
    asyncio.run(main())
```
