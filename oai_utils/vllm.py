from agents.extensions.models.litellm_model import LitellmModel
from pydantic import Field
from pathlib import Path
import subprocess
import time
from typing import Self, Any
import httpx
import json

try:
    import torch
except ImportError:
    torch = None

from openai import AsyncOpenAI
from pydantic import BaseModel, InstanceOf


class RopeScaling(BaseModel):
    rope_type: str = "yarn"
    factor: float
    original_max_position_embeddings: int


class VLLMSetup(BaseModel):
    model: str
    lora_adapters: dict[str, Path] = Field(default_factory=dict)
    port: int = 5222
    max_model_len: int = 32768
    api_key: str = "your_api_key_here"
    vllm_process: InstanceOf[subprocess.Popen | None] = None
    data_parallel_size: int | None = None
    reasoning_parser: str | None = None
    rope_scaling: RopeScaling | None = Field(default=None)
    quantization: str | None = Field(default=None)

    @classmethod
    def qwen3(cls, **kwargs) -> Self:
        return cls(
            model="Qwen/Qwen3-4B",
            reasoning_parser="deepseek_r1",
            **kwargs,
        )

    @classmethod
    def qwen3_reasoning(cls, **kwargs) -> Self:
        return cls(
            model="Qwen/Qwen3-4B-Thinking-2507",
            reasoning_parser="deepseek_r1",
            **kwargs,
        )

    @classmethod
    def phi4(cls, **kwargs) -> Self:
        return cls(model="microsoft/Phi-4-mini-instruct", **kwargs)

    @classmethod
    def phi4_reasoning(cls, **kwargs) -> Self:
        return cls(
            model="microsoft/Phi-4-mini-reasoning",
            **kwargs,
        )

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    async def is_vllm_running(self) -> bool:
        url = f"{self.base_url}/health"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url)
                return response.status_code == 200
            except httpx.ConnectError:
                return False

    def launch_vllm_server(self) -> subprocess.Popen:
        yarn_scaling = (
            self.rope_scaling.factor if self.rope_scaling is not None else 1.0
        )
        commands: list[str] = [
            "vllm",
            "serve",
            self.model,
            "--port",
            str(self.port),
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--max-model-len",
            str(int(self.max_model_len * yarn_scaling)),
        ]
        if self.lora_adapters:
            commands.extend(["--enable-lora", "--lora-modules"])
            for adapter_name, adapter_path in self.lora_adapters.items():
                commands.append(f"{adapter_name}={adapter_path}")
        if self.reasoning_parser is not None:
            commands.extend(
                [
                    "--reasoning-parser",
                    self.reasoning_parser,
                ]
            )
        if self.data_parallel_size is None:
            if torch is None:
                raise ImportError(
                    "torch is not installed. Please install oai_utils[vllm] to use VLLMSetup."
                )
            device_count = torch.cuda.is_available()
        else:
            device_count = self.data_parallel_size
        if device_count > 1:
            commands.extend(
                [
                    "--data-parallel-size",
                    str(device_count),
                    "--max-num-seqs",
                    f"{256 * device_count}",
                ]
            )
        if self.quantization:
            commands.extend(
                [
                    "--quantization",
                    self.quantization,
                ]
            )
        if self.rope_scaling:
            hf_overrides = {
                "rope_parameters": self.rope_scaling.model_dump(),
            }
            commands.extend(["--hf-overrides", json.dumps(hf_overrides)])
        vllm_process = subprocess.Popen(commands)
        self.vllm_process = vllm_process
        return vllm_process

    def wait_for_server(self, timeout: int = 180) -> None:
        url = f"{self.base_url}/health"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = httpx.get(url, headers=headers)
                if response.status_code == 200:
                    return
            except httpx.ConnectError:
                time.sleep(5)
        raise TimeoutError("VLLM server did not start within the given timeout.")

    async def ensure_vllm_running(self) -> None:
        # Setup vLLM server if not running
        if not await self.is_vllm_running():
            print("VLLM server not running. Launching...")
            process = self.launch_vllm_server()
            try:
                self.wait_for_server()
                print("VLLM server is up and running.")
            except TimeoutError as e:
                process.terminate()
                raise e
        else:
            print("VLLM server is already running.")

    def get_openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=self.api_key,
        )

    @property
    def effective_model_name(self, lora: str | None = None) -> str:
        """Returns the adapter name if LoRA is used, otherwise the base model."""
        if lora is not None:
            return lora
        return self.model

    def litellm_model(self, lora: str | None = None) -> str:
        if lora is not None:
            return f"openai/{lora}"
        return f"openai/{self.model}"

    def as_litellm_model(self, lora: str | None = None) -> LitellmModel:
        # Use effective_model_name here as well
        return LitellmModel(
            model=self.litellm_model(lora),
            base_url=f"{self.base_url}/v1",
            api_key=self.api_key,
        )
