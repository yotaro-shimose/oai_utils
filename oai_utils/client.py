import os
from openai.lib.azure import AsyncAzureOpenAI
from agents.models.openai_responses import OpenAIResponsesModel


def get_aoai(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
) -> OpenAIResponsesModel:
    api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("AZURE_OPENAI_API_KEY is not set")

    base_url = base_url or os.getenv("AZURE_OPENAI_BASE_URL")
    if base_url is None:
        raise ValueError("AZURE_OPENAI_BASE_URL is not set")

    api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
    if api_version is None:
        raise ValueError("AZURE_OPENAI_API_VERSION is not set")

    return OpenAIResponsesModel(
        model=model,
        openai_client=AsyncAzureOpenAI(
            api_key=api_key, azure_endpoint=base_url, api_version=api_version
        ),
    )
