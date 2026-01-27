from contextlib import asynccontextmanager

import httpx
import litellm


@asynccontextmanager
async def litellm_concurrent_limit(
    max_concurrent: int = 100,
):
    async with httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max_concurrent, max_keepalive_connections=max_concurrent
        )
    ) as session:
        litellm.aclient_session = session
        yield session
