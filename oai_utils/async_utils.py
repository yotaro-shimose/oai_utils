import asyncio
from typing import Awaitable, Iterable, TypeVar

from tqdm.asyncio import tqdm

T = TypeVar("T")


async def gather_with_semaphore(
    tasks: Iterable[Awaitable[T]],
    max_concurrent: int,
    progress_bar: bool = False,
    **tqdm_kwargs,
) -> list[T]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def worker(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    worker_tasks = [worker(task) for task in tasks]

    if progress_bar:
        return await tqdm.gather(*worker_tasks, **tqdm_kwargs)
    return await asyncio.gather(*worker_tasks)
