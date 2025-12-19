import os

from agents.mcp.server import MCPServerStreamableHttp, MCPServerStreamableHttpParams


def github_mcp(github_pat: str | None = None) -> MCPServerStreamableHttp:
    if github_pat is None:
        github_pat = os.getenv("GITHUB_PAT")
        if github_pat is None:
            raise ValueError("Pass github_pat or set GITHUB_PAT environment variable")
    return MCPServerStreamableHttp(
        params=MCPServerStreamableHttpParams(
            url="https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {github_pat}"},
        )
    )
