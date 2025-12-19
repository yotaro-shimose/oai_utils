from agents.mcp.server import MCPServerStdio, MCPServerStdioParams


def filesystem_mcp(
    allowed_directories: list[str], client_session_timeout_seconds: int = 5, read_only: bool = False
) -> MCPServerStdio:
    args = [
        "-y",
        "@modelcontextprotocol/server-filesystem",
    ]
    if read_only:
        args.append("--ro")
    return MCPServerStdio(
        params=MCPServerStdioParams(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                *allowed_directories,
            ],
        ),
        client_session_timeout_seconds=client_session_timeout_seconds,
    )
