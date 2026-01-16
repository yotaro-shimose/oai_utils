import os

from agents import set_tracing_export_api_key
from agents.tracing import TracingProcessor
from agents.tracing.span_data import (
    AgentSpanData,
    FunctionSpanData,
    ResponseSpanData,
)
from dotenv import load_dotenv


def setup_openai_tracing():
    load_dotenv()
    set_tracing_export_api_key(os.environ["OPENAI_API_KEY"])


class AgentContentPrinter(TracingProcessor):
    """Prints agent activity for observability.

    This processor logs tool calls, agent messages, and other activity
    to stdout for debugging and monitoring.

    Example:
        from agents.tracing import add_trace_processor
        from oai_utils.tracing import AgentContentPrinter

        add_trace_processor(AgentContentPrinter())
    """

    def on_trace_start(self, trace):
        print(f"\n🚀 TRACE START: {trace.name}")

    def on_trace_end(self, trace):
        pass

    def on_span_start(self, span):
        pass

    def on_span_end(self, span):
        data = span.span_data

        if isinstance(data, AgentSpanData):
            print(f"\n👤 AGENT: {data.name}")

        elif isinstance(data, FunctionSpanData):
            print(f"\n🛠️  TOOL CALL: {data.name}")
            print(f"   Input: {data.input}")
            if data.output:
                outcome = str(data.output)
                if len(outcome) > 500:
                    outcome = outcome[:500] + "..."
                print(f"   Outcome: {outcome}")

        elif isinstance(data, ResponseSpanData):
            response = getattr(data, "response", None)
            if response:
                items = getattr(response, "output_items", [])
                for item in items:
                    if hasattr(item, "type") and "message" in item.type:
                        content = None
                        if hasattr(item, "message") and hasattr(
                            item.message, "content"
                        ):
                            content = item.message.content
                        elif hasattr(item, "content"):
                            content = item.content
                        if content:
                            print(f"\n🤖 AGENT MESSAGE:\n{content}")

    def force_flush(self):
        pass

    def shutdown(self):
        pass
