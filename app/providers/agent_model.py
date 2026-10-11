"""One provider step at a time; the async agent owns execution and cancellation."""
from __future__ import annotations

import json

from app.providers.llm_client import LlmClient


class AgentModelSession:
    def __init__(self, client):
        self.client = client
        self.history = []
        self.pending = ()

    def step(self, lines, tools, images, feedback):
        client = self.client
        if not isinstance(client, LlmClient) or not client._responses_enabled():
            options = {"allow_web_search": False, "conversation_key": None}
            if images:
                options["images"] = list(images)
            return client.generate_text(lines, **options)
        if self.pending:
            output = json.dumps(feedback[-1] if feedback else {"error": "not_executed"}, ensure_ascii=False)
            for index, call in enumerate(self.pending):
                self.history.append({"type": "function_call_output", "call_id": call.call_id,
                                     "output": output if index == 0 else '{"error":"parallel_calls_not_executed"}'})
            self.pending = ()
        instructions, input_lines = client._split_prompt_lines(lines)
        model = client._image_chat_model(default_model=client.responses_model) if images else client.responses_model
        payload = client._build_responses_payload(
            model=model, instructions=instructions, input_lines=input_lines, images=list(images),
            tools=[{"type": "function", "name": tool.name, "description": tool.description,
                    "parameters": {**tool.parameters, "additionalProperties": False}} for tool in tools],
            extra_input_items=self.history, allow_web_search=False, max_output_tokens=client.max_output_tokens,
        )
        payload["parallel_tool_calls"] = False
        result = client._request_responses_stream_result(responses_payload=payload, model=model)
        client._record_usage(result.usage)
        self.history.extend(result.reasoning_items)
        for call in result.function_calls:
            self.history.append({"type": "function_call", "call_id": call.call_id,
                                 "name": call.name, "arguments": call.arguments})
        self.pending = result.function_calls
        if self.pending:
            try:
                args = json.loads(self.pending[0].arguments)
            except (ValueError, TypeError):
                args = None
            return json.dumps({"tool": self.pending[0].name, "arguments": args}, ensure_ascii=False)
        return result.text or '{"finish":true}'
