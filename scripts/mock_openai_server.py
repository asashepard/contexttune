#!/usr/bin/env python3
"""Minimal mock OpenAI-compatible server for testing.

Returns a dummy diff response for any chat completion request.
Usage: python scripts/mock_openai_server.py [port]
"""
from __future__ import annotations

import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

MOCK_DIFF = """```diff
--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def hello():
-    return "Hello"
+    return "Hello, World!"
```"""


class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        request = json.loads(body) if body else {}

        # Echo some info to stderr for debugging
        model = request.get("model", "unknown")
        messages = request.get("messages", [])
        print(f"Received request: model={model}, messages={len(messages)}", file=sys.stderr)

        response = {
            "id": "mock-123",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": MOCK_DIFF,
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = HTTPServer(("127.0.0.1", port), MockHandler)
    print(f"Mock OpenAI server running on http://127.0.0.1:{port}/v1")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
