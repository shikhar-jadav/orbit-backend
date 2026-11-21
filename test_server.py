import os
import sys
import json
import base64
import subprocess
from io import BytesIO
import tempfile
import unittest
import wave
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import patch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import server
from chat_store import PostgresChatStore
from test_database import IsolatedPostgresDatabase
from config_manager import RoundRobinProviderManager
from agent import ReActAgent
from tools import AGENT_TOOLS, calculate, get_current_time, get_weather


def _make_isolated_user_and_thread(store: PostgresChatStore):
    user, _ = store.create_user(
        email="tester@example.com",
        display_name="Test User",
        password_hash=server.auth.hash_password("TestPassword123!"),
    )
    user["session_id"] = "sess-test"
    thread = store.create_thread(user_id=user["user_id"])
    return user, thread


class TestServerStructureAndRoutes(unittest.TestCase):
    def setUp(self):
        mock_env = {
            "API_KEY1": "sk-test-openai-111",
            "BASE_URL1": "https://api.openai.com/v1",
            "MODEL1": "gpt-4o-mini",
            "MODEL_LIST1": "gpt-4o-mini,gpt-4o",
            "PROVIDER_NAME1": "Test-OpenAI",

            "API_KEY2": "gsk-test-groq-222",
            "BASE_URL2": "https://api.groq.com/openai/v1",
            "MODEL2": "llama-3.3-70b-versatile",
            "MODEL_LIST2": "llama-3.3-70b-versatile",
            "PROVIDER_NAME2": "Test-Groq",
        }
        self.mock_manager = RoundRobinProviderManager(env_dict=mock_env)
        self.mock_agent = ReActAgent(provider_manager=self.mock_manager, tools=AGENT_TOOLS)

        server.provider_manager = self.mock_manager
        server.agent = self.mock_agent

    def test_routes_registered(self):
        """Verifies that all required endpoints are mounted on FastAPI app."""
        routes = {route.path for route in server.app.routes}
        thread_methods = set().union(
            *(
                route.methods
                for route in server.app.routes
                if route.path == "/threads/{thread_id}"
            )
        )
        self.assertIn("/", routes)
        self.assertIn("/health", routes)
        self.assertIn("/providers", routes)
        self.assertIn("/chat", routes)
        self.assertIn("/chat/stream", routes)
        self.assertIn("/chat/regenerate", routes)
        self.assertIn("/ag-ui", routes)
        self.assertIn("/threads", routes)
        self.assertIn("/threads/{thread_id}", routes)
        self.assertIn("/threads/{thread_id}/attachments", routes)
        self.assertIn("PUT", thread_methods)
        self.assertIn("/threads/{thread_id}/branch", routes)
        self.assertIn(
            "/threads/{thread_id}/turns/{user_message_id}/version", routes
        )
        self.assertIn("/settings", routes)
        for auth_route in (
            "/auth/session",
            "/auth/signup",
            "/auth/login",
            "/auth/logout",
            "/auth/recovery/reset",
            "/auth/me",
            "/auth/profile",
            "/auth/password",
            "/auth/sessions",
            "/auth/account/delete",
        ):
            self.assertIn(auth_route, routes)

    def test_thread_ownership_guard_uses_lightweight_lookup(self):
        with patch.object(server, "chat_store") as store:
            store.thread_exists.return_value = True
            server._require_owned_thread("thread-1", {"user_id": "user-1"})

            store.thread_exists.assert_called_once_with(
                "thread-1", user_id="user-1"
            )
            store.get_thread.assert_not_called()

    def test_providers_endpoint_data(self):
        """Verifies provider manager status integration."""
        status = server.provider_manager.get_status()
        self.assertEqual(status["total_providers"], 2)
        self.assertEqual(len(status["providers"]), 2)
        self.assertEqual(status["providers"][0]["name"], "Test-OpenAI")
        self.assertEqual(status["providers"][1]["name"], "Test-Groq")

    def test_schemas_validation(self):
        """Verifies Pydantic request and response schemas."""
        req = server.ChatRequest(
            message="Calculate 25 * 4",
            history=[server.ChatMessage(role="user", content="Hi")]
        )
        self.assertEqual(req.message, "Calculate 25 * 4")
        self.assertEqual(len(req.history), 1)

        resp = server.ChatResponse(
            response="100",
            provider_used={"name": "Test-OpenAI", "model": "gpt-4o-mini"}
        )
        self.assertEqual(resp.response, "100")

    def test_round_robin_rotation(self):
        """Verifies key rotation wrap-around logic."""
        p1 = self.mock_manager.get_next_provider()
        p2 = self.mock_manager.get_next_provider()
        p3 = self.mock_manager.get_next_provider()

        self.assertEqual(p1.name, "Test-OpenAI")
        self.assertEqual(p2.name, "Test-Groq")
        self.assertEqual(p3.name, "Test-OpenAI")  # Wraps around to 1!

    def test_tools_available(self):
        self.assertEqual(len(AGENT_TOOLS), 3)
        tool_names = {t.name for t in AGENT_TOOLS}
        self.assertEqual(tool_names, {"calculate", "get_current_time", "get_weather"})

        from tools import ALL_AGENT_TOOLS, render_a2ui_canvas, render_generative_ui
        self.assertEqual(len(ALL_AGENT_TOOLS), 5)
        all_tool_names = {t.name for t in ALL_AGENT_TOOLS}
        self.assertEqual(
            all_tool_names,
            {"calculate", "get_current_time", "get_weather", "render_a2ui_canvas", "render_generative_ui"}
        )

        # Direct execution of A2UI Canvas tool
        a2_out = render_a2ui_canvas.invoke({
            "title": "Test Telemetry",
            "spec": json.dumps({"type": "metric", "label": "CPU", "value": "45%"})
        })
        self.assertIn("Test Telemetry", a2_out)
        self.assertIn("metric", a2_out)

        # Direct execution of Generative UI tool
        gen_out = render_generative_ui.invoke({
            "title": "Interactive Slider",
            "html_content": "<input type='range' min='0' max='100' class='w-full' />",
            "description": "Range slider"
        })
        self.assertIn("generative_ui_rendered", gen_out)
        self.assertIn("Interactive Slider", gen_out)


class TestAGUIEndpoint(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_store = server.chat_store
        self.original_agent = server.agent
        self.original_manager = server.provider_manager
        self.test_database = IsolatedPostgresDatabase()
        self.temp_dir = tempfile.TemporaryDirectory()
        server.chat_store = PostgresChatStore(self.test_database.database)
        self.test_user, self.thread = _make_isolated_user_and_thread(server.chat_store)
        server.provider_manager = RoundRobinProviderManager(
            env_dict={"API_KEY1": "test-key", "MODEL1": "test-model"}
        )

        class StubAgent:
            def get_thread_history(self, thread_id):
                return []

            def save_user_checkpoint(self, thread_id, message):
                pass

            def save_assistant_checkpoint(self, thread_id, message):
                pass

            async def astream_run(self, **kwargs):
                yield {"type": "provider_selected", "data": {"provider": "Test", "model": "mock"}}
                yield {"type": "token", "data": {"token": "Hello"}}
                yield {"type": "done", "data": {"content": "Hello"}}

        server.agent = StubAgent()

    async def asyncTearDown(self):
        server.chat_store = self.original_store
        server.agent = self.original_agent
        server.provider_manager = self.original_manager
        self.test_database.close()
        self.temp_dir.cleanup()

    async def test_streams_agui_events(self):
        run_input = server.RunAgentInput(
            threadId=self.thread["thread_id"],
            runId="run-test",
            messages=[{"id": "user-1", "role": "user", "content": "Hi"}],
        )
        response = await server.agui_endpoint(run_input, current_user=self.test_user)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8"))
        body = b"".join(chunks).decode("utf-8")
        events = [
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]

        self.assertEqual(events[0]["type"], "RUN_STARTED")
        self.assertIn("TEXT_MESSAGE_CONTENT", [event["type"] for event in events])
        self.assertIn("RUN_FINISHED", [event["type"] for event in events])
        self.assertIn("CUSTOM", [event["type"] for event in events])


class TestConversationVersionEndpoints(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_store = server.chat_store
        self.original_agent = server.agent
        self.original_manager = server.provider_manager
        self.test_database = IsolatedPostgresDatabase()
        self.temp_dir = tempfile.TemporaryDirectory()
        server.chat_store = PostgresChatStore(self.test_database.database)
        self.test_user, self.thread = _make_isolated_user_and_thread(server.chat_store)
        self.branch_ids = []
        user = server.chat_store.add_message(
            self.thread["thread_id"], "user", "Original prompt"
        )
        assistant = server.chat_store.add_message(
            self.thread["thread_id"], "assistant", "Original answer"
        )
        server.chat_store.add_turn_version(
            user["id"], assistant["id"], "Original prompt", "Original answer"
        )
        server.chat_store.add_message(
            self.thread["thread_id"], "user", "A later question"
        )
        server.chat_store.add_message(
            self.thread["thread_id"], "assistant", "A later answer"
        )
        self.user_message_id = user["id"]

        class StubAgent:
            def __init__(self):
                self.replaced_histories = []

            async def astream_run(self, **kwargs):
                yield {"type": "token", "data": {"token": "Generated"}}
                yield {"type": "done", "data": {"content": "Generated answer"}}

            def replace_thread_history(self, thread_id, history):
                self.replaced_histories.append((thread_id, history))

        self.stub_agent = StubAgent()
        server.agent = self.stub_agent
        server.provider_manager = RoundRobinProviderManager(
            env_dict={"API_KEY1": "test-key", "MODEL1": "test-model"}
        )

    async def asyncTearDown(self):
        server.chat_store = self.original_store
        server.agent = self.original_agent
        server.provider_manager = self.original_manager
        self.test_database.close()
        self.temp_dir.cleanup()

    async def test_regenerate_select_and_branch_response_versions(self):
        response = await server.regenerate_chat_endpoint(
            server.RegenerateRequest(
                thread_id=self.thread["thread_id"],
                user_message_id=self.user_message_id,
                message="Edited prompt",
            ),
            current_user=self.test_user,
        )
        chunks = [
            chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            async for chunk in response.body_iterator
        ]
        events = [
            json.loads(line[6:])
            for line in "".join(chunks).splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events[0]["type"], "turn_version_start")
        self.assertEqual(events[-1]["data"]["version"], 2)

        loaded = server.chat_store.get_thread(self.thread["thread_id"])
        self.assertEqual(
            [message["content"] for message in loaded["messages"]],
            ["Edited prompt", "Generated answer"],
        )
        self.assertEqual(len(loaded["messages"][1]["variants"]), 2)
        self.assertEqual(
            self.stub_agent.replaced_histories[-1][1],
            [
                {"role": "user", "content": "Edited prompt"},
                {"role": "assistant", "content": "Generated answer"},
            ],
        )

        selected = await server.select_turn_version(
            self.thread["thread_id"],
            self.user_message_id,
            server.TurnVersionRequest(
                user_message_id=self.user_message_id, version=1
            ),
            current_user=self.test_user,
        )
        self.assertEqual(selected["messages"][1]["content"], "Original answer")

        branched = await server.branch_thread(
            self.thread["thread_id"],
            server.TurnVersionRequest(
                user_message_id=self.user_message_id, version=2
            ),
            current_user=self.test_user,
        )
        self.branch_ids.append(branched["thread"]["thread_id"])
        self.assertEqual(branched["messages"][0]["content"], "Edited prompt")
        self.assertEqual(branched["messages"][1]["content"], "Generated answer")

    async def test_chat_stream_persists_tools_and_parts_on_reload(self):
        class ToolStubAgent:
            def get_thread_history(self, thread_id):
                return []

            def save_user_checkpoint(self, thread_id, user_query):
                pass

            def save_assistant_checkpoint(self, thread_id, response):
                pass

            async def astream_run(self, **kwargs):
                yield {"type": "token", "data": {"token": "Writing file..."}}
                yield {
                    "type": "tool_start",
                    "data": {
                        "tool": "write_workspace_file",
                        "args": {"file_path": "essay.md"},
                        "tool_call_id": "call-1",
                    },
                }
                yield {
                    "type": "tool_end",
                    "data": {
                        "tool": "write_workspace_file",
                        "output": json.dumps({"status": "written", "file_path": "essay.md"}),
                        "tool_call_id": "call-1",
                    },
                }
                yield {"type": "token", "data": {"token": " Done."}}
                yield {"type": "done", "data": {"content": "Writing file... Done."}}

        server.agent = ToolStubAgent()
        req = server.ChatRequest(
            thread_id=self.thread["thread_id"],
            message="make essay",
        )
        response = await server.chat_stream_endpoint(req, current_user=self.test_user)
        chunks = [
            chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            async for chunk in response.body_iterator
        ]
        self.assertTrue(any("workspace_file_changed" in c for c in chunks))

        # Simulate browser refresh by fetching thread
        reloaded = server.chat_store.get_thread(self.thread["thread_id"])
        assistant_msg = reloaded["messages"][-1]
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertEqual(len(assistant_msg["tools"]), 1)
        self.assertEqual(assistant_msg["tools"][0]["name"], "write_workspace_file")
        self.assertEqual(assistant_msg["tools"][0]["status"], "done")
        self.assertEqual(len(assistant_msg["parts"]), 3)
        self.assertEqual(assistant_msg["parts"][0]["type"], "text")
        self.assertEqual(assistant_msg["parts"][0]["text"], "Writing file...")
        self.assertEqual(assistant_msg["parts"][1]["type"], "tool")
        self.assertEqual(assistant_msg["parts"][1]["tool"]["name"], "write_workspace_file")
        self.assertEqual(assistant_msg["parts"][2]["type"], "text")
        self.assertEqual(assistant_msg["parts"][2]["text"], " Done.")


class TestAttachmentEndpoints(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_store = server.chat_store
        self.original_uploads_dir = server.UPLOADS_DIR
        self.original_agent = server.agent
        self.original_manager = server.provider_manager
        self.test_database = IsolatedPostgresDatabase()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        server.chat_store = PostgresChatStore(self.test_database.database)
        server.UPLOADS_DIR = root / "uploads"
        self.test_user, self.thread = _make_isolated_user_and_thread(server.chat_store)

        class StubAgent:
            def __init__(self):
                self.user_inputs = []
                self.calls = []

            def get_thread_history(self, thread_id):
                return []

            def save_user_checkpoint(self, thread_id, user_query):
                pass

            def save_assistant_checkpoint(self, thread_id, response):
                pass

            async def astream_run(self, **kwargs):
                self.user_inputs.append(kwargs["user_query"])
                self.calls.append(kwargs)
                yield {"type": "done", "data": {"content": "Read the note."}}

        self.stub_agent = StubAgent()
        server.agent = self.stub_agent
        server.provider_manager = RoundRobinProviderManager(
            env_dict={"API_KEY1": "test-key", "MODEL1": "test-model"}
        )

    async def asyncTearDown(self):
        server.chat_store = self.original_store
        server.UPLOADS_DIR = self.original_uploads_dir
        server.agent = self.original_agent
        server.provider_manager = self.original_manager
        self.test_database.close()
        self.temp_dir.cleanup()

    async def test_uploaded_text_is_saved_and_sent_with_message(self):
        from starlette.requests import Request

        body = b"Orbit keeps this note."
        events = iter(
            [{"type": "http.request", "body": body, "more_body": False}]
        )

        async def receive():
            return next(events)

        request = Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/",
                "raw_path": b"/",
                "query_string": b"",
                "headers": [
                    (b"content-type", b"text/plain"),
                    (b"content-length", str(len(body)).encode()),
                ],
                "client": ("127.0.0.1", 1234),
                "server": ("test", 80),
            },
            receive,
        )
        uploaded = await server.upload_attachment(
            self.thread["thread_id"], request, "notes.txt", current_user=self.test_user
        )
        self.assertEqual(uploaded["filename"], "notes.txt")

        response = await server.chat_stream_endpoint(
            server.ChatRequest(
                message="",
                thread_id=self.thread["thread_id"],
                attachment_ids=[uploaded["attachment_id"]],
            ),
            current_user=self.test_user,
        )
        async for _ in response.body_iterator:
            pass

        thread = server.chat_store.get_thread(self.thread["thread_id"])
        self.assertEqual(thread["messages"][0]["content"], "Please review the attached files.")
        self.assertEqual(thread["messages"][0]["attachments"][0]["filename"], "notes.txt")
        self.assertIn("Orbit keeps this note.", self.stub_agent.user_inputs[0])

    async def test_document_follow_up_uses_latest_attachment_by_default(self):
        first_attachment = server.chat_store.add_attachment(
            self.thread["thread_id"],
            "first-file",
            "trigonometry.txt",
            "text/plain",
            "trigonometry.txt",
            30,
            "TRIGONOMETRY CONTENT ONLY",
        )
        first_message = server.chat_store.add_message(
            self.thread["thread_id"], "user", "Explain this"
        )
        server.chat_store.attach_to_message(
            self.thread["thread_id"], first_message["id"], [first_attachment["attachment_id"]]
        )
        server.chat_store.add_message(
            self.thread["thread_id"], "assistant", "A brief trigonometry summary."
        )

        latest_attachment = server.chat_store.add_attachment(
            self.thread["thread_id"],
            "latest-file",
            "chemistry.txt",
            "text/plain",
            "chemistry.txt",
            28,
            "CHEMICAL BONDING CONTENT ONLY",
        )
        latest_message = server.chat_store.add_message(
            self.thread["thread_id"], "user", "Explain this"
        )
        server.chat_store.attach_to_message(
            self.thread["thread_id"],
            latest_message["id"],
            [latest_attachment["attachment_id"]],
        )
        server.chat_store.add_message(
            self.thread["thread_id"], "assistant", "A brief chemistry summary."
        )

        response = await server.chat_stream_endpoint(
            server.ChatRequest(
                message="Explain this in detail.",
                thread_id=self.thread["thread_id"],
            ),
            current_user=self.test_user,
        )
        async for _ in response.body_iterator:
            pass

        call = self.stub_agent.calls[-1]
        self.assertIn("CHEMICAL BONDING CONTENT ONLY", call["user_query"])
        self.assertNotIn("TRIGONOMETRY CONTENT ONLY", call["user_query"])
        history_text = "\n".join(item["content"] for item in call["chat_history"])
        self.assertIn("chemistry.txt", history_text)
        self.assertNotIn("TRIGONOMETRY CONTENT ONLY", history_text)

    async def test_document_follow_up_can_select_older_file_by_name(self):
        older_attachment = server.chat_store.add_attachment(
            self.thread["thread_id"],
            "older-file",
            "trigonometry.txt",
            "text/plain",
            "trigonometry.txt",
            30,
            "TRIGONOMETRY CONTENT ONLY",
        )
        older_message = server.chat_store.add_message(
            self.thread["thread_id"], "user", "Explain this"
        )
        server.chat_store.attach_to_message(
            self.thread["thread_id"], older_message["id"], [older_attachment["attachment_id"]]
        )
        server.chat_store.add_message(
            self.thread["thread_id"], "assistant", "A brief trigonometry summary."
        )
        latest_attachment = server.chat_store.add_attachment(
            self.thread["thread_id"],
            "latest-file",
            "chemistry.txt",
            "text/plain",
            "chemistry.txt",
            28,
            "CHEMICAL BONDING CONTENT ONLY",
        )
        latest_message = server.chat_store.add_message(
            self.thread["thread_id"], "user", "Explain this"
        )
        server.chat_store.attach_to_message(
            self.thread["thread_id"],
            latest_message["id"],
            [latest_attachment["attachment_id"]],
        )

        response = await server.chat_stream_endpoint(
            server.ChatRequest(
                message="Explain trigonometry.txt in detail.",
                thread_id=self.thread["thread_id"],
            ),
            current_user=self.test_user,
        )
        async for _ in response.body_iterator:
            pass

        call = self.stub_agent.calls[-1]
        self.assertIn("TRIGONOMETRY CONTENT ONLY", call["user_query"])
        self.assertNotIn("CHEMICAL BONDING CONTENT ONLY", call["user_query"])
        history_text = "\n".join(item["content"] for item in call["chat_history"])
        self.assertIn("trigonometry.txt", history_text)
        self.assertNotIn("chemistry.txt", history_text)

    async def test_docx_text_and_image_content_are_prepared_for_the_model(self):
        document = BytesIO()
        with ZipFile(document, "w") as archive:
            archive.writestr(
                "word/document.xml",
                b'<w:document xmlns:w="urn:word"><w:body>'
                b'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
                b'<w:r><w:t>Project note</w:t></w:r></w:p>'
                b'<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Item</w:t></w:r></w:p></w:tc>'
                b'<w:tc><w:p><w:r><w:t>Status</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
                b"</w:body></w:document>",
            )
        self.assertEqual(
            server.extract_attachment_text(
                "notes.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                document.getvalue(),
            ),
            "[Heading1] Project note\n| Item | Status |",
        )

        image_bytes = b"\x89PNG\r\n\x1a\nimage"
        server.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        image_path = server.UPLOADS_DIR / "image.png"
        image_path.write_bytes(image_bytes)
        model_input = server.build_attachment_input(
            [
                {
                    "filename": "image.png",
                    "content_type": "image/png",
                    "storage_name": "image.png",
                    "extracted_text": "",
                }
            ],
            "Describe this image.",
        )
        self.assertEqual(
            model_input[0]["text"],
            "Describe this image.\n\nAttached image: image.png",
        )
        self.assertEqual(model_input[1]["type"], "image")
        self.assertEqual(
            model_input[1]["base64"],
            base64.b64encode(image_bytes).decode("ascii"),
        )
        self.assertEqual(model_input[1]["mime_type"], "image/png")
        from imageio_ffmpeg import get_ffmpeg_exe

        convertible_path = server.UPLOADS_DIR / "converted.png"
        image_result = subprocess.run(
            [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=16x16:d=1",
                "-frames:v",
                "1",
                "-y",
                str(convertible_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(image_result.returncode, 0, image_result.stderr)
        converted_image = server.make_image_input(
            convertible_path, "image.bmp", "image/bmp"
        )
        self.assertEqual(converted_image["type"], "image")
        self.assertEqual(converted_image["mime_type"], "image/png")

        pdf_input = server.build_attachment_input(
            [
                {
                    "filename": "scan.pdf",
                    "content_type": "application/pdf",
                    "storage_name": "scan.pdf",
                    "extracted_text": "Extracted PDF text.",
                }
            ],
            "Read this scan.",
        )
        self.assertIsInstance(pdf_input, str)
        self.assertIn("Attached document: scan.pdf", pdf_input)
        self.assertIn("Extracted PDF text.", pdf_input)
        self.assertNotIn("base64", pdf_input)

        scanned_pdf_input = server.build_attachment_input(
            [
                {
                    "filename": "scanned.pdf",
                    "content_type": "application/pdf",
                    "storage_name": "scan.pdf",
                    "extracted_text": "",
                }
            ],
            "Read this scan.",
        )
        self.assertIsInstance(scanned_pdf_input, str)
        self.assertIn("may be scanned and require OCR", scanned_pdf_input)

    async def test_pptx_and_xlsx_keep_slide_and_table_structure(self):
        presentation = BytesIO()
        with ZipFile(presentation, "w") as archive:
            archive.writestr(
                "ppt/slides/slide1.xml",
                b'<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:p><a:r><a:t>Quarter</a:t>'
                b'</a:r></a:p><a:p><a:r><a:t>Revenue</a:t></a:r></a:p></p:sld>',
            )
        self.assertEqual(
            server.extract_attachment_text(
                "report.pptx", "application/vnd.openxmlformats", presentation.getvalue()
            ),
            "[Slide 1]\nQuarter\nRevenue",
        )

        spreadsheet = BytesIO()
        with ZipFile(spreadsheet, "w") as archive:
            archive.writestr(
                "xl/sharedStrings.xml",
                b'<sst xmlns="urn:x"><si><t>Name</t></si><si><t>Value</t></si>'
                b"<si><t>Orbit</t></si><si><t>42</t></si></sst>",
            )
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                b'<worksheet xmlns="urn:x"><sheetData><row><c t="s"><v>0</v></c>'
                b'<c t="s"><v>1</v></c></row><row><c t="s"><v>2</v></c>'
                b'<c t="s"><v>3</v></c></row></sheetData></worksheet>',
            )
        self.assertEqual(
            server.extract_attachment_text(
                "report.xlsx",
                "application/vnd.openxmlformats",
                spreadsheet.getvalue(),
            ),
            "[Sheet 1]\n| Name | Value |\n| Orbit | 42 |",
        )

    async def test_audio_and_video_are_passed_as_langchain_multimodal_blocks(self):
        server.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        audio_stream = BytesIO()
        with wave.open(audio_stream, "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(8000)
            audio_file.writeframes(b"\x00\x00" * 800)
        audio_bytes = audio_stream.getvalue()
        (server.UPLOADS_DIR / "voice.wav").write_bytes(audio_bytes)
        audio_input = server.build_attachment_input(
            [
                {
                    "filename": "voice.wav",
                    "content_type": "audio/wav",
                    "storage_name": "voice.wav",
                    "extracted_text": "",
                }
            ],
            "Transcribe this clip.",
        )
        self.assertEqual(audio_input[1]["type"], "audio")
        self.assertEqual(audio_input[1]["mime_type"], "audio/wav")
        converted_audio = BytesIO(base64.b64decode(audio_input[1]["base64"]))
        with wave.open(converted_audio, "rb") as audio_file:
            self.assertEqual(audio_file.getframerate(), 16000)

        from imageio_ffmpeg import get_ffmpeg_exe

        video_path = server.UPLOADS_DIR / "clip.mp4"
        result = subprocess.run(
            [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x64:d=1:r=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-c:v",
                "mpeg4",
                "-c:a",
                "pcm_s16le",
                "-shortest",
                "-y",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        video_input = server.build_attachment_input(
            [
                {
                    "filename": video_path.name,
                    "content_type": "video/mp4",
                    "storage_name": video_path.name,
                    "extracted_text": "",
                }
            ],
            "Summarize this video.",
        )
        self.assertEqual(
            [block["type"] for block in video_input[1:]],
            ["text", "image", "audio"],
        )
        self.assertEqual(
            [block["mime_type"] for block in video_input[2:]],
            ["image/png", "audio/wav"],
        )


class TestCopilotKitAndA2UIEndpoints(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_store = server.chat_store
        self.original_agent = server.agent
        self.original_manager = server.provider_manager
        self.test_database = IsolatedPostgresDatabase()
        self.temp_dir = tempfile.TemporaryDirectory()
        server.chat_store = PostgresChatStore(self.test_database.database)
        self.test_user, self.thread = _make_isolated_user_and_thread(server.chat_store)

        class StubAgent:
            def get_thread_history(self, thread_id):
                return []

            def save_user_checkpoint(self, thread_id, message):
                pass

            def save_assistant_checkpoint(self, thread_id, message):
                pass

            async def astream_run(self, **kwargs):
                yield {"type": "provider_selected", "data": {"provider": "CopilotTest", "model": "gpt-4o"}}
                yield {"type": "token", "data": {"token": "Copilot response"}}
                yield {"type": "done", "data": {"content": "Copilot response completed"}}

        server.agent = StubAgent()
        server.provider_manager = RoundRobinProviderManager(
            env_dict={"API_KEY1": "test-key", "MODEL1": "test-model"}
        )

    async def asyncTearDown(self):
        server.chat_store = self.original_store
        server.agent = self.original_agent
        server.provider_manager = self.original_manager
        self.test_database.close()
        self.temp_dir.cleanup()

    async def test_canvases_crud_and_a2ui_action(self):
        # Create canvas
        canvas = await server.create_thread_canvas(
            self.thread["thread_id"],
            server.CanvasCreateRequest(
                title="System Telemetry",
                spec={"type": "card", "title": "System Telemetry", "badge": "Active"}
            ),
            current_user=self.test_user,
        )
        self.assertEqual(canvas["title"], "System Telemetry")
        self.assertIn("canvas_id", canvas)

        # List canvases
        canvases = await server.list_thread_canvases(
            self.thread["thread_id"], current_user=self.test_user
        )
        self.assertEqual(len(canvases), 1)
        self.assertEqual(canvases[0]["title"], "System Telemetry")

        # Trigger A2UI action
        action_resp = await server.handle_a2ui_action(
            self.thread["thread_id"],
            server.A2UIActionRequest(action_id="refresh_metrics", payload={"metric": "cpu"}),
            current_user=self.test_user,
        )
        chunks = []
        async for chunk in action_resp.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
        body = "".join(chunks)
        self.assertIn("Copilot response", body)

    async def test_copilotkit_endpoint(self):
        from starlette.requests import Request

        payload = json.dumps({
            "thread_id": self.thread["thread_id"],
            "messages": [{"role": "user", "content": "Help me with CopilotKit"}],
        }).encode("utf-8")

        events = iter([{"type": "http.request", "body": payload, "more_body": False}])
        async def receive():
            return next(events)

        request = Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/copilotkit",
                "raw_path": b"/copilotkit",
                "query_string": b"",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                ],
                "client": ("127.0.0.1", 1234),
                "server": ("test", 80),
            },
            receive,
        )

        response = await server.copilotkit_endpoint(request, current_user=self.test_user)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
        body = "".join(chunks)
        self.assertIn("Copilot response", body)

    async def test_generative_ui_and_a2ui_endpoints(self):
        from starlette.requests import Request

        # Direct HTML rendering via /generative_ui
        gen_payload = json.dumps({
            "html": "<button class='px-4 py-2 bg-blue-500 text-white rounded'>Click Me</button>",
            "title": "Button Widget"
        }).encode("utf-8")
        events = iter([{"type": "http.request", "body": gen_payload, "more_body": False}])
        async def receive_gen():
            return next(events)
        req_gen = Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/generative_ui",
                "raw_path": b"/generative_ui",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 1234),
                "server": ("test", 80),
            },
            receive_gen,
        )
        res_gen = await server.generative_ui_endpoint(req_gen, current_user=self.test_user)
        self.assertEqual(res_gen["status"], "generative_ui_rendered")
        self.assertIn("Click Me", res_gen["html"])

        # Direct spec submission via /a2ui
        a2ui_payload = json.dumps({
            "thread_id": self.thread["thread_id"],
            "title": "Direct Canvas",
            "spec": {"type": "metric", "label": "Uptime", "value": "99.9%"}
        }).encode("utf-8")
        events_a2 = iter([{"type": "http.request", "body": a2ui_payload, "more_body": False}])
        async def receive_a2():
            return next(events_a2)
        req_a2 = Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/a2ui",
                "raw_path": b"/a2ui",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 1234),
                "server": ("test", 80),
            },
            receive_a2,
        )
        res_a2 = await server.a2ui_endpoint(req_a2, current_user=self.test_user)
        self.assertEqual(res_a2["title"], "Direct Canvas")
        self.assertIn("canvas_id", res_a2)

    async def test_chronological_streaming_and_persistence(self):
        class MultiTurnStubAgent:
            def get_thread_history(self, thread_id):
                return []

            def save_user_checkpoint(self, thread_id, message):
                pass

            def save_assistant_checkpoint(self, thread_id, message):
                pass

            async def astream_run(self, **kwargs):
                yield {"type": "provider_selected", "data": {"provider": "TestProvider", "model": "test-model"}}
                # Turn 1 preamble
                yield {"type": "token", "data": {"token": "Writing file... "}}
                # Tool start
                yield {
                    "type": "tool_start",
                    "data": {
                        "tool": "write_workspace_file",
                        "args": {"file_path": "essay.md", "content": "Essay text"},
                        "tool_call_id": "call_123",
                    },
                }
                # Tool end
                yield {
                    "type": "tool_end",
                    "data": {
                        "tool": "write_workspace_file",
                        "output": json.dumps({"status": "written", "file_path": "essay.md"}),
                        "tool_call_id": "call_123",
                    },
                }
                # Turn 2 follow up
                yield {"type": "token", "data": {"token": "The essay has been saved."}}
                yield {
                    "type": "done",
                    "data": {
                        "content": "Writing file... \n\nThe essay has been saved.",
                    },
                }

        server.agent = MultiTurnStubAgent()
        response = await server.chat_stream_endpoint(
            server.ChatRequest(
                message="can you write essay.md",
                thread_id=self.thread["thread_id"],
            ),
            current_user=self.test_user,
        )
        events = []
        async for chunk in response.body_iterator:
            for line in chunk.split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

        event_types = [e["type"] for e in events]
        self.assertIn("tool_start", event_types)
        self.assertIn("tool_end", event_types)
        self.assertIn("workspace_file_changed", event_types)
        self.assertIn("done", event_types)

        tool_start_idx = event_types.index("tool_start")
        tool_end_idx = event_types.index("tool_end")
        done_idx = event_types.index("done")
        self.assertLess(tool_start_idx, tool_end_idx)
        self.assertLess(tool_end_idx, done_idx)

        # Check persistence in PostgreSQL
        thread = server.chat_store.get_thread(self.thread["thread_id"])
        assistant_msgs = [m for m in thread["messages"] if m["role"] == "assistant"]
        self.assertEqual(len(assistant_msgs), 1)
        self.assertEqual(assistant_msgs[0]["content"], "Writing file... \n\nThe essay has been saved.")


class TestMarketplaceEndpoints(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_store = server.chat_store
        self.test_database = IsolatedPostgresDatabase()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        server.chat_store = PostgresChatStore(self.test_database.database)
        self.test_user, _ = _make_isolated_user_and_thread(server.chat_store)

    async def asyncTearDown(self):
        server.chat_store = self.original_store
        self.test_database.close()
        self.temp_dir.cleanup()

    async def test_marketplace_catalog_starts_empty(self):
        catalog = await server.get_marketplace_catalog(current_user=self.test_user)
        self.assertIn("skills", catalog)
        self.assertIn("connectors", catalog)
        self.assertEqual(len(catalog["skills"]), 0)
        self.assertEqual(len(catalog["connectors"]), 0)
        self.assertIn("stats", catalog)
        self.assertEqual(catalog["stats"]["total_connectors"], 0)
        self.assertEqual(catalog["stats"]["total_skills"], 0)

    async def test_skill_readme_endpoint(self):
        # Register a custom skill with readme
        server.marketplace.register_custom_skill({
            "id": "my-custom-skill",
            "name": "My Custom Skill",
            "category": "Workflow & Standards",
            "description": "Custom skill for agent",
            "readme": "# My Custom Skill\nDetailed instructions.",
            "type": "skill",
        })
        readme_data = await server.get_skill_readme_endpoint(
            "my-custom-skill", current_user=self.test_user
        )
        self.assertEqual(readme_data["skill_id"], "my-custom-skill")
        self.assertIn("My Custom Skill", readme_data["readme"])
        server.marketplace.clear_all_skills()

    async def test_connector_configuration_and_toggle(self):
        # Register custom connector
        server.marketplace.register_connector({
            "id": "custom_github",
            "name": "Custom GitHub",
            "category": "Developer Tools",
            "description": "GitHub integration",
            "icon": "gitBranch",
            "configFields": [
                {"key": "api_token", "type": "password", "required": True},
                {"key": "default_owner", "type": "text", "required": False},
            ],
        })

        # Configure connector
        saved = await server.configure_connector_endpoint(
            "custom_github",
            server.ConnectorConfigRequest(
                enabled=True,
                config={"api_token": "ghp_secret_token_12345", "default_owner": "test-org"},
            ),
            current_user=self.test_user,
        )
        self.assertTrue(saved["enabled"])
        self.assertTrue(saved["is_configured"])

        # Fetch catalog to verify masking
        catalog = await server.get_marketplace_catalog(current_user=self.test_user)
        gh = next(c for c in catalog["connectors"] if c["id"] == "custom_github")
        self.assertTrue(gh["is_enabled"])
        self.assertTrue(gh["is_configured"])
        self.assertEqual(gh["status"], "active")
        self.assertTrue("••••" in gh["config"]["api_token"])
        self.assertEqual(gh["config"]["default_owner"], "test-org")

        # Toggle off
        toggled = await server.toggle_connector_endpoint(
            "custom_github",
            server.ConnectorToggleRequest(enabled=False),
            current_user=self.test_user,
        )
        self.assertFalse(toggled["enabled"])
        catalog = await server.get_marketplace_catalog(current_user=self.test_user)
        gh = next(c for c in catalog["connectors"] if c["id"] == "custom_github")
        self.assertFalse(gh["is_enabled"])

        # Clean up
        server.marketplace.clear_all_connectors()

    async def test_api_routes_do_not_return_html_fallback(self):
        with self.assertRaises(server.HTTPException) as ctx:
            await server.serve_frontend("api/unknown_endpoint")
        self.assertEqual(ctx.exception.status_code, 404)

        with self.assertRaises(server.HTTPException) as ctx:
            await server.serve_frontend("auth/unknown")
        self.assertEqual(ctx.exception.status_code, 404)

        with self.assertRaises(server.HTTPException) as ctx:
            await server.serve_frontend("marketplace/nonexistent")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
