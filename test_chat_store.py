import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres import PostgresSaver

from agent import ReActAgent
from chat_store import PostgresChatStore
from config_manager import RoundRobinProviderManager
from psycopg.rows import dict_row
from test_database import IsolatedPostgresDatabase


class TestPostgresChatStore(unittest.TestCase):
    def setUp(self):
        self.test_database = IsolatedPostgresDatabase()
        self.store = PostgresChatStore(self.test_database.database)
        self.connections = []

    def tearDown(self):
        for connection in self.connections:
            connection.close()
        self.test_database.close()

    def test_persists_threads_messages_and_settings(self):
        thread = self.store.create_thread()
        self.assertEqual(len(thread["thread_id"]), 12)

        self.store.add_message(thread["thread_id"], "user", "  Remember   this chat ")
        self.store.add_message(thread["thread_id"], "assistant", "Saved.")

        loaded = self.store.get_thread(thread["thread_id"])
        self.assertTrue(self.store.thread_exists(thread["thread_id"]))
        self.assertFalse(
            self.store.thread_exists(thread["thread_id"], "another-user")
        )
        self.assertEqual(loaded["thread"]["title"], "Remember this chat")
        self.assertEqual(
            [message["role"] for message in loaded["messages"]],
            ["user", "assistant"],
        )
        self.assertEqual(self.store.list_threads()[0]["message_count"], 2)

        self.assertTrue(self.store.rename_thread(thread["thread_id"], "Project notes"))
        self.assertEqual(
            self.store.get_thread(thread["thread_id"])["thread"]["title"],
            "Project notes",
        )

        self.store.save_settings("gpt-test", 2)
        self.assertEqual(
            self.store.get_settings(),
            {"model": "gpt-test", "provider_index": 2},
        )

        self.assertTrue(self.store.delete_thread(thread["thread_id"]))
        self.assertIsNone(self.store.get_thread(thread["thread_id"]))

    def test_turn_versions_can_be_selected_and_branched(self):
        thread = self.store.create_thread()
        user = self.store.add_message(thread["thread_id"], "user", "First prompt")
        assistant = self.store.add_message(thread["thread_id"], "assistant", "First answer")
        self.store.add_turn_version(
            user["id"], assistant["id"], "First prompt", "First answer"
        )
        self.store.add_message(thread["thread_id"], "user", "Follow-up")
        self.store.add_message(thread["thread_id"], "assistant", "Follow-up answer")

        updated = self.store.save_turn_version(
            thread["thread_id"], user["id"], "Edited prompt", "Alternate answer"
        )
        self.assertEqual(updated, {"version": 2, "version_count": 2})
        loaded = self.store.get_thread(thread["thread_id"])
        self.assertEqual(
            [message["content"] for message in loaded["messages"]],
            ["Edited prompt", "Alternate answer"],
        )
        self.assertEqual(loaded["messages"][0]["selected_version"], 2)
        self.assertEqual(len(loaded["messages"][1]["variants"]), 2)

        self.assertTrue(
            self.store.select_turn_version(thread["thread_id"], user["id"], 1)
        )
        branch = self.store.branch_thread(thread["thread_id"], user["id"], 2)
        self.assertIsNotNone(branch)
        source_thread = self.store.get_thread(thread["thread_id"])
        self.assertEqual(branch["thread"]["parent_thread_id"], thread["thread_id"])
        self.assertEqual(
            branch["thread"]["parent_title"], source_thread["thread"]["title"]
        )
        self.assertTrue(branch["thread"]["title"].startswith("Branch · "))
        self.assertEqual(
            branch["thread"]["branch_from_message_id"],
            branch["messages"][1]["id"],
        )
        self.assertEqual(
            branch["thread"]["branch_label"],
            "Alternate answer",
        )
        listed_branch = next(
            item
            for item in self.store.list_threads()
            if item["thread_id"] == branch["thread"]["thread_id"]
        )
        self.assertEqual(listed_branch["parent_thread_id"], thread["thread_id"])
        self.assertEqual(
            listed_branch["parent_title"], source_thread["thread"]["title"]
        )
        self.assertEqual(branch["messages"][0]["content"], "Edited prompt")
        self.assertEqual(branch["messages"][1]["content"], "Alternate answer")
        self.assertEqual(len(branch["messages"][1]["variants"]), 2)

    def test_attachments_are_persisted_and_copied_with_branches(self):
        thread = self.store.create_thread()
        user = self.store.add_message(thread["thread_id"], "user", "Review this file")
        self.store.add_attachment(
            thread["thread_id"],
            "attachment-one",
            "notes.txt",
            "text/plain",
            "attachment-one.txt",
            12,
            "A short note",
        )
        self.store.attach_to_message(
            thread["thread_id"], user["id"], ["attachment-one"]
        )
        assistant = self.store.add_message(
            thread["thread_id"], "assistant", "The note says hello."
        )
        self.store.add_turn_version(
            user["id"], assistant["id"], "Review this file", "The note says hello."
        )

        loaded = self.store.get_thread(thread["thread_id"])
        self.assertEqual(loaded["messages"][0]["attachments"][0]["filename"], "notes.txt")

        branch = self.store.branch_thread(thread["thread_id"], user["id"], 1)
        self.assertIsNotNone(branch)
        self.assertEqual(
            branch["messages"][0]["attachments"][0]["filename"], "notes.txt"
        )
        self.assertNotEqual(
            branch["messages"][0]["attachments"][0]["attachment_id"],
            "attachment-one",
        )

    def test_failed_edit_keeps_new_prompt_and_marks_version_retryable(self):
        thread = self.store.create_thread()
        user = self.store.add_message(thread["thread_id"], "user", "Original prompt")
        assistant = self.store.add_message(
            thread["thread_id"], "assistant", "Original answer"
        )
        self.store.add_turn_version(
            user["id"], assistant["id"], "Original prompt", "Original answer"
        )
        self.store.add_message(thread["thread_id"], "user", "Later prompt")
        self.store.add_message(thread["thread_id"], "assistant", "Later answer")

        pending = self.store.prepare_turn_version(
            thread["thread_id"], user["id"], "Edited prompt"
        )
        self.assertEqual(pending, {"version": 2, "version_count": 2})
        self.store.fail_turn_version(
            thread["thread_id"], user["id"], pending["version"]
        )

        loaded = self.store.get_thread(thread["thread_id"])
        self.assertEqual(
            [message["content"] for message in loaded["messages"]],
            ["Edited prompt", ""],
        )
        self.assertTrue(loaded["messages"][1]["failed"])
        self.assertTrue(loaded["messages"][1]["variants"][1]["failed"])

        self.store.complete_turn_version(
            thread["thread_id"], user["id"], pending["version"], "Retry succeeded"
        )
        loaded = self.store.get_thread(thread["thread_id"])
        self.assertEqual(loaded["messages"][1]["content"], "Retry succeeded")
        self.assertFalse(loaded["messages"][1]["failed"])

    def test_langgraph_checkpoints_restore_thread_memory(self):
        manager = RoundRobinProviderManager(
            env_dict={"API_KEY1": "test-key", "MODEL1": "test-model"}
        )
        connection = self.test_database.database.connect(
            autocommit=True, row_factory=dict_row
        )
        self.connections.append(connection)
        saver = PostgresSaver(connection)
        saver.setup()
        agent = ReActAgent(manager, checkpointer=saver)

        agent.save_user_checkpoint("thread-1", "My name is Sam.")
        agent.save_assistant_checkpoint("thread-1", "Nice to meet you, Sam.")
        with patch.object(agent, "_call_llm", return_value=AIMessage(content="Ready.")):
            self.assertEqual(agent.run("Continue.", thread_id="thread-2")["response"], "Ready.")

        restored_connection = self.test_database.database.connect(
            autocommit=True, row_factory=dict_row
        )
        self.connections.append(restored_connection)
        restored_agent = ReActAgent(
            manager,
            checkpointer=PostgresSaver(restored_connection),
        )
        self.assertEqual(
            restored_agent.get_thread_history("thread-1"),
            [
                {"role": "user", "content": "My name is Sam."},
                {"role": "assistant", "content": "Nice to meet you, Sam."},
            ],
        )
        restored_agent.replace_thread_history(
            "thread-1",
            [
                {"role": "user", "content": "My name is Alex."},
                {"role": "assistant", "content": "Nice to meet you, Alex."},
            ],
        )
        self.assertEqual(
            restored_agent.get_thread_history("thread-1"),
            [
                {"role": "user", "content": "My name is Alex."},
                {"role": "assistant", "content": "Nice to meet you, Alex."},
            ],
        )

    def test_tools_and_parts_persistence(self):
        thread = self.store.create_thread()
        tid = thread["thread_id"]
        tools_sample = [
            {
                "id": "tool-1",
                "name": "write_workspace_file",
                "args": {"file_path": "essay.md"},
                "output": '{"status": "written", "file_path": "essay.md"}',
                "status": "done",
            }
        ]
        parts_sample = [
            {"type": "tool", "tool": tools_sample[0]},
            {"type": "text", "text": "The essay has been written."},
        ]
        user = self.store.add_message(tid, "user", "write essay")
        assistant = self.store.add_message(
            tid,
            "assistant",
            "The essay has been written.",
            tools=tools_sample,
            parts=parts_sample,
        )
        self.store.add_turn_version(
            user["id"],
            assistant["id"],
            "write essay",
            "The essay has been written.",
            tools=tools_sample,
            parts=parts_sample,
        )

        # Create fresh store pointing to same database file to simulate browser reload / restart
        reloaded_store = PostgresChatStore(self.test_database.database)
        loaded = reloaded_store.get_thread(tid)
        self.assertIsNotNone(loaded)
        loaded_assistant = next(m for m in loaded["messages"] if m["role"] == "assistant")
        self.assertEqual(len(loaded_assistant["tools"]), 1)
        self.assertEqual(loaded_assistant["tools"][0]["name"], "write_workspace_file")
        self.assertEqual(loaded_assistant["tools"][0]["args"]["file_path"], "essay.md")
        self.assertEqual(len(loaded_assistant["parts"]), 2)
        self.assertEqual(loaded_assistant["parts"][0]["type"], "tool")
        self.assertEqual(loaded_assistant["parts"][1]["type"], "text")
        self.assertEqual(loaded_assistant["parts"][1]["text"], "The essay has been written.")

        # Test turn version contains tools and parts
        self.assertEqual(len(loaded_assistant["variants"]), 1)
        self.assertEqual(len(loaded_assistant["variants"][0]["tools"]), 1)
        self.assertEqual(len(loaded_assistant["variants"][0]["parts"]), 2)

    def test_historical_assistant_message_fallback_synthesizes_tools(self):
        thread = self.store.create_thread()
        tid = thread["thread_id"]
        # Create workspace file for this thread
        ws_dir = Path("data/workspaces") / tid
        ws_dir.mkdir(parents=True, exist_ok=True)
        try:
            (ws_dir / "essay.md").write_text("# AI Essay\nContent", encoding="utf-8")

            # Insert historical message directly with NULL tools_json and parts_json
            self.store.add_message(tid, "user", "create essay.md")
            self.store.add_message(
                tid,
                "assistant",
                "The essay on Artificial Intelligence has been successfully saved to essay.md.",
            )

            # get_thread should detect essay.md in the message and workspace, and synthesize tools and parts
            with patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("history load read workspace file"),
            ):
                loaded = self.store.get_thread(tid)
            loaded_assistant = next(m for m in loaded["messages"] if m["role"] == "assistant")
            self.assertEqual(len(loaded_assistant["tools"]), 1)
            self.assertEqual(loaded_assistant["tools"][0]["name"], "write_workspace_file")
            self.assertEqual(loaded_assistant["tools"][0]["args"]["file_path"], "essay.md")
            synthesized_output = json.loads(loaded_assistant["tools"][0]["output"])
            self.assertEqual(
                synthesized_output["size_bytes"],
                len("# AI Essay\nContent".encode("utf-8")),
            )
            self.assertEqual(len(loaded_assistant["parts"]), 2)
            self.assertEqual(loaded_assistant["parts"][0]["type"], "tool")
            self.assertEqual(loaded_assistant["parts"][1]["type"], "text")
        finally:
            import shutil
            shutil.rmtree(ws_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
