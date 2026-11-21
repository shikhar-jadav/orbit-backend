import unittest
import asyncio
from typing import Dict, Any

from memory_store import MemoryStoreManager, create_memory_store
from test_database import IsolatedPostgresDatabase
import tools
from agent import ReActAgent
from config_manager import RoundRobinProviderManager, ProviderConfig
from event_streaming import (
    stream_messages,
    stream_values,
    stream_subgraphs,
    stream_raw_events,
    format_sse_event,
    StreamEventAdapter,
)


class TestMemoryStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_database = IsolatedPostgresDatabase()
        self.store = create_memory_store(self.test_database.database)

    def tearDown(self):
        self.test_database.close()

    def test_sync_crud(self):
        ns = ("user_1", "memories")
        self.store.put(ns, "k1", {"fact": "likes coffee"})
        val = self.store.get(ns, "k1")
        self.assertIsNotNone(val)
        self.assertEqual(val.get("fact"), "likes coffee")

        # Search by prefix
        results = self.store.search(("user_1",))
        self.assertTrue(any(r.key == "k1" for r in results))

        # Delete
        self.store.delete(ns, "k1")
        self.assertIsNone(self.store.get(ns, "k1"))

    def test_postgres_persistence_across_instances(self):
        ns = ("user_2", "preferences")
        self.store.put(ns, "theme", {"value": "dark"})

        store2 = create_memory_store(self.test_database.database)
        val = store2.get(ns, "theme")
        self.assertIsNotNone(val)
        self.assertEqual(val.get("value"), "dark")

    async def test_async_methods(self):
        ns = ("user_async", "memories")
        await self.store.aput(ns, "a1", {"data": "async value"})
        val = await self.store.aget(ns, "a1")
        self.assertIsNotNone(val)
        self.assertEqual(val.get("data"), "async value")

        items = await self.store.asearch(("user_async",))
        self.assertEqual(len(items), 1)

        await self.store.adelete(ns, "a1")
        self.assertIsNone(await self.store.aget(ns, "a1"))

    def test_convenience_helpers(self):
        user_id = "alice"
        key = self.store.save_user_memory(user_id, {"food": "sushi"})
        memories = self.store.get_user_memories(user_id)
        self.assertTrue(any(m["key"] == key and m["value"]["food"] == "sushi" for m in memories))

        self.store.save_user_preference(user_id, "editor", "vscode")
        prefs = self.store.get_user_preferences(user_id)
        self.assertIn("editor", prefs)
        self.assertEqual(prefs["editor"]["value"], "vscode")


class TestMemoryTools(unittest.TestCase):
    def setUp(self):
        from memory_store import active_user_id_var, active_memory_store_var
        self.test_database = IsolatedPostgresDatabase()
        self.store = create_memory_store(self.test_database.database)
        self.token_user = active_user_id_var.set("user_test")
        self.token_store = active_memory_store_var.set(self.store)

    def tearDown(self):
        from memory_store import active_user_id_var, active_memory_store_var
        active_user_id_var.reset(self.token_user)
        active_memory_store_var.reset(self.token_store)
        self.test_database.close()

    def test_manage_memory_saves_and_searches(self):
        import json
        names = {agent_tool.name for agent_tool in tools.WORKSPACE_AGENT_TOOLS}
        self.assertIn("manage_memory", names)
        self.assertNotIn("save_memory", names)
        self.assertNotIn("search_memory", names)
        self.assertFalse(hasattr(tools, "save_memory"))
        self.assertFalse(hasattr(tools, "search_memory"))

        res_save = tools.manage_memory.invoke({
            "action": "save",
            "fact": "User is a senior python developer",
        })
        data_save = json.loads(res_save)
        self.assertEqual(data_save["status"], "memory_saved")
        self.assertIn("key", data_save)

        res_search = tools.manage_memory.invoke({"action": "search", "query": "python"})
        data_search = json.loads(res_search)
        self.assertEqual(data_search["status"], "memories_retrieved")
        self.assertGreaterEqual(data_search["count"], 1)

        missing_fact = json.loads(tools.manage_memory.invoke({"action": "save"}))
        self.assertEqual(missing_fact["status"], "error")


class TestEventStreamingHelpers(unittest.IsolatedAsyncioTestCase):
    def test_format_sse_event(self):
        sse = format_sse_event("token", {"token": "hello"})
        self.assertTrue(sse.startswith("event: token\n"))
        self.assertIn('"token": "hello"', sse)
        self.assertTrue(sse.endswith("\n\n"))

    async def test_stream_adapter(self):
        mock_raw_events = [
            {
                "method": "messages",
                "params": {
                    "data": [
                        {
                            "event": "content-block-delta",
                            "delta": {"type": "text-delta", "text": "Hello "},
                        }
                    ]
                },
            },
            {
                "method": "messages",
                "params": {
                    "data": [
                        {
                            "event": "content-block-delta",
                            "delta": {"type": "text-delta", "text": "world!"},
                        }
                    ]
                },
            },
            {
                "method": "tools",
                "params": {
                    "data": {
                        "event": "tool-started",
                        "tool_name": "calculate",
                        "input": {"expression": "2+2"},
                        "tool_call_id": "c1",
                    }
                },
            },
            {
                "method": "tools",
                "params": {
                    "data": {
                        "event": "tool-finished",
                        "tool_name": "calculate",
                        "output": "4",
                        "tool_call_id": "c1",
                    }
                },
            },
        ]
        adapter = StreamEventAdapter(mock_raw_events)
        events = [e async for e in adapter.to_orbit_events()]
        self.assertTrue(any(e["type"] == "token" and e["data"]["token"] == "Hello " for e in events))
        self.assertTrue(any(e["type"] == "tool_start" and e["data"]["tool"] == "calculate" for e in events))
        self.assertTrue(any(e["type"] == "tool_end" and e["data"]["output"] == "4" for e in events))
        self.assertTrue(any(e["type"] == "done" and "Hello world!" in e["data"]["content"] for e in events))


class TestMemoryStoreStandalone(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from memory_store import create_memory_store
        self.store = create_memory_store(database=None)

    def test_sync_crud(self):
        ns = ("user_unit", "memories")
        self.store.put(ns, "m1", {"fact": "prefers dark mode"})
        val = self.store.get(ns, "m1")
        self.assertEqual(val.get("fact"), "prefers dark mode")

        results = self.store.search(ns, query="dark")
        self.assertEqual(len(results), 1)

        self.store.delete(ns, "m1")
        self.assertIsNone(self.store.get(ns, "m1"))

    async def test_async_crud(self):
        ns = ("user_unit_async", "notes")
        await self.store.aput(ns, "n1", {"note": "async unit test"})
        val = await self.store.aget(ns, "n1")
        self.assertEqual(val.get("note"), "async unit test")

        results = await self.store.asearch(ns, query="unit")
        self.assertEqual(len(results), 1)

        await self.store.adelete(ns, "n1")
        self.assertIsNone(await self.store.aget(ns, "n1"))


if __name__ == "__main__":
    unittest.main()
