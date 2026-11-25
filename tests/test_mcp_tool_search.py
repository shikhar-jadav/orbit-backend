import unittest
from mcp_tool_search import (
    get_cached_sanitized_schema,
    invalidate_mcp_tool_cache,
    rank_mcp_tools,
    validate_mcp_tool_args,
)


class RankMcpToolsTests(unittest.TestCase):
    def setUp(self):
        invalidate_mcp_tool_cache()
        self.tools = [
            {
                "raw_name": "create_issue",
                "connector_name": "GitHub",
                "description": "Create an issue in a repository",
                "params": {"properties": {"title": {"description": "Issue title"}}},
                "perm": "approval",
            },
            {
                "raw_name": "list_repositories",
                "connector_name": "GitHub",
                "description": "List repositories available to the user",
                "params": {"properties": {"visibility": {"description": "Repository visibility"}}},
                "perm": "allow",
            },
            {
                "raw_name": "search_messages",
                "connector_name": "Gmail",
                "description": "Search email messages",
                "params": {"properties": {"query": {"description": "Email search query"}}},
                "perm": "allow",
            },
            {
                "raw_name": "postChannelMessage",
                "connector_name": "Slack",
                "description": "Post a message to a Slack channel",
                "params": {"properties": {"channel": {"description": "Channel ID"}}},
                "perm": "approval",
            },
        ]

    def test_action_and_service_query(self):
        result = rank_mcp_tools("GitHub create issue", self.tools, service="GitHub")
        self.assertEqual(result[0]["raw_name"], "create_issue")
        self.assertTrue(all(item["connector_name"] == "GitHub" for item in result))

    def test_synonyms(self):
        self.assertEqual(
            rank_mcp_tools("GitHub create ticket", self.tools, service="GitHub")[0]["raw_name"],
            "create_issue",
        )

    def test_camelcase_and_fuzzy_matching(self):
        result = rank_mcp_tools("slack send message", self.tools, service="Slack")
        self.assertEqual(result[0]["raw_name"], "postChannelMessage")
        fuzzy = rank_mcp_tools("GitHub list repos", self.tools, service="GitHub")
        self.assertEqual(fuzzy[0]["raw_name"], "list_repositories")

    def test_service_filter(self):
        self.assertEqual(rank_mcp_tools("search email", self.tools, service="GitHub"), [])

    def test_parameter_descriptions(self):
        self.assertEqual(
            rank_mcp_tools("email search query", self.tools, service="Gmail")[0]["raw_name"],
            "search_messages",
        )

    def test_limit_is_bounded(self):
        self.assertLessEqual(len(rank_mcp_tools("GitHub", self.tools, limit=100)), 8)

    def test_schema_cache_and_invalidation(self):
        calls = 0

        def fake_sanitizer(schema):
            nonlocal calls
            calls += 1
            return dict(schema or {})

        s1 = get_cached_sanitized_schema("conn-1", "create_issue", {"type": "object"}, fake_sanitizer)
        s2 = get_cached_sanitized_schema("conn-1", "create_issue", {"type": "object"}, fake_sanitizer)
        self.assertEqual(calls, 1)
        self.assertEqual(s1, s2)

        # Schema change invalidates automatically
        get_cached_sanitized_schema("conn-1", "create_issue", {"type": "object", "required": ["title"]}, fake_sanitizer)
        self.assertEqual(calls, 2)

        # Explicit invalidation clears connector entries
        removed = invalidate_mcp_tool_cache("conn-1")
        self.assertEqual(removed, 1)
        get_cached_sanitized_schema("conn-1", "create_issue", {"type": "object", "required": ["title"]}, fake_sanitizer)
        self.assertEqual(calls, 3)

    def test_validate_mcp_tool_args(self):
        schema = {"type": "object", "required": ["owner", "repo", "title"]}
        self.assertIsNotNone(validate_mcp_tool_args("bad", schema))
        self.assertIn("repo", validate_mcp_tool_args({"owner": "a", "title": "x"}, schema))
        self.assertIsNone(validate_mcp_tool_args({"owner": "a", "repo": "b", "title": "x"}, schema))


if __name__ == "__main__":
    unittest.main()
