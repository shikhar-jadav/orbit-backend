import unittest
from mcp_tool_search import rank_mcp_tools

class RankMcpToolsTests(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {"raw_name": "create_issue", "connector_name": "GitHub",
             "description": "Create an issue in a repository",
             "params": {"properties": {"title": {"description": "Issue title"}}},
             "perm": "approval"},
            {"raw_name": "list_repositories", "connector_name": "GitHub",
             "description": "List repositories available to the user",
             "params": {"properties": {"visibility": {"description": "Repository visibility"}}},
             "perm": "allow"},
            {"raw_name": "search_messages", "connector_name": "Gmail",
             "description": "Search email messages",
             "params": {"properties": {"query": {"description": "Email search query"}}},
             "perm": "allow"},
        ]

    def test_action_and_service_query(self):
        result = rank_mcp_tools("GitHub create issue", self.tools, service="GitHub")
        self.assertEqual(result[0]["raw_name"], "create_issue")
        self.assertTrue(all(item["connector_name"] == "GitHub" for item in result))

    def test_synonyms(self):
        self.assertEqual(rank_mcp_tools("GitHub create ticket", self.tools, service="GitHub")[0]["raw_name"], "create_issue")

    def test_service_filter(self):
        self.assertEqual(rank_mcp_tools("search email", self.tools, service="GitHub"), [])

    def test_parameter_descriptions(self):
        self.assertEqual(rank_mcp_tools("email search query", self.tools, service="Gmail")[0]["raw_name"], "search_messages")

    def test_limit_is_bounded(self):
        self.assertLessEqual(len(rank_mcp_tools("GitHub", self.tools, limit=100)), 8)

if __name__ == "__main__":
    unittest.main()
