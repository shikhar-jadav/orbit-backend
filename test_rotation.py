import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from langchain_core.tools import tool
from config_manager import RoundRobinProviderManager, ProviderConfig
from tools import WORKSPACE_AGENT_TOOLS, ask_question, search_available_tools
from agent import ReActAgent


@tool
def calculate(expression: str) -> str:
    """Calculate a math expression for testing."""
    return str(eval(expression, {"__builtins__": None}, {}))



class TestRoundRobinAndAgent(unittest.TestCase):
    def setUp(self):
        # Set up mock environment variables with 3 distinct providers
        self.mock_env = {
            "API_KEY1": "sk-test-openai-12345",
            "BASE_URL1": "https://api.openai.com/v1",
            "MODEL1": "gpt-4o-mini",
            "MODEL_LIST1": "gpt-4o-mini,gpt-4o",
            "PROVIDER_NAME1": "OpenAI-Account",

            "API_KEY2": "gsk-test-groq-67890",
            "BASE_URL2": "https://api.groq.com/openai/v1",
            "MODEL2": "llama-3.3-70b-versatile",
            "MODEL_LIST2": "llama-3.3-70b-versatile,llama-3.1-8b-instant",
            "PROVIDER_NAME2": "Groq-Account",

            "API_KEY3": "sk-or-test-openrouter-99999",
            "BASE_URL3": "https://openrouter.ai/api/v1",
            "MODEL3": "deepseek/deepseek-chat",
            "MODEL_LIST3": "deepseek/deepseek-chat",
            "PROVIDER_NAME3": "OpenRouter-Account",
        }

    def test_provider_discovery_and_metadata(self):
        manager = RoundRobinProviderManager(env_dict=self.mock_env)
        self.assertEqual(manager.get_provider_count(), 3)

        # Verify Provider 1
        p1 = manager.providers[0]
        self.assertEqual(p1.name, "OpenAI-Account")
        self.assertEqual(p1.api_key, "sk-test-openai-12345")
        self.assertEqual(p1.base_url, "https://api.openai.com/v1")
        self.assertEqual(p1.default_model, "gpt-4o-mini")
        self.assertIn("gpt-4o", p1.model_list)

        # Verify Provider 2
        p2 = manager.providers[1]
        self.assertEqual(p2.name, "Groq-Account")
        self.assertEqual(p2.api_key, "gsk-test-groq-67890")
        self.assertEqual(p2.base_url, "https://api.groq.com/openai/v1")
        self.assertEqual(p2.default_model, "llama-3.3-70b-versatile")
        self.assertIn("llama-3.1-8b-instant", p2.model_list)

        # Verify Provider 3
        p3 = manager.providers[2]
        self.assertEqual(p3.name, "OpenRouter-Account")
        self.assertEqual(p3.api_key, "sk-or-test-openrouter-99999")
        self.assertEqual(p3.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(p3.default_model, "deepseek/deepseek-chat")

    def test_round_robin_rotation_and_wraparound(self):
        manager = RoundRobinProviderManager(env_dict=self.mock_env)

        # Step 1: Provider 1
        llm, provider, req_num = manager.get_next_llm()
        self.assertEqual(provider.name, "OpenAI-Account")
        self.assertEqual(provider.api_key, "sk-test-openai-12345")
        self.assertEqual(provider.default_model, "gpt-4o-mini")
        self.assertEqual(req_num, 1)

        # Step 2: Provider 2
        llm, provider, req_num = manager.get_next_llm()
        self.assertEqual(provider.name, "Groq-Account")
        self.assertEqual(provider.api_key, "gsk-test-groq-67890")
        self.assertEqual(provider.base_url, "https://api.groq.com/openai/v1")
        self.assertEqual(provider.default_model, "llama-3.3-70b-versatile")
        self.assertEqual(req_num, 2)

        # Step 3: Provider 3
        llm, provider, req_num = manager.get_next_llm()
        self.assertEqual(provider.name, "OpenRouter-Account")
        self.assertEqual(provider.api_key, "sk-or-test-openrouter-99999")
        self.assertEqual(provider.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(provider.default_model, "deepseek/deepseek-chat")
        self.assertEqual(req_num, 3)

        # Step 4: Wraps around to Provider 1!
        llm, provider, req_num = manager.get_next_llm()
        self.assertEqual(provider.name, "OpenAI-Account")
        self.assertEqual(provider.api_key, "sk-test-openai-12345")
        self.assertEqual(req_num, 4)

        # Step 5: Provider 2
        llm, provider, req_num = manager.get_next_llm()
        self.assertEqual(provider.name, "Groq-Account")
        self.assertEqual(req_num, 5)

    def test_tools_functionality(self):
        # Math test tool
        calc_res = calculate.invoke({"expression": "10 * 5 + 4"})
        self.assertEqual(calc_res, "54")

        # System catalog tools
        search_res = search_available_tools.invoke({"query": "memory"})
        self.assertIn("search_tools", search_res)

        ask_res = ask_question.invoke({"questions": [{"question": "Proceed?", "options": ["yes", "no"]}]})
        self.assertIn("question_asked", ask_res)

    def test_react_agent_execution_with_mocked_llm(self):
        manager = RoundRobinProviderManager(env_dict=self.mock_env)
        agent = ReActAgent(provider_manager=manager, tools=[calculate])

        # Mock LLM sequence:
        # Step 1: Model node runs (Request #1) -> Decides to call "calculate" tool
        # Step 2: Tool node executes -> returns ToolMessage
        # Step 3: Model node runs again (Request #2) -> Returns final answer
        tool_call_msg = AIMessage(
            content="Let me calculate this for you.",
            tool_calls=[{
                "name": "calculate",
                "args": {"expression": "25 * 4"},
                "id": "call_12345",
                "type": "tool_call",
            }]
        )
        final_answer_msg = AIMessage(
            content="25 * 4 is 100."
        )

        mock_llm_responses = [tool_call_msg, final_answer_msg]

        with patch.object(ReActAgent, "_call_llm", side_effect=mock_llm_responses) as mock_call:
            res = agent.graph.invoke({"messages": [HumanMessage(content="What is 25 * 4?")]})

            # Check that _call_llm was called twice (initial ReAct step + follow-up step)
            self.assertEqual(mock_call.call_count, 2)

            # Check final message is final_answer_msg
            final_message = res["messages"][-1]
            self.assertEqual(final_message.content, "25 * 4 is 100.")

            # Check that tool execution took place in between
            tool_messages = [m for m in res["messages"] if isinstance(m, ToolMessage)]
            self.assertEqual(len(tool_messages), 1)
            self.assertEqual(tool_messages[0].content, "100")


if __name__ == "__main__":
    unittest.main()
