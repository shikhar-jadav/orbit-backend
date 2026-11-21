import sys

# Reconfigure stdout/stderr for Unicode safety on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config_manager import RoundRobinProviderManager
from agent import ReActAgent
from tools import AGENT_TOOLS


def main():
    print("\n" + "=" * 75)
    print("🤖 LangGraph ReAct Agent with Multi-Provider API Key Round-Robin")
    print("=" * 75)

    try:
        # 1. Initialize configuration manager from .env
        manager = RoundRobinProviderManager()
        manager.print_summary()

        # 2. Build ReAct agent
        agent = ReActAgent(
            provider_manager=manager,
            tools=AGENT_TOOLS,
            enable_failover=True,
        )

    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")
        print("Please check your .env file and ensure API_KEY1, API_KEY2, etc. are properly set.")
        sys.exit(1)

    print("\nSelect Mode:")
    print("1. Run automated demo queries to observe round-robin rotation")
    print("2. Interactive chat mode (type your own queries)")
    print("3. Exit")

    choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"

    if choice == "1":
        demo_queries = [
            "What is 125 * 38 + sqrt(144)? Use the calculator tool.",
            "What is the current UTC time?",
            "What is the weather like in Tokyo right now?",
            "Calculate (50 * 4) / 5 and tell me the result.",
        ]

        print("\n🚀 Starting automated demonstration queries across round-robin endpoints...\n")
        for i, q in enumerate(demo_queries, start=1):
            print(f"\n{'#'*75}")
            print(f"🔹 Execution Step {i}/{len(demo_queries)}")
            print(f"{'#'*75}")
            try:
                agent.run(q)
            except Exception as e:
                print(f"Execution failed: {e}")

    elif choice == "2":
        print("\n💬 Entering interactive chat mode. Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break
                agent.run(user_input)
            except KeyboardInterrupt:
                print("\nSession interrupted. Exiting.")
                break
            except Exception as e:
                print(f"Error executing query: {e}")


if __name__ == "__main__":
    main()
