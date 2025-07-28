#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.cognillm import CogniLLM

ENDPOINT: str = os.getenv("OPENAI_API_ENDPOINT")
DEPLOYMENT: str = os.getenv("OPENAI_API_DEPLOYMENT_NAME")
API_KEY: str = os.getenv("OPENAI_API_KEY")
API_VERSION: str = os.getenv("OPENAI_API_VERSION")


def main():
    if not ENDPOINT or not DEPLOYMENT or not API_KEY or not API_VERSION:
        raise ValueError("Missing environment variables")

    print("CogniLLM Interactive Demo with Summary Debug")
    print("=" * 50)

    # Initialize from environment variables with debug settings
    try:
        cognillm = CogniLLM(
            profile_path="profiles/MadamChan",
            endpoint=ENDPOINT,
            deployment=DEPLOYMENT,
            api_key=API_KEY,
            api_version=API_VERSION,
            summary_enabled=True,
            summary_window_size=2,  # Small window for demo
            summary_start_round=4,  # Debug: start summarizing after 10 rounds
        )
        print("✓ CogniLLM initialized with summary memory enabled")
        print("✓ Summary will start after 4 conversation rounds")
        print("✓ Each summary window contains 2 conversation rounds")
        print("Type 'quit' to exit, 'info' to see summary status\n")

    except Exception as e:
        print(f"Error initializing: {e}")
        return

    round_count = 0

    # Simple chat loop
    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                break

            if user_input.lower() == "info":
                summary_info = cognillm.get_summary_info()
                print("\n📊 Summary System Status:")
                print(f"  • Total conversation rounds: {summary_info['total_conversation_rounds']}")
                print(f"  • Summaries created: {summary_info['summaries_count']}")
                print(f"  • Rounds pending summary: {summary_info['conversation_rounds_pending']}")
                print(f"  • Window size: {summary_info['window_size']}")
                print(f"  • Start round threshold: {summary_info['start_round']}")
                print(f"  • Summary enabled: {summary_info['summary_enabled']}")
                print()
                continue

            if not user_input:
                continue

            round_count += 1
            print(f"\n🔄 Round {round_count}")
            
            # Show summary info before processing
            summary_info = cognillm.get_summary_info()
            print(f"📊 Before: {summary_info['total_conversation_rounds']} rounds, {summary_info['summaries_count']} summaries, {summary_info['conversation_rounds_pending']} pending")

            response, history = cognillm.send_message(user_input)
            
            # Show the AI response
            print(f"AI: {response.get('message', 'No message in response')}")
            print(f"💭 Tokens used: {history.get('tokens')}")
            print(f"🆔 Response ID: {history.get('id', 'N/A')}")
            
            # Show updated summary info after processing
            summary_info = cognillm.get_summary_info()
            print(f"📊 After: {summary_info['total_conversation_rounds']} rounds, {summary_info['summaries_count']} summaries, {summary_info['conversation_rounds_pending']} pending")
            
            # Show if we've reached the summary threshold
            if summary_info['total_conversation_rounds'] >= summary_info['start_round']:
                print("✅ Summary system is now active!")
                if summary_info['summaries_count'] > 0:
                    print(f"📝 {summary_info['summaries_count']} summary(ies) have been created")
            else:
                remaining = summary_info['start_round'] - summary_info['total_conversation_rounds']
                print(f"⏳ {remaining} more rounds until summary system activates")
            
            print("-" * 50)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
