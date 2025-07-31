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

SUMMARY_WINDOW_SIZE: int = 2
SUMMARY_START_ROUND: int = 4

PROFILE_PATH: str = "profiles/MadamChan"


def main():
    if not ENDPOINT or not DEPLOYMENT or not API_KEY or not API_VERSION:
        raise ValueError("Missing environment variables")

    # Initialize from environment variables with debug settings
    try:
        cognillm = CogniLLM(
            profile_path=PROFILE_PATH,
            endpoint=ENDPOINT,
            deployment=DEPLOYMENT,
            api_key=API_KEY,
            api_version=API_VERSION,
            summary_enabled=True,
            summary_window_size=SUMMARY_WINDOW_SIZE,
            summary_start_round=SUMMARY_START_ROUND,
        )
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    print(f"Using profile: {PROFILE_PATH}")
    print("Type 'quit' to exit, 'info' to see summary status\n")
    round_count = 0

    # Simple chat loop
    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                break

            if user_input.lower() == "info":
                summary_info = cognillm.get_summary_info()
                print("\nSummary system status:")
                print(
                    f"Total conversation rounds: {summary_info['total_conversation_rounds']}"
                )
                print(f"Summaries created: {summary_info['summaries_count']}")
                print(
                    f"Rounds pending summary: {summary_info['conversation_rounds_pending']}"
                )
                print(f"Window size: {summary_info['window_size']}")
                print(f"Start round threshold: {summary_info['start_round']}")
                print(f"Summary enabled: {summary_info['summary_enabled']}")
                continue

            if user_input.lower() == "history":
                history = cognillm.get_conversation_history()
                print("\nConversation history:")
                for message in history:
                    print(f"{message['role']}: {message['content']}")
                continue

            if not user_input:
                continue

            print(f"\nAttempting round {round_count + 1}")

            # Show summary info before processing
            summary_info = cognillm.get_summary_info()
            print(
                f"Before: {summary_info['total_conversation_rounds']} rounds, {summary_info['summaries_count']} summaries, {summary_info['conversation_rounds_pending']} pending"
            )

            if cognillm.stage_manager.current_stage.is_final_stage:
                print("Conversation has already ended")
                break

            try:
                message, tokens_used = cognillm.send_message(user_input)

                # Only increment round count if message was successfully processed
                round_count += 1

                # Show the AI response
                print(f"AI: {message if message else 'No message in response'}")
                print(f"Tokens used: {tokens_used if tokens_used else 'N/A'}")

                # Show updated summary info after processing
                summary_info = cognillm.get_summary_info()
                print(
                    f"After: {summary_info['total_conversation_rounds']} rounds, {summary_info['summaries_count']} summaries, {summary_info['conversation_rounds_pending']} pending"
                )

                # Show if we've reached the summary threshold
                if (
                    summary_info["total_conversation_rounds"]
                    >= summary_info["start_round"]
                ):
                    if summary_info["summaries_count"] > 0:
                        print(
                            f"{summary_info['summaries_count']} summaries have been created"
                        )
                else:
                    remaining = (
                        summary_info["start_round"]
                        - summary_info["total_conversation_rounds"]
                    )
                    print(f"{remaining} more rounds until summary system activates.")
            except Exception as e:
                print(f"error in round {round_count + 1}: {e}")

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    main()
