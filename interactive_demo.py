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

    print("CogniLLM Interactive Demo")
    print("=" * 40)

    # Initialize from environment variables
    try:
        cognillm = CogniLLM(
            profile_path="profiles/MadamChan.yaml",
            endpoint=ENDPOINT,
            deployment=DEPLOYMENT,
            api_key=API_KEY,
            api_version=API_VERSION,
        )
        print(f"Loaded: {cognillm.profile_name}")
        print("Type 'quit' to exit\n")

    except Exception as e:
        print(f"Error initializing: {e}")
        return

    # Simple chat loop
    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                break

            if not user_input:
                continue

            response, _ = cognillm.send_message(user_input)
            print(f"AI: {response['message']}")
            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
