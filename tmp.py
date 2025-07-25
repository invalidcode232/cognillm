from src.lib.memory.summary import SummaryBasedMemory
from dotenv import load_dotenv
import os
import sys

load_dotenv()

ENDPOINT: str = os.getenv("OPENAI_API_ENDPOINT")
DEPLOYMENT: str = os.getenv("OPENAI_API_DEPLOYMENT_NAME")
API_KEY: str = os.getenv("OPENAI_API_KEY")
API_VERSION: str = os.getenv("OPENAI_API_VERSION")

def main():
    # Write profile path using os
    # The path is "cognillm/profiles/MadamChan/profile.yaml"
    # Current working directory is "cognillm/src"
    profile_path = os.path.join(os.path.dirname(__file__), "profiles", "MadamChan")

    memory_test = SummaryBasedMemory(
        profile_path=profile_path,
        tool_name="openai",
        endpoint=ENDPOINT,
        deployment=DEPLOYMENT,
        api_key=API_KEY,
        api_version=API_VERSION
    )

    while True:
        user_input = input("Enter a message (or 'quit' to exit): ")

        memory_test.update(history_item=user_input)

        print("="*20)
        print("Current History:")
        for item in memory_test.history_list:
            print(f"- {item}")
        print("="*20)
        print("Current Summary:")
        for summary in memory_test.summary_list:
            print(f"- {summary}")
        print("="*20)


if __name__ == "__main__":
    main()