import os
import json

PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "include", "prompts", "cognillm.txt"
)


class PromptManager:
    def __init__(
        self, profile_name: str, profile_cognitive_model: dict[str, str]
    ) -> str:
        """
        Retrieves the main CogniLLM prompt and replaces dynamic variables with the provided values.

        Args:
            profile_name (str): The name of the character/profile being roleplayed.
            profile_cognitive_model (dict[str, str]): The cognitive model of the profile.
        """
        with open(PROMPT_PATH, "r") as file:
            contents = file.read()

        self.base_prompt: str = contents.replace("%name%", profile_name).replace(
            "%profile%", json.dumps(profile_cognitive_model)
        )

    def get_base_prompt(self) -> str:
        """
        Gets the baseline prompt with the dynamic variables replaced.

        Returns:
            str: The formatted prompt with dynamic variables replaced.

        Example:
            >>> prompt = PromptManager.get_prompt()
            >>> print(prompt)
        """
        return self.base_prompt

    def get_prompt(self, user_message: str) -> str:
        """
        Returns the prompt to respond to the user's message for every conversation.

        Args:
            user_message (str): The message from the user.

        Returns:
            str: The formatted prompt with the user message added.

        Example:
            >>> prompt = PromptManager.get_prompt("Hello!")
            >>> print(prompt)
        """
        return user_message
