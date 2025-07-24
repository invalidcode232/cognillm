import os
import json
import yaml


# Path to prompt we use to establish the base AI logic
PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "include", "prompts", "cognillm.txt"
)


class PromptManager:
    @staticmethod
    def _validate_config(config: str) -> dict[str, str]:
        """
        Validates the config data.
        """
        config = yaml.safe_load(config)

        if not config:
            raise ValueError("Config data is empty")

        return config

    @staticmethod
    def _validate_profile(profile: str) -> dict[str, str]:
        """
        Validates the profile data.
        """
        profile = yaml.safe_load(profile)

        if not profile:
            raise ValueError("Profile data is empty")

        profile_fields = ["name", "goal", "short_description", "cognitive_model"]
        for field in profile_fields:
            if field not in profile:
                raise ValueError(f"Field {field} not found in profile")

        return profile

    def __init__(self, profile_path: str) -> str:
        """
        Retrieves the main CogniLLM prompt and replaces dynamic variables with the provided values.

        Args:
            profile_path (str): The path to the profile directory.
        """
        file_names = ["config.yaml", "profile.yaml"]

        profile_data = {}

        for file_name in file_names:
            if os.path.exists(os.path.join(profile_path, file_name)):
                with open(os.path.join(profile_path, file_name), "r") as f:
                    profile_data[file_name] = f.read()
            else:
                raise FileNotFoundError(f"File {file_name} not found in {profile_path}")

        # Read base AI prompt contents
        if os.path.exists(PROMPT_PATH):
            with open(PROMPT_PATH, "r") as f:
                contents = f.read()
        else:
            raise FileNotFoundError(f"File {PROMPT_PATH} not found")

        # 1) Process profile.yaml --
        # Validate and parse profile.yaml
        profile = self._validate_profile(profile_data["profile.yaml"])

        # Replace dynamic variables in base AI prompt with profile data
        self.base_prompt = contents.replace("%profile%", json.dumps(profile)).replace(
            "%name%", profile["name"]
        )

        # 2) Process config.yaml --
        # Validate and parse config.yaml
        self.config = self._validate_config(profile_data["config.yaml"])

    def get_base_prompt(self) -> str:
        """
        Gets the baseline prompt with the dynamic variables replaced.

        Returns:
            str: The formatted prompt with dynamic variables replaced.

        Example:
            >>> prompt = PromptManager.get_base_prompt()
            >>> print(prompt)
        """
        return self.base_prompt

    def get_message_prompt(self, user_message: str) -> str:
        """
        Returns the prompt to respond to the user's message for every conversation.

        Args:
            user_message (str): The message from the user.

        Returns:
            str: The formatted prompt with the user message added.

        Example:
            >>> prompt = PromptManager.get_message_prompt("Hello!")
            >>> print(prompt)
        """
        return user_message

    def get_config(self) -> dict[str, str]:
        """
        Returns the config of the profile.

        Returns:
            dict[str, str]: The config of the profile.
        """
        return self.config
