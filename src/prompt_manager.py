import os
import json
import yaml
from .lib.stage_manager.types import StageConfig, EvaluationConfig


# Path to prompt we use to establish the base AI logic
PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "include", "prompts", "cognillm_a.txt"
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

    @staticmethod
    def _validate_stage_config(config: dict) -> StageConfig:
        """
        Validates and parses the stage configuration data.

        Args:
            config (dict): The config dictionary containing stage configurations.

        Returns:
            StageConfig: The validated stage configuration object.

        Raises:
            ValueError: If the stage configuration is invalid.
        """
        stage_configs = {}

        for stage_name in ["pre_contemplation", "contemplation", "preparation"]:
            if stage_name in config:
                stage_data = config[stage_name]

                if not isinstance(stage_data, dict):
                    raise ValueError(f"Stage {stage_name} must be a dictionary")

                if "type" not in stage_data:
                    raise ValueError(f"Stage {stage_name} must have a 'type' field")

                evaluation_type = stage_data["type"]

                if evaluation_type == "objective_completion":
                    if "objectives" not in stage_data:
                        raise ValueError(
                            f"Stage {stage_name} with objective_completion type must have 'objectives' field"
                        )

                    objectives = stage_data["objectives"]
                    if not isinstance(objectives, list) or not all(
                        isinstance(obj, str) for obj in objectives
                    ):
                        raise ValueError(
                            f"Objectives in stage {stage_name} must be a list of strings"
                        )

                    stage_configs[stage_name] = EvaluationConfig.from_objectives(
                        objectives
                    )

                elif evaluation_type == "table_comparison":
                    if "default_table" not in stage_data:
                        raise ValueError(
                            f"Stage {stage_name} with table_comparison type must have 'default_table' field"
                        )

                    table_data = stage_data["default_table"]
                    if not isinstance(table_data, dict):
                        raise ValueError(
                            f"default_table in stage {stage_name} must be a dictionary"
                        )

                    if "costs" not in table_data or "rewards" not in table_data:
                        raise ValueError(
                            f"default_table in stage {stage_name} must have 'costs' and 'rewards' fields"
                        )

                    costs = table_data["costs"]
                    rewards = table_data["rewards"]

                    if not isinstance(costs, list) or not all(
                        isinstance(cost, str) for cost in costs
                    ):
                        raise ValueError(
                            f"Costs in stage {stage_name} must be a list of strings"
                        )

                    if not isinstance(rewards, list) or not all(
                        isinstance(reward, str) for reward in rewards
                    ):
                        raise ValueError(
                            f"Rewards in stage {stage_name} must be a list of strings"
                        )

                    stage_configs[stage_name] = EvaluationConfig.from_table(
                        costs, rewards
                    )

                else:
                    raise ValueError(
                        f"Unknown evaluation type '{evaluation_type}' in stage {stage_name}"
                    )

        return StageConfig(
            pre_contemplation=stage_configs.get("pre_contemplation"),
            contemplation=stage_configs.get("contemplation"),
            preparation=stage_configs.get("preparation"),
        )

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

        # 3) Process stage configuration --
        # Validate and create stage config from the loaded config
        self.stage_config = self._validate_stage_config(self.config)

    def override_cognillm_variable(self, variable: str, value: str) -> None:
        """
        Overrides a variable in the base AI prompt.
        """
        # Check if variable exists in the base prompt
        if f"%{variable}%" not in self.base_prompt:
            raise ValueError(f"Variable {variable} not found in base prompt")

        return self.base_prompt.replace(f"%{variable}%", value)

    def get_stage_config(self) -> StageConfig:
        """
        Returns the stage config of the profile.

        Returns:
            StageConfig: The stage configuration object containing evaluation configs for each stage.
        """
        return self.stage_config

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

    def get_message_prompt(self, user_message: str, stage_info: list[str]) -> str:
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
        # return user_message
        prompt_obj = {
            "stage_info": stage_info,
            "user_message": user_message,
        }

        return json.dumps(prompt_obj, indent=2)

    def get_config(self) -> dict[str, str]:
        """
        Returns the config of the profile.

        Returns:
            dict[str, str]: The config of the profile.
        """
        return self.config
