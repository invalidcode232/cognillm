from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
import json
import yaml
import logging

from .lib.base.ai import Client
from .prompt_manager import PromptManager


logger = logging.getLogger(__name__)


MAX_TOKENS: int = 1000
TEMPERATURE: float = 0.5


class CogniLLM:
    """
    A roleplaying system that uses a LLM to generate a cognitive model for each response,
    and then uses the cognitive model to generate a response.

    Attributes:
        profile_name (str): The name of the baseline profile to embody
        profile_goal (str): The goal of the profile
        profile_short_description (str): The short description of the profile
        profile_cognitive_model (dict[str, str]): The cognitive model of the profile
        history (list[ChatCompletionMessageParam] | None): The history of messages
    """

    @staticmethod
    def _validate_profile(profile: dict[str, str]) -> None:
        """
        Validate the profile dictionary.

        Args:
            profile (dict[str, str]): The profile dictionary

        Raises:
            ValueError: If the profile is missing a required key or the cognitive model is not a dictionary
        """
        required_keys = ["name", "goal", "short_description", "cognitive_model"]

        # Check if the profile has the required keys
        for key in required_keys:
            if key not in profile:
                logger.error(f"Profile is missing required key: {key}")
                raise ValueError(f"Profile is missing required key: {key}")

        # Check if the cognitive model is a dictionary
        if not isinstance(profile["cognitive_model"], dict):
            logger.error("Cognitive model is not a dictionary")
            raise ValueError("Cognitive model is not a dictionary")

        # Check if the cognitive model has the required keys
        required_cm_keys = [
            "core_beliefs",
            "intermediate_beliefs",
            "coping_strategies",
            "situation",
        ]

        for key in required_cm_keys:
            if key not in profile["cognitive_model"]:
                logger.error(f"Cognitive model is missing required key: {key}")
                raise ValueError(f"Cognitive model is missing required key: {key}")

    @staticmethod
    def _parse_response(response: str) -> dict[str, str]:
        """
        Parse the response from the AI client.

        Args:
            response (str): The response from the AI client.

        Returns:
            dict[str, str]: The parsed response.
        """
        try:
            response: dict[str, str] = json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Error parsing response: {response}")
            raise json.JSONDecodeError(f"Error parsing response: {response}")

        fields = [
            "automatic_thoughts",
            "emotions",
            "behaviors",
            "openness",
            "message",
            "chain_of_thought",
        ]

        for field in fields:
            if field not in response:
                logger.error(f"Response is missing required field: {field}")
                raise ValueError(f"Response is missing required field: {field}")

        return response

    def __init__(
        self,
        profile_path: str,
        endpoint: str,
        deployment: str,
        api_key: str,
        api_version: str,
        history: list[ChatCompletionMessageParam] | None = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ):
        """
        Initialize the CogniLLM roleplaying system.

        Args:
            profile_path (str): The path to the profile YAML file
            endpoint (str): Azure OpenAI endpoint URL
            deployment (str): The deployment/model name to use
            api_key (str): Azure OpenAI API key
            api_version (str): API version to use
            history (list[ChatCompletionMessageParam] | None): The history of messages
            temperature (float): The temperature to use for the AI client
            max_tokens (int): The maximum number of tokens to use for the AI client
        """
        self.history: list[ChatCompletionMessageParam] | None = history

        # Check and load the profile
        try:
            logger.info(f"Loading profile from {profile_path}")

            with open(profile_path, "r") as file:
                profile = yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Profile file not found: {profile_path}")
            raise FileNotFoundError(f"Profile file not found: {profile_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing profile file: {e}")
            raise yaml.YAMLError(f"Error parsing profile file: {e}")

        self._validate_profile(profile)

        # Get the profile attributes
        self.profile_name: str = profile.get("name")
        self.profile_goal: str = profile.get("goal")
        self.profile_short_description: str = profile.get("short_description")
        self.profile_cognitive_model: dict[str, str] = profile.get("cognitive_model")

        logger.info(f"Profile validated and loaded successfully: {self.profile_name}")

        # Generate the prompt for AI Client
        self.prompt_manager: PromptManager = PromptManager(
            profile_name=self.profile_name,
            profile_cognitive_model=self.profile_cognitive_model,
        )

        self.prompt: str = self.prompt_manager.get_base_prompt()

        logger.info(f"Prompt generated successfully: {self.prompt}")

        # Initialize the AI Client
        self.ai_client: Client = Client(
            system_prompt=self.prompt,
            endpoint=endpoint,
            deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            max_tokens=max_tokens,
            temperature=temperature,
            history=history,
        )

        logger.info(f"AI Client initialized successfully: {self.ai_client}")

    def _clean_response(self) -> None:
        """
        Removes chain_of_thought from the response and overrides the history of the AI client with the filtered response.
        Used to minimize context length of the chat history.

        Args:
            None

        Returns:
            None

        Example:
            >>> original_response = CogniLLM.send_message("Hello, how are you?")
            >>> CogniLLM._clean_response()
            >>> return original_response # We will return the full response to the user, but clean it up on the backend.
        """
        last_message = self.ai_client.get_history_index(-1)
        if "content" in last_message:
            last_message_data = json.loads(last_message["content"])
            last_message_data.pop("chain_of_thought", None)
            last_message["content"] = json.dumps(last_message_data)
        else:
            logger.warning(
                "Last message is not a dictionary, skipping clean up... (last_message might be of unexpected value)"
            )

    def send_message(self, user_message: str) -> tuple[dict[str, str], int]:
        """
        Send a message to the AI client,
        returns a complex response containing the updated dynamic cognitive model fields and final message output:
        - automatic_thoughts
        - emotions
        - behaviors
        - openness
        - message

        Args:
            user_message (str): The message from the user.

        Returns:
            tuple[dict[str, str], int]: The response from the AI Client and the number of tokens used.

        Example:
            >>> response = CogniLLM.send_message("Hello, how are you?")
            >>> print(response)
        """
        prompt: str = self.prompt_manager.get_prompt(user_message)
        response, tokens_used = self.ai_client.send_message(prompt)

        self._clean_response()

        # Validates and parses the response
        return self._parse_response(response), tokens_used

    def get_conversation_history(self) -> list[ChatCompletionMessageParam]:
        """
        Get the conversation history.

        Returns:
            list[ChatCompletionMessageParam]: The conversation history.
        """

        return self.ai_client.get_history()

    def reset_conversation(self) -> None:
        """
        Reset the conversation history.
        """
        self.ai_client.reset_conversation()
