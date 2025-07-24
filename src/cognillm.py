from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
import json

from .lib.base.ai import Client
from .prompt_manager import PromptManager
from .logger_config import setup_logger


# Configuration constants
MAX_TOKENS: int = 1000
TEMPERATURE: float = 0.5

logger = setup_logger()


class CogniLLM:
    """
    A roleplaying system that uses a LLM to generate a cognitive model for each response,
    and then uses the cognitive model to generate a response.
    """

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
                raise ValueError(
                    f"Text generator LLM response is missing required field: {field}"
                )

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

        # Initialize PromptManager
        self.prompt_manager: PromptManager = PromptManager(
            profile_path=profile_path,
        )
        self.base_prompt: str = self.prompt_manager.get_base_prompt()

        logger.debug("Base prompt retrieved successfully")

        # Initialize the AI Client
        self.ai_client: Client = Client(
            system_prompt=self.base_prompt,
            endpoint=endpoint,
            deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            max_tokens=max_tokens,
            temperature=temperature,
            history=history,
        )

        logger.info(f"Initialized <CogniLLM> successfully")

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
        if "content" in last_message and "role" in last_message:
            if last_message["role"] != "assistant":
                raise ValueError(
                    "Last message is not an assistant message, skipping clean up... (last_message might be of unexpected value)"
                )

            last_message_data = json.loads(last_message["content"])
            last_message_data.pop("chain_of_thought", None)
            last_message["content"] = json.dumps(last_message_data)
        else:
            raise ValueError(
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
        prompt = self.prompt_manager.get_message_prompt(user_message)
        response, tokens_used = self.ai_client.send_message(prompt)

        # Clean up the response to minimize context length;
        # right now, it simply removes the chain_of_thought from the response.
        self._clean_response()

        # Validates and parses the response
        parsed_response = self._parse_response(response)
        logger.debug(f"Parsed response:\n{json.dumps(parsed_response, indent=2)}")
        logger.debug(f"Tokens used: {tokens_used}")
        logger.debug("=" * 40)

        return parsed_response.get("message"), tokens_used

    def get_conversation_history(self) -> list[ChatCompletionMessageParam]:
        """
        Get the conversation history.

        Args:
            None

        Returns:
            list[ChatCompletionMessageParam]: The conversation history.

        Example:
            >>> history = CogniLLM.get_conversation_history()
            >>> print(history)
        """

        return self.ai_client.get_history()

    def reset_conversation(self) -> None:
        """
        Reset the conversation history.

        Args:
            None

        Returns:
            None

        Example:
            >>> CogniLLM.reset_conversation()
            >>> print(CogniLLM.get_conversation_history())
        """
        self.ai_client.reset_conversation()
