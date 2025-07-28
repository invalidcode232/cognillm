from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
import json
import logging

from .lib.base.ai import Client
from .prompt_manager import PromptManager


logger = logging.getLogger(__name__)


# AI completion parameter configuration
MAX_TOKENS: int = 1000
TEMPERATURE: float = 0.5


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
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing response: {response}")
            raise e

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
                logger.warning(f"Response is missing required field: {field}")

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
        summary_enabled: bool = True,
        summary_window_size: int = 5,
        summary_model: str = "openai",
        summary_start_round: int = 50,
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
            summary_enabled (bool): Whether to enable summary-based memory
            summary_window_size (int): Number of rounds per summary window
            summary_model (str): Model to use for summarization
            summary_start_round (int): Round number to start using summaries
        """
        self.history: list[ChatCompletionMessageParam] | None = history
        self.profile_path = profile_path
        self.summary_enabled: bool = summary_enabled

        # Generate the prompt for AI Client
        self.prompt_manager: PromptManager = PromptManager(
            profile_path=profile_path,
        )
        self.base_prompt: str = self.prompt_manager.get_base_prompt()

        logger.info(f"Prompt generated successfully: {self.base_prompt}")

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
            summary_enabled=summary_enabled,
            summary_window_size=summary_window_size,
            summary_model=summary_model,
            summary_start_round=summary_start_round,
            profile_path=profile_path,
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
        if "content" in last_message and "role" in last_message:
            if last_message["role"] != "assistant":
                logger.warning(
                    "Last message is not an assistant message, skipping clean up... (last_message might be of unexpected value)"
                )
                return

            last_message_data = json.loads(last_message["content"])
            last_message_data.pop("chain_of_thought", None)
            last_message["content"] = json.dumps(last_message_data)
        else:
            logger.warning(
                "Last message is not a dictionary, skipping clean up... (last_message might be of unexpected value)"
            )

    def send_message(self, user_message: str) -> tuple[dict[str, str], ChatCompletionMessageParam]:
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
            tuple[dict[str, str], ChatCompletionMessageParam]: The parsed response from the AI Client and the message dictionary containing response details.

        Example:
            >>> response, history = CogniLLM.send_message("Hello, how are you?")
            >>> print(response)
            >>> print(history["content"])
            >>> print(history["tokens"])
        """
        prompt = self.prompt_manager.get_message_prompt(user_message)
        response_history = self.ai_client.send_message(prompt)

        # Clean up the response to minimize context length;
        # right now, it simply removes the chain_of_thought from the response.
        self._clean_response()

        # Validates and parses the response
        return self._parse_response(response_history["content"]), response_history

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

    def get_summary_info(self) -> dict:
        """
        Get information about the summary system state.
        
        Returns:
            dict: Dictionary containing summary system information.
        """
        return self.ai_client.get_summary_info()
    
    def get_summary_list(self) -> list[str] | None:
        """
        Get the list of summaries generated so far.
        
        Returns:
            list[str]: List of summaries.
        """
        if self.summary_enabled:
            return self.ai_client.get_summary_list()
        return None
