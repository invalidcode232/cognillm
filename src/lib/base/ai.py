from openai.types.chat import ChatCompletionMessageParam
from openai import AzureOpenAI
from .memory.summary import SummaryBasedMemory
import logging


class CompletionConfig:
    """
    Configuration container for Azure OpenAI chat completion parameters.

    This class encapsulates all the parameters needed to configure the behavior
    of the chat completion API, providing a clean separation of concerns and
    making it easy to manage completion settings.

    Attributes:
        model (str): The deployment/model name to use for completions.
        max_tokens (int): Maximum number of tokens to generate in the completion.
        temperature (float): Sampling temperature (0.0 to 2.0). Higher values
            make output more random, lower values more deterministic.
        top_p (float): Nucleus sampling parameter (0.0 to 1.0). Controls diversity
            by considering only tokens with cumulative probability up to top_p.
        frequency_penalty (float): Penalty for frequent tokens (-2.0 to 2.0).
            Positive values decrease likelihood of repeating tokens.
        presence_penalty (float): Penalty for new tokens (-2.0 to 2.0).
            Positive values increase likelihood of talking about new topics.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
    ):
        """
        Initialize the CompletionConfig with the specified parameters.

        Args:
            model (str): The deployment/model name to use for completions.
            max_tokens (int): Maximum number of tokens to generate (1-4096).
            temperature (float): Sampling temperature (0.0-2.0).
            top_p (float): Nucleus sampling parameter (0.0-1.0).
            frequency_penalty (float): Frequency penalty (-2.0 to 2.0).
            presence_penalty (float): Presence penalty (-2.0 to 2.0).
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty


class Client:
    """
    Azure OpenAI Chat Completion Client.

    This client proves a convenient interface for interacting with Azure OpenAI's
    chat completion API. It manages the conversation context, handles authentication,
    and proves methods for sending messages and receiving responses.

    The client maintains conversation history automatically and applies the specified
    system prompt to gue the AI's behavior throughout the conversation.

    Attributes:
        client (AzureOpenAI): The underlying Azure OpenAI client instance.
        completion_config (CompletionConfig): Configuration for completion parameters.
        chat_history (list[ChatCompletionMessageParam]): Conversation history including
            system prompt and all messages exchanged.

    Example:
        >>> client = Client(
        ...     system_prompt="You are a helpful coding assistant.",
        ...     endpoint="https://your-resource.openai.azure.com/",
        ...     deployment="gpt-4",
        ...     api_key="your-api-key",
        ...     api_version="2023-12-01-preview"
        ... )
        >>> response = client.send_message("Explain Python decorators")
        >>> print(response)
    """

    def __init__(
        self,
        system_prompt: str,
        endpoint: str,
        deployment: str,
        api_key: str,
        api_version: str,
        max_tokens: int = 800,
        temperature: float = 0.7,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        history: list[ChatCompletionMessageParam] | None = None,
        summary_enabled: bool = True,
        summary_window_size: int = 5,
        summary_model: str = "openai",
        summary_start_round: int = 50,
        profile_path: str | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the Azure OpenAI Client.

        Sets up the Azure OpenAI client connection, configures completion parameters,
        and initializes the conversation with the proved system prompt.

        Args:
            system_prompt (str): The system prompt that defines the AI's role and behavior.
            endpoint (str): Azure OpenAI endpoint URL (e.g., "https://your-resource.openai.azure.com/").
            deployment (str): The deployment/model name to use (e.g., "gpt-4", "gpt-35-turbo").
            api_key (str): Azure OpenAI API key for authentication.
            api_version (str): API version to use (e.g., "2023-12-01-preview").
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 800.
            temperature (float, optional): Sampling temperature (0.0-2.0). Defaults to 0.7.
            top_p (float, optional): Nucleus sampling parameter (0.0-1.0). Defaults to 0.95.
            frequency_penalty (float, optional): Frequency penalty (-2.0 to 2.0). Defaults to 0.
            presence_penalty (float, optional): Presence penalty (-2.0 to 2.0). Defaults to 0.
            history (list[ChatCompletionMessageParam] | None, optional): Existing conversation history. Defaults to None.
            summary_enabled (bool, optional): Whether to enable summary-based memory. Defaults to True.
            summary_window_size (int, optional): Number of rounds per summary window. Defaults to 5.
            summary_model (str, optional): Model to use for summarization. Defaults to "openai".
            summary_start_round (int, optional): Round number to start using summaries. Defaults to 50.
            profile_path (str | None, optional): Path to profile for summary system. Required if summary_enabled is True. Defaults to None.

        Raises:
            Exception: If Azure OpenAI client initialization fails due to inval credentials
                or configuration.
        """
        # Initialize the Azure OpenAI client with authentication
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

        # Create completion configuration with the specified parameters
        self.completion_config = CompletionConfig(
            model=deployment,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        # Store the original system prompt for reset functionality
        self.original_system_prompt = system_prompt

        # Store summary configuration
        self.summary_enabled = summary_enabled
        self.summary_window_size = summary_window_size
        self.summary_start_round = summary_start_round

        # Initialize summary memory if enabled and profile_path is proved
        self.summary_memory: SummaryBasedMemory | None = None
        if summary_enabled and profile_path:
            self.summary_memory = SummaryBasedMemory(
                profile_path=profile_path,
                summary_window_size=summary_window_size,
                endpoint=endpoint,
                deployment=deployment,
                api_key=api_key,
                api_version=api_version,
            )

        self.chat_prompt: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if history is not None:
            self.chat_prompt = history

        self.logger = logger

    def _prepare_prompt(self) -> list[ChatCompletionMessageParam]:
        """
        Prepare the prompt based on summary settings.

        Only replaces the earliest conversation windows with summaries,
        keeping recent conversations in their original form.

        Returns:
            list[ChatCompletionMessageParam]: The prepared prompt for the completion.
        """
        if not self.summary_enabled or not self.summary_memory:
            return self.chat_prompt

        # Calculate the number of conversation rounds (excluding system message)
        # Each round = 1 user message + 1 assistant message = 2 messages
        conversation_rounds = (len(self.chat_prompt) - 1) // 2

        # If we haven't reached the start round threshold, use history directly
        if conversation_rounds < self.summary_start_round:
            return self.chat_prompt

        # Calculate available summaries
        available_summaries = len(self.summary_memory.summary_list)
        required_summaries = (
            conversation_rounds - self.summary_start_round
        ) // self.summary_window_size + 1

        # If no summaries are available yet, return original history
        if available_summaries == 0:
            return self.chat_prompt

        # Start building the prompt with system message
        prompt = [self.chat_prompt[0]]

        # Add summaries for the earliest windows only
        for summary_idx in range(min(available_summaries, required_summaries)):
            start_round = summary_idx * self.summary_window_size + 1
            end_round = (summary_idx + 1) * self.summary_window_size

            # Access the summary content from the summary dictionary
            summary = self.summary_memory.summary_list[summary_idx]
            prompt.append(
                {
                    "role": "assistant",
                    "content": f"Previous conversation summary (rounds {start_round}-{end_round}): {summary}",
                }
            )

        # Calculate the starting index for remaining unsummarized history
        # Start from: 1 (system) + (available_summaries * summary_window_size * 2) messages
        summarized_messages = required_summaries * self.summary_window_size * 2
        remaining_start_index = 1 + summarized_messages

        # Add all remaining unsummarized conversation history
        if remaining_start_index < len(self.chat_prompt):
            prompt.extend(self.chat_prompt[remaining_start_index:])

        return prompt

    def _validate_completion(self, completion) -> None:
        """
        Validate the completion response from Azure OpenAI.

        Raises:
            ValidationError: If the completion response is invalid or empty.
        """
        # Check if completion has received
        if not completion.choices or not completion.choices[0].message.content:
            error_msg = "Invalid completion response: No choices or empty content"
            self.logger and self.logger.error(error_msg)
            raise Exception(error_msg)

    def _handle_api_error(self, error: Exception, operation: str = "API call") -> None:
        """
        Handle API errors with consistent logging and cleanup.

        Args:
            error (Exception): The exception that occurred
            operation (str): Description of the operation that failed
        """
        error_msg = f"{operation} failed: {error}"
        self.logger and self.logger.error(error_msg)

        # Remove the last user message from chat prompt if API call fails
        if self.chat_prompt and len(self.chat_prompt) > 1:
            last_message = self.chat_prompt[-1]
            if last_message.get("role") == "user":
                self.chat_prompt.pop()
                self.logger and self.logger.debug(
                    "Removed last user message from history due to error"
                )

    def send_message(self, message: str) -> tuple[str, int | None]:
        """
        Send a message to the AI and return the response.

        This method adds the user's message to the conversation history, sends the
        entire conversation context to the Azure OpenAI API, and returns the AI's
        response content and token usage.

        Args:
            message (str): The user's message to send to the AI.

        Returns:
            tuple[str, int | None]: A tuple containing the assistant's response content
                and the total tokens used (or None if not available).

        Raises:
            ValidationError: If the API returns no completion choices or empty content.
            CompletionError: If the API call fails due to network issues, authentication
                problems, or other API errors.

        Example:
            >>> response, tokens = client.send_message("What is machine learning?")
            >>> print(response)
            "Machine learning is a subset of artificial intelligence..."
            >>> print(tokens)
            100
        """
        user_history = {"role": "user", "content": message}

        # Add the user's message to the conversation history
        self.chat_prompt.append(user_history)

        # Prepare the prompt based on summary settings
        prompt = self._prepare_prompt()

        # self.logger and self.logger.debug(f"Prompt:\n {json.dumps(prompt, indent=2)}")

        try:
            # Send the conversation to Azure OpenAI and get the completion
            completion = self.client.chat.completions.create(
                model=self.completion_config.model,
                messages=prompt,
                max_tokens=self.completion_config.max_tokens,
                temperature=self.completion_config.temperature,
                top_p=self.completion_config.top_p,
                frequency_penalty=self.completion_config.frequency_penalty,
                presence_penalty=self.completion_config.presence_penalty,
            )

        except Exception as e:
            self.logger and self.logger.error(f"Chat completion failed: {e}")
            raise Exception(f"Chat completion failed: {e}") from e

        # Add assistant response to history
        assistant_response = completion.choices[0].message.content

        assistant_history = {"role": "assistant", "content": assistant_response}

        self.chat_prompt.append(assistant_history)

        # Update summary in background if enabled
        if self.summary_enabled and self.summary_memory:
            # Create a conversation round (pair) for summary
            conversation_round = [user_history, assistant_history]
            # Update summary
            self.summary_memory.update(conversation_round)

        # Return the AI's response and token usage
        return assistant_response, (
            completion.usage.total_tokens if completion.usage else None
        )

    def add_message_to_history(self, message: str) -> None:
        """
        Add a message to the conversation history without sending it to the API.

        This method is useful for adding context messages that will be included
        in subsequent API calls.

        Args:
            message (str): The message to add to conversation history.
        """
        self.chat_prompt.append(
            {
                "role": "user",
                "content": message,
            }
        )

    def reset_conversation(self) -> None:
        """
        Reset the conversation history to only include the original system prompt.

        This method is useful for benchmarking or when you want to start a fresh
        conversation while reusing the same client instance.
        """
        self.chat_prompt = [
            {
                "role": "system",
                "content": self.original_system_prompt,
            }
        ]

    def get_history_index(self, index: int) -> ChatCompletionMessageParam:
        """
        Get the index of the last client message in the conversation history.
        """

        try:
            return self.chat_prompt[index]
        except IndexError:
            raise IndexError(f"No message at index {index}")

    def get_history(self) -> list[ChatCompletionMessageParam]:
        """
        Get the conversation history.
        """
        return self.chat_prompt

    def get_summary_info(self) -> dict:
        """
        Get information about the summary system state.

        Returns:
            dict: Dictionary containing summary system information.
        """
        # Calculate current conversation rounds
        conversation_rounds = (len(self.chat_prompt) - 1) // 2

        if not self.summary_memory:
            return {
                "summary_enabled": self.summary_enabled,
                "summary_memory": None,
                "summaries_count": 0,
                "conversation_rounds_pending": 0,
                "total_conversation_rounds": conversation_rounds,
            }

        return {
            "summary_enabled": self.summary_enabled,
            "summary_memory": True,
            "summaries_count": len(self.summary_memory.summary_list),
            "conversation_rounds_pending": len(self.summary_memory.history_list),
            "total_conversation_rounds": conversation_rounds,
            "window_size": self.summary_window_size,
            "start_round": self.summary_start_round,
        }

    def get_summary_list(self) -> list[str] | None:
        """
        Get the list of summaries generated so far.

        Returns:
            list[str] | None: List of summaries, or None if summary is not enabled.
        """
        if self.summary_enabled and self.summary_memory:
            return self.summary_memory.get_summary_list()
        return None
