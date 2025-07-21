from openai.types.chat import ChatCompletionMessageParam
from openai import AzureOpenAI


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

    This client provides a convenient interface for interacting with Azure OpenAI's
    chat completion API. It manages the conversation context, handles authentication,
    and provides methods for sending messages and receiving responses.

    The client maintains conversation history automatically and applies the specified
    system prompt to guide the AI's behavior throughout the conversation.

    Attributes:
        client (AzureOpenAI): The underlying Azure OpenAI client instance.
        completion_config (CompletionConfig): Configuration for completion parameters.
        chat_prompt (list[ChatCompletionMessageParam]): Conversation history including
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
    ):
        """
        Initialize the Azure OpenAI Client.

        Sets up the Azure OpenAI client connection, configures completion parameters,
        and initializes the conversation with the provided system prompt.

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

        Raises:
            Exception: If Azure OpenAI client initialization fails due to invalid credentials
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

        self.chat_prompt: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if history is not None:
            self.chat_prompt = history

    def send_message(self, message: str) -> tuple[str, int | None]:
        """
        Send a message to the AI and return the response.

        This method adds the user's message to the conversation history, sends the
        entire conversation context to the Azure OpenAI API, and returns the AI's
        response. The conversation history is automatically maintained.

        Args:
            message (str): The user's message to send to the AI.

        Returns:
            tuple[str, int]: A tuple containing:
                - The AI's response to the message (str)
                - The total number of tokens used in the completion (int)

        Raises:
            ValueError: If the API returns no completion choices or empty content.
            Exception: If the API call fails due to network issues, authentication
                problems, or other API errors.

        Example:
            >>> response, tokens = client.send_message("What is machine learning?")
            >>> print(response)
            "Machine learning is a subset of artificial intelligence..."
            >>> print(tokens)
            100
        """
        # Add the user's message to the conversation history
        self.chat_prompt.append(
            {
                "role": "user",
                "content": message,
            }
        )

        # Send the conversation to Azure OpenAI and get the completion
        completion = self.client.chat.completions.create(
            model=self.completion_config.model,
            messages=self.chat_prompt,
            max_tokens=self.completion_config.max_tokens,
            temperature=self.completion_config.temperature,
            top_p=self.completion_config.top_p,
            frequency_penalty=self.completion_config.frequency_penalty,
            presence_penalty=self.completion_config.presence_penalty,
        )

        # Validate that we received a proper response
        if not completion.choices or not completion.choices[0].message.content:
            raise ValueError("No completion choices returned")

        self.chat_prompt.append(
            {
                "role": "assistant",
                "content": completion.choices[0].message.content,
            }
        )

        # Return the AI's response content
        return completion.choices[0].message.content, (
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

        return self.chat_prompt[index]

    def get_history(self):
        return self.chat_prompt
