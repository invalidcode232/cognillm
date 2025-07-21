from .lib.base import Client
import json
from .prompt_manager import PromptManager
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

class CogniLLM:
    """
    A dual-LLM roleplaying system that separates internal thought processes from response generation.

    This system uses two LLM instances:
    1. Brain LLM: Generates internal thoughts, emotions, and mental state based on the persona
    2. Text Generator LLM: Creates the actual response using the thoughts, emotions, and user input

    This architecture allows for more nuanced and consistent character behavior by separating
    the character's internal mental processes from their external communication.

    Attributes:
        brain_client (Client): LLM instance responsible for generating thoughts and emotions
        text_generator_client (Client): LLM instance responsible for generating responses
        profile_name (str): The character description/persona being roleplayed

    """

    def __init__(
        self,
        profile_name: str,
        endpoint: str,
        deployment: str,
        api_key: str,
        api_version: str,
        cognitive_model: str,
        brain_max_tokens: int = 16384,
        text_max_tokens: int = 16384,
        brain_temperature: float = 0.8,
        text_temperature: float = 0.7,
        brain_history: list[ChatCompletionMessageParam] | None = None,
        messages_history: list[ChatCompletionMessageParam] | None = None,
    ):
        """
        Initialize the dual-LLM roleplaying system.

        Args:
            profile_name (str): Character description/personality that the AI should embody
            endpoint (str): Azure OpenAI endpoint URL
            deployment (str): The deployment/model name to use
            api_key (str): Azure OpenAI API key
            api_version (str): API version to use
            brain_max_tokens (int): Max tokens for brain LLM (internal thoughts)
            text_max_tokens (int): Max tokens for text generator LLM (responses)
            brain_temperature (float): Temperature for brain LLM (higher for more creative thoughts)
            text_temperature (float): Temperature for text generator LLM
            **kwargs: Additional parameters passed to both Client instances
        """
        self.profile_name = profile_name

        # Prompt manager to retrieve the system prompts
        self.prompt_manager = PromptManager(
            brain_prompt_name="brain_cot.txt",
            messagegen_prompt_name="textgen.txt",
            cognitive_model=cognitive_model,
            profile_name=profile_name,
        )  # cognitive_model will be injected into the system prompts by PromptManager

        # Store the cognitive model name for reference if needed
        self.cognitive_model_name = cognitive_model

        # Initialize the Brain LLM - responsible for internal thoughts and emotions
        brain_system_prompt = self.prompt_manager.get_brain_prompt()

        self.brain_client = Client(
            system_prompt=brain_system_prompt,
            endpoint=endpoint,
            deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            max_tokens=brain_max_tokens,
            temperature=brain_temperature,
            history=brain_history,
        )

        # Initialize the Message Generator LLM - responsible for actual responses
        message_system_prompt = self.prompt_manager.get_textgen_prompt()
        self.message_generator_client = Client(
            system_prompt=message_system_prompt,
            endpoint=endpoint,
            deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            max_tokens=text_max_tokens,
            temperature=text_temperature,
            history=messages_history,
        )

        self.last_client_message = None

        # Parse the cognitive model JSON string into a dict
        try:
            self.current_cognitive_model = json.loads(cognitive_model)
        except json.JSONDecodeError:
            print(f"Error parsing cognitive model JSON: {cognitive_model}")
            self.current_cognitive_model = {}

    def _generate_thoughts(self, user_message: str) -> tuple[dict[str, str] | None, int | None]:
        """
        Generate internal thoughts and emotions using the brain LLM.

        Args:
            user_message (str): The message from the user

        Returns:
            tuple[dict[str, str] | None, int | None]: A tuple containing:
                - JSON object containing thoughts, emotions, and mental state (dict)
                - The total number of tokens used in the brain completion (int)
        """

        messagegen_input = {
            "user_message": user_message,
        }

        last_client_message = self.message_generator_client.get_history_index(-1)
        if last_client_message and "content" in last_client_message and isinstance(last_client_message["content"], str) and last_client_message["role"] == "assistant":
            messagegen_input["last_client_message"] = last_client_message["content"]

        brain_response, brain_token_usage = self.brain_client.send_message(json.dumps(messagegen_input))
        try:
            # Parse the JSON response from the brain
            thoughts_data: dict[str, str] = json.loads(brain_response)

            # Merge with existing cognitive model, keeping existing fields and updating only the new ones
            if self.current_cognitive_model:
                # Update existing cognitive model with new fields
                self.current_cognitive_model.update(thoughts_data)

                return self.current_cognitive_model, brain_token_usage
            else:
                # Initialize cognitive model if it's empty
                self.current_cognitive_model = thoughts_data
                return thoughts_data, brain_token_usage

        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            print(f"Error parsing JSON: {brain_response}")
            return None, brain_token_usage

    def send_message(self, user_message: str) -> tuple[str, dict[str, str] | None, int | None, int | None]:
        """
        Process a user message and generate a character response.

        This method orchestrates the two-LLM process:
        1. Brain LLM analyzes the message and generates internal thoughts/emotions
        2. Text Generator LLM creates the actual response based on the internal state

        Args:
            user_message (str): The message from the user

        Returns:
            tuple[str, dict]: A tuple containing:
                - The character's response (str)
                - The internal thoughts and emotions (dict)
        """
        # Step 1: Generate internal thoughts and emotions
        thoughts_data, brain_token_usage = self._generate_thoughts(user_message)

        context_data = {
            "user_message": user_message,
            "current_cognitive_state": thoughts_data,
            "instructions": "Generate your character's response to the user's message, MAKING SURE THAT IT REFLECTS THE CHARACTER'S CURRENT COGNITIVE STATE.",
        }

        # Step 3: Generate the actual response
        character_response, message_token_usage = self.message_generator_client.send_message(json.dumps(context_data))

        self._clean_last_client_message()

        return character_response, thoughts_data, brain_token_usage, message_token_usage
    
    def _clean_last_client_message(self) -> None:
        """
        Clean the last client message, this includes removing the persona, core beliefs, intermediate beliefs, and coping strategies.
        This is done to reduce token count and address the context window limitations, or at least make it last longer before it's forgotten.
        """
        last_user_message = self.message_generator_client.get_history_index(-2)

        if last_user_message["role"] == "user" and "content" in last_user_message and isinstance(last_user_message["content"], str):
            try:
                last_user_message_content = json.loads(last_user_message["content"])
                last_user_cognitive_state = last_user_message_content["current_cognitive_state"]

                if not last_user_cognitive_state:
                    print(f"No cognitive state found in last user message: {last_user_message_content}", flush=True)
                    return
                
                # Remove unwanted keys if they exist
                last_user_cognitive_state.pop("persona", None)
                last_user_cognitive_state.pop("core_beliefs", None)
                last_user_cognitive_state.pop("intermediate_beliefs", None)
                last_user_cognitive_state.pop("coping_strategies", None)
                last_user_cognitive_state.pop("situation", None)
                
                # Put the cleaned content back into the message
                last_user_message["content"] = json.dumps(last_user_message_content)
            except json.JSONDecodeError:
                print(f"Error parsing JSON content: {last_user_message['content']}", flush=True)
        else:
            print(f"Last user message is not a user message: {last_user_message}", flush=True)

    def get_conversation_history(self) -> tuple[list, list]:
        """
        Get the conversation history from both LLM instances.

        Returns:
            tuple[list, list]: text generator conversation history [0] and brain conversation history [1]
        """
        return self.message_generator_client.get_history(), self.brain_client.get_history()

    def reset_conversation(self) -> None:
        """
        Reset both LLM instances to start fresh conversations.
        """
        self.brain_client.reset_conversation()
        self.message_generator_client.reset_conversation()
