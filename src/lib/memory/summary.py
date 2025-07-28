import json
import os
import yaml
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam

PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "include", "prompts", "summaryllm.txt")

def get_summary_system_prompt(profile_path: str) -> str:
    """
    Returns the system prompt for the summary-based memory system.
    Default path is 'src/include/prompts/summaryllm.txt'.
    
    Returns:
        str: The system prompt for the summary-based memory system.
    """
    path = PROMPT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found at {path}")
    try:
        with open(path, "r") as file:
            content = file.read()
    except Exception as e:
        raise RuntimeError(f"Error reading summary system prompt file: {e}")
    
    # Read the profile.yaml file to get the profile
    profile = yaml.safe_load(open(os.path.join(profile_path, "profile.yaml"), "r"))
    if not profile:
        raise ValueError("Profile data is empty")
    return content.strip().replace("%profile%", json.dumps(profile))
    
class CompletionConfig:
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

class SummaryBasedMemory:
    """
    A class that represents a memory system based on summaries.
    
    This class is designed to handle memory operations that involve summarizing information.
    It can be used to store, retrieve, and manage summaries of various data.
    """

    def __init__(self,
        profile_path: str,
        summary_window_size: int = 5,
        tool_name: str = "openai",
        endpoint: str = "",
        deployment: str = "",
        api_key: str = "",
        api_version: str = "",
        max_tokens: int = 10000,
        temperature: float = 0.0,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
    ):
        self.window_size: int = summary_window_size
        self.history_list: list[ChatCompletionMessageParam] = []  # Now stores basic message dictionaries
        self.summary_list: list[dict] = []  # Now stores basic summary dictionaries
        self.system_prompt: str = get_summary_system_prompt(profile_path=profile_path)
        self.round_count: int = 0
        
        if tool_name != "openai":
            raise ValueError("Currently, only 'openai' is supported as a tool name.")
        
        self.summary_tool = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

        self.completion_config = CompletionConfig(
            model=deployment,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty= presence_penalty,
        )
    
    def increase_round_count(self):
        """
        Increases the round count by 1.
        This can be used to track the number of conversation rounds.
        """
        self.round_count += 1
    
    def reset_round_count(self):
        """
        Resets the round count to 0.
        This can be used to reset the conversation state.
        """
        self.round_count = 0

    def summary_signal(self):
        """
        Generates a summary signal based on the current history size as well as the summary window size.
        Returns:
            A boolean indicating whether the summary should be generated.
        """
        if self.round_count >= self.window_size:
            return True
        return False

    def add_to_history(self, history_item: ChatCompletionMessageParam):
        """
        Adds an item to the history list and maintains the window size.
        
        Args:
            history_item (ChatCompletionMessageParam): The message dictionary to be added to the history.
        """
        self.history_list.append(history_item)
    
    def _clear_history(self):
        """
        Clears the history list.
        """
        self.history_list: list[ChatCompletionMessageParam] = []

    def add_summary(self):
        """
        Generates a summary of the current history and adds it to the summary list.
        If the summary signal is triggered, it will create a summary using the summary tool.
        """
        
        # Create prompt with system message followed by conversation history in natural order
        prompt = [{"role": "system", "content": self.system_prompt}]
        
        # Add each message in the order they appear, maintaining user-assistant flow
        prompt.extend(self.history_list)
        
        # Debug
        # print("=" * 40)
        # import json
        # print(f"Prompt for summary:\n {json.dumps(prompt, indent=2)}")
        # print("=" * 40)

        response = self.summary_tool.chat.completions.create(
            messages=prompt,
            model=self.completion_config.model,
            max_tokens=self.completion_config.max_tokens,
            temperature=self.completion_config.temperature,
            top_p=self.completion_config.top_p,
            frequency_penalty=self.completion_config.frequency_penalty,
            presence_penalty=self.completion_config.presence_penalty,
        )
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("No completion choices returned")
        summary_content = response.choices[0].message.content.strip()
    
        # Create summary dictionary
        summary = {
            "summary": summary_content
        }
        
        self.summary_list.append(summary)
        self._clear_history()
        self.reset_round_count()
        return summary
    
    def update(self, history_items: list[ChatCompletionMessageParam]):
        """
        Automatically update memory with a list of message dictionaries
        
        Args:
            history_items (list[ChatCompletionMessageParam]): List of message dictionaries to be added.
        """
        for history_item in history_items:
            self.add_to_history(history_item)
        self.increase_round_count()
        if self.summary_signal():
            self.add_summary()

    def get_summary_list(self) -> list[dict]:
        """
        Returns the list of summaries.
        
        Returns:
            list[dict]: The list of summary dictionaries.
        """
        return self.summary_list if self.summary_list else []