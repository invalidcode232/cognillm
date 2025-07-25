from base import Client
import json
import os
import yaml

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "include", "prompts", "summaryllm.txt")

def get_summary_system_prompt(profile_path: str) -> str:
    """
    Returns the system prompt for the summary-based memory system.
    Default path is 'src/include/prompts/cognillm.txt'.
    
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
    ):
        self.window_size: int = summary_window_size
        self.history_list: list = []
        self.summary_list: list = []
        self.system_prompt: str = get_summary_system_prompt(profile_path=profile_path)
        
        if tool_name != "openai":
            raise ValueError("Currently, only 'openai' is supported as a tool name.")
        
        self.summary_tool: Client = Client(
            system_prompt=self.system_prompt,
            endpoint=endpoint,
            deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    def summary_signal(self):
        """
        Generates a summary signal based on the current history size as well as the summary window size.
        Returns:
            A boolean indicating whether the summary should be generated.
        """
        if len(self.history_list) >= self.window_size:
            return True
        return False

    def add_to_history(self, history_item):
        """
        Adds an item to the history list and maintains the window size.
        
        Args:
            history_item: The item to be added to the history.
        """
        self.history_list.append(history_item)
    
    def _clear_history(self):
        """
        Clears the history list.
        """
        self.history_list = []

    def add_summary(self):
        """
        Generates a summary of the current history and adds it to the summary list.
        If the summary signal is triggered, it will create a summary using the summary tool.
        """
        prompt = json.dumps(self.history_list)
        summary = self.summary_tool.send_message(message=prompt)
        self.summary_list.append(summary)
        self._clear_history()
        return summary
    
    def update(self, history_item):
        """
        Automatically update memory
        """
        self.add_to_history(history_item)
        if self.summary_signal():
            self.add_summary()
