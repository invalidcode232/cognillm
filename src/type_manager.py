from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict

@dataclass
class History:
    """
    Represents a single conversation history entry.
    """

    id: str
    role: str
    content: str
    tokens: Optional[int] = None


@dataclass
class Summary:
    """
    Represents a summary of conversation history.
    """

    id: str
    summarized_messages: List[str] # List of ids of summarized messages
    summary: str

