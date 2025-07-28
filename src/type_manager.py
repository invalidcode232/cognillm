from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict

@dataclass
class History:
    """
    Represents a single conversation history entry.
    """

    role: str
    content: str
    tokens: Optional[int] = None


@dataclass
class Summary:
    """
    Represents a summary of conversation history.
    """

    summary: str

