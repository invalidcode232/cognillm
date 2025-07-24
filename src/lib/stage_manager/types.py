from enum import Enum
from typing import List, Union, Optional
from dataclasses import dataclass


class Stage(Enum):
    """
    The first three stages of the Motivational Interviewing framework.
    """

    PRE_CONTEMPLATION = "pre_contemplation"
    CONTEMPLATION = "contemplation"
    PREPARATION = "preparation"

    def __str__(self) -> str:
        return self.value

    def next_stage(self) -> Optional["Stage"]:
        """
        Returns the next stage in the Motivational Interviewing progression.
        Returns None if this is the final stage.
        """
        stage_progression = {
            Stage.PRE_CONTEMPLATION: Stage.CONTEMPLATION,
            Stage.CONTEMPLATION: Stage.PREPARATION,
            Stage.PREPARATION: None,  # Final stage
        }
        return stage_progression.get(self)

    @property
    def is_final_stage(self) -> bool:
        """Returns True if this is the final stage in the progression."""
        return self.next_stage() is None


class EvaluationMethods(Enum):
    """
    The evaluation methods.
    """

    OBJECTIVE_COMPLETION = "objective_completion"
    TABLE_COMPARISON = "table_comparison"

    def __str__(self) -> str:
        return self.value


@dataclass
class TableData:
    """
    Table structure for table comparison evaluation method.
    """

    costs: List[str]
    rewards: List[str]


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation methods.
    Can handle either objective completion (list of strings) or table comparison (TableData).
    """

    method: EvaluationMethods
    data: Union[List[str], TableData]

    @classmethod
    def from_objectives(cls, objectives: List[str]) -> "EvaluationConfig":
        """Create an EvaluationConfig for objective completion."""
        return cls(method=EvaluationMethods.OBJECTIVE_COMPLETION, data=objectives)

    @classmethod
    def from_table(cls, costs: List[str], rewards: List[str]) -> "EvaluationConfig":
        """Create an EvaluationConfig for table comparison."""
        return cls(
            method=EvaluationMethods.TABLE_COMPARISON,
            data=TableData(costs=costs, rewards=rewards),
        )
