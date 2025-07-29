from enum import Enum
from typing import List, Union, Optional, Dict, Any
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


@dataclass
class StageConfig:
    """
    Configuration for the stage manager.

    Sample stage config:
    >>> {
    ...     Stage.PRE_CONTEMPLATION: EvaluationConfig.from_objectives(["Madam Chan must be able to provide for her children"]),
    ...     Stage.CONTEMPLATION: EvaluationConfig.from_table(costs=["Madam Chan must be able to provide for her children with her husband"], rewards=["If madam chan has more money, she can provide for her children"]),
    ...     Stage.PREPARATION: EvaluationConfig.from_objectives(["Make Madam Chan realize that she has the right to leave her husband"]),
    ... }
    """

    pre_contemplation: Optional[EvaluationConfig]
    contemplation: Optional[EvaluationConfig]
    preparation: Optional[EvaluationConfig]

    def __getitem__(self, stage: Stage) -> Optional[EvaluationConfig]:
        """
        Get the evaluation config for a specific stage.

        Args:
            stage (Stage): The stage to get the evaluation config for.

        Returns:
            Optional[EvaluationConfig]: The evaluation config for the stage, or None if not configured.

        Raises:
            KeyError: If the stage is not a valid Stage enum value.
        """
        stage_mapping = {
            Stage.PRE_CONTEMPLATION: self.pre_contemplation,
            Stage.CONTEMPLATION: self.contemplation,
            Stage.PREPARATION: self.preparation,
        }

        if stage not in stage_mapping:
            raise KeyError(f"Invalid stage: {stage}")

        return stage_mapping[stage]

    def __contains__(self, stage: Stage) -> bool:
        """
        Check if a stage is configured (has a non-None evaluation config).

        Args:
            stage (Stage): The stage to check.

        Returns:
            bool: True if the stage is configured, False otherwise.
        """
        try:
            return self[stage] is not None
        except KeyError:
            return False

    def __len__(self) -> int:
        """
        Return the number of configured stages (non-None evaluation configs).

        Returns:
            int: The number of stages that have been configured.
        """
        count = 0
        if self.pre_contemplation is not None:
            count += 1
        if self.contemplation is not None:
            count += 1
        if self.preparation is not None:
            count += 1
        return count
