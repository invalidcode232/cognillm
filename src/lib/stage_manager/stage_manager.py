from .types import Stage, EvaluationConfig, EvaluationMethods


class StageManager:
    """
    The class that manages the progression of the Motivational Interviewing process.

    Sample stage config:
    >>> {
    ...     Stage.PRE_CONTEMPLATION: EvaluationConfig.from_objectives(["Madam Chan must be able to provide for her children"]),
    ...     Stage.CONTEMPLATION: EvaluationConfig.from_table(costs=["Madam Chan must be able to provide for her children with her husband"], rewards=["If madam chan has more money, she can provide for her children"]),
    ...     Stage.PREPARATION: EvaluationConfig.from_objectives(["Make Madam Chan realize that she has the right to leave her husband"]),
    ... }
    """

    def __init__(
        self,
        stage_config: dict[Stage, EvaluationConfig],
        initial_stage: Stage = Stage.PRE_CONTEMPLATION,
        message_index: int = 0,
    ):
        """
        Initializes the stage manager.

        Args:
            stage_config (dict[Stage, EvaluationConfig]): The stage config mapping stages to their evaluation configurations.
            initial_stage (Stage): The initial stage.
        """

        self.stage_config = stage_config
        self.current_stage = initial_stage
        self.current_message_index = message_index
        self.stage_history = {
            initial_stage: {
                "start": self.current_message_index,
                "end": -1,
                "messages": [],
            }
        }

    def advance_stage(self) -> bool:
        """
        Advances to the next stage in the progression.
        Uses the internally tracked current_message_index.

        Returns:
            bool: True if stage was advanced successfully, False if already at final stage.

        Raises:
            ValueError: If trying to advance to a stage not in the stage_config.
        """
        if self.current_stage.is_final_stage:
            return False

        next_stage = self.current_stage.next_stage()

        # Verify the next stage is configured
        if next_stage not in self.stage_config:
            raise ValueError(
                f"Next stage {next_stage} is not configured in stage_config"
            )

        # Update current stage's end index
        self.stage_history[self.current_stage]["end"] = self.current_message_index

        # Create new stage entry
        self.stage_history[next_stage] = {
            "start": self.current_message_index,
            "end": -1,
            "messages": [],
        }

        # Update current stage
        self.current_stage = next_stage
        return True

    def add_message(self, message: str) -> None:
        """
        Adds a message to the current stage and increments the message index.

        Args:
            message (str): The message to add to the current stage.
        """
        self.stage_history[self.current_stage]["messages"].append(message)
        self.current_message_index += 1

    def evaluate_stage(self) -> bool:
        """
        Evaluates the current stage.
        """

    # def increment_message_index(self) -> None:
    #     """
    #     Increments the current message index without adding a message.
    #     Useful when tracking message count without storing the actual messages.
    #     """
    #     self.current_message_index += 1

    # def can_advance(self) -> bool:
    #     """
    #     Checks if the current stage can be advanced to the next stage.

    #     Returns:
    #         bool: True if advancement is possible, False otherwise.
    #     """
    #     if self.current_stage.is_final_stage:
    #         return False

    #     next_stage = self.current_stage.next_stage()
    #     return next_stage in self.stage_config
