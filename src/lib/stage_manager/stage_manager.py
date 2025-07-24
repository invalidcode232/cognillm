import logging
from .types import Stage, EvaluationConfig, EvaluationMethods
from .evaluators import Evaluator


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
        endpoint: str,
        deployment: str,
        api_key: str,
        api_version: str,
        logger: logging.Logger,
        stage_config: dict[Stage, EvaluationConfig],
        initial_stage: Stage = Stage.PRE_CONTEMPLATION,
        message_index: int = 0,
    ):
        """
        Initializes the stage manager.

        Args:
            endpoint (str): The endpoint of the AI client.
            deployment (str): The deployment of the AI client.
            api_key (str): The API key of the AI client.
            api_version (str): The API version of the AI client.
            logger (logging.Logger): The logger to use for logging.
            stage_config (dict[Stage, EvaluationConfig]): The stage config mapping stages to their evaluation configurations.
            initial_stage (Stage): The initial stage.
            message_index (int): The index of the first message, defaults to 0.
        """

        self.stage_config = stage_config
        self.current_stage = initial_stage
        self.current_message_index = message_index
        self.logger = logger

        # Used for checking if stage should be advanced
        self.evaluator = Evaluator(
            endpoint=endpoint,
            deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            logger=logger,
        )

        # Stage tracking
        self.stage_history = {
            initial_stage: {
                "start": self.current_message_index,
                "end": -1,
                "messages": [],
            }
        }

        self.logger.info(
            f"Initialized <StageManager> with {len(self.stage_config)} stages, starting at {initial_stage}, message index {message_index}"
        )

    def handle_message_add(self, message: str) -> None:
        """
        Handles the addition of a message to the current stage.

        Args:
            message (str): The message to add to the current stage.
        """
        # Add the message to the stage history
        self.stage_history[self.current_stage]["messages"].append(message)
        self.current_message_index += 1

        # Evaluate the stage and advance the stage if the objective is completed
        result = self.evaluate_stage()
        if result is None:
            self.logger.error(
                f"[StageManager] Invalid response from <Evaluator> for stage {self.current_stage}"
            )
        elif result is True:
            self.logger.info(
                f"Stage {self.current_stage} completed, advancing to {self.current_stage.next_stage()}"
            )
            self.advance_stage()

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

    def evaluate_stage(self) -> bool | None:
        """
        Evaluates the current stage and advances the stage if the objective is completed.

        Returns:
            bool | None: True if the stage was advanced, False otherwise, will return None if the response is invalid.
        """

        # Get the evaluation config for the current stage
        evaluation_config = self.stage_config[self.current_stage]
        result = None
        if evaluation_config.method == EvaluationMethods.OBJECTIVE_COMPLETION:
            result = self.evaluator.evaluate_objective_completion(
                evaluation_config.data,
                self.stage_history[self.current_stage]["messages"],
            )
        elif evaluation_config.method == EvaluationMethods.TABLE_COMPARISON:
            result = self.evaluator.evaluate_table_comparison(
                evaluation_config.data,
                self.stage_history[self.current_stage]["messages"],
            )
        else:
            raise ValueError(
                f"Invalid/unsupported evaluation method: {evaluation_config.method}"
            )

        return result

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
