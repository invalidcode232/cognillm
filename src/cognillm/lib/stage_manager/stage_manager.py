import logging
import os
import json
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from .types import Stage, EvaluationMethods, StageConfig
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

    def _process_existing_history(
        self, history: list[ChatCompletionMessageParam]
    ) -> tuple[dict[Stage, list[ChatCompletionMessageParam]], Stage | None]:
        """
        Processes the existing history:
        - Converts a list of messages into a dictionary of messages by stage.
        - Retrieves the last stage from the history

        Expected format:
        - First message: system prompt
        - Then pairs: user message (with JSON containing user_message and stage_info) + assistant message
        - Pattern: system-user-assistant-user-assistant-...

        Args:
            history: List of chat completion messages from existing session

        Returns:
            Tuple containing:
            - Dictionary mapping stages to their message history
            - Last stage encountered in the history (None if no valid stages found)

        Raises:
            ValueError: If history format is invalid or JSON parsing fails
        """
        if not history:
            return {}, None

        # Check if first message is system message
        if len(history) == 0 or history[0].get("role") != "system":
            raise ValueError("History must start with a system message")

        # Initialize result dictionary and last stage tracker
        stage_history = {}
        last_stage = None

        # Skip the system message and process user-assistant pairs
        i = 1
        while i < len(history):
            # Check for user message
            if i >= len(history) or history[i].get("role") != "user":
                self.logger.warning(
                    f"Expected user message at index {i}, got {history[i].get('role') if i < len(history) else 'end of history'}"
                )

            user_message = history[i]

            # Parse the JSON content from user message
            try:
                message_content = user_message.get("content", "")
                if not message_content:
                    raise ValueError(f"User message at index {i} has empty content")

                parsed_data = json.loads(message_content)

                # Extract required fields with error checking
                if "user_message" not in parsed_data:
                    self.logger.warning(
                        f"User message at index {i} missing 'user_message' field"
                    )

                if "stage_info" not in parsed_data:
                    self.logger.warning(
                        f"User message at index {i} missing 'stage_info' field"
                    )

                stage_info = parsed_data["stage_info"]
                if not isinstance(stage_info, dict):
                    self.logger.warning(
                        f"User message at index {i}: 'stage_info' must be a dictionary"
                    )

                if "current_stage" not in stage_info:
                    self.logger.warning(
                        f"User message at index {i} missing 'current_stage' in stage_info"
                    )

                stage_str = stage_info["current_stage"]

                # Convert string to Stage enum
                try:
                    stage = Stage(stage_str)
                    last_stage = stage  # Track the last valid stage encountered
                except ValueError:
                    raise ValueError(
                        f"Invalid stage '{stage_str}' at index {i}. Valid stages: {[s.value for s in Stage]}"
                    )

            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in user message at index {i}: {e}")

            # Check for assistant message
            if i + 1 >= len(history) or history[i + 1].get("role") != "assistant":
                self.logger.warning(
                    f"Expected assistant message at index {i + 1}, got {history[i + 1].get('role') if i + 1 < len(history) else 'end of history'}"
                )

            assistant_message = history[i + 1]

            # Initialize stage in dictionary if not exists
            if stage not in stage_history:
                stage_history[stage] = []

            # Create properly formatted messages for this stage (following handle_message_add format)
            stage_history[stage].append(
                {"role": "user", "message": parsed_data["user_message"]}
            )

            stage_history[stage].append(
                {"role": "assistant", "message": assistant_message.get("content", "")}
            )

            self.logger.debug(
                f"Processed message pair for stage {stage}: user_message='{parsed_data['user_message'][:50]}...'"
            )

            # Move to next pair
            i += 2

        self.logger.info(
            f"Processed existing history: {len(history)} total messages into {len(stage_history)} stages"
        )

        return stage_history, last_stage

    def __init__(
        self,
        endpoint: str,
        deployment: str,
        api_key: str,
        api_version: str,
        logger: logging.Logger,
        stage_config: StageConfig,
        initial_stage: Stage = Stage.PRE_CONTEMPLATION,
        stage_history: dict[Stage, list[ChatCompletionMessageParam]] | None = None,
    ):
        """
        Initializes the stage manager.

        Args:
            endpoint (str): The endpoint of the AI client.
            deployment (str): The deployment of the AI client.
            api_key (str): The API key of the AI client.
            api_version (str): The API version of the AI client.
            logger (logging.Logger): The logger to use for logging.
            stage_config (StageConfig): The stage config mapping stages to their evaluation configurations.
            initial_stage (Stage): The initial stage.
            message_index (int): The index of the first message, defaults to 0.
        """

        self.stage_config = stage_config
        self.current_stage = initial_stage
        self.logger = logger

        # Get stage info (descriptions) from stages.json
        stage_path = os.path.join(os.path.dirname(__file__), "include", "stages.json")
        try:
            with open(stage_path, "r") as f:
                self.stage_info = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Stage info file not found at {stage_path}")
            raise FileNotFoundError(f"Stage info file not found at {stage_path}")
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in stage info file at {stage_path}")
            raise json.JSONDecodeError(
                f"Invalid JSON in stage info file at {stage_path}"
            )
        except Exception as e:
            raise Exception(
                f"Unknown error loading stage info file at {stage_path}: {e}"
            )

        # Used for checking if stage should be advanced
        self.evaluator = Evaluator(
            endpoint=endpoint,
            deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            logger=logger,
        )

        if stage_history is None:
            # Stage tracking
            self.stage_history = {initial_stage: []}
        else:
            self.logger.info(
                f"Existing history found with length {len(stage_history)} messages"
            )

            processed_history, last_stage = self._process_existing_history(
                stage_history
            )
            self.logger.info(f"Processed history with length: {len(processed_history)}")
            self.stage_history = processed_history

            # If we have a valid last stage from history, update current_stage
            if last_stage is not None:
                self.current_stage = last_stage

        self.logger.info(
            f"Initialized <StageManager> with {len(self.stage_config)} stages, starting at {self.current_stage}, stage history: {self.stage_history}"
        )

    def get_stage_info(self) -> list[str]:
        """
        Returns the stage info for the current stage.
        """
        # Check if stage info exists
        if self.current_stage.value not in self.stage_info:
            raise ValueError(f"Stage info not found for stage {self.current_stage}")

        stage_info = self.stage_info[self.current_stage.value]

        if not isinstance(stage_info, list):
            self.logger.warning(
                f"Stage info for stage {self.current_stage} is not a list"
            )

        return {
            "current_stage": self.current_stage.value,
            "stage_info": stage_info,
        }

    def handle_message_add(self, user_message: str, response: str) -> None:
        """
        Handles the addition of a message to the current stage.

        Args:
            message (str): The message to add to the current stage.
        """
        # Add the message to the stage history
        current_stage = self.stage_history[self.current_stage]
        current_stage.append(
            {
                "role": "user",
                "message": user_message,
            }
        )
        current_stage.append(
            {
                "role": "assistant",
                "message": response,
            }
        )

        # Evaluate the stage and advance the stage if the objective is completed
        result = self.evaluate_stage()
        if result is None:
            self.logger.error(
                f"[StageManager] Invalid response from <Evaluator> for stage {self.current_stage}"
            )
        elif result is True or "[[ADMIN: FORCE OPEN]]" in user_message:
            self.logger.info(
                f"Stage {self.current_stage} completed, advancing to {self.current_stage.next_stage()}"
            )
            self.advance_stage()

    def advance_stage(self) -> bool:
        """
        Advances to the next stage in the progression.

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

        # Create new stage entry
        self.stage_history[next_stage] = []

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

        self.logger.debug(
            f"Evaluating stage {self.current_stage} with evaluation type: {evaluation_config.method}"
        )

        current_stage_history = self.stage_history[self.current_stage]

        result = None
        if evaluation_config.method == EvaluationMethods.OBJECTIVE_COMPLETION:
            result = self.evaluator.evaluate_objective_completion(
                evaluation_config.data,
                current_stage_history,
            )
        elif evaluation_config.method == EvaluationMethods.TABLE_COMPARISON:
            result = self.evaluator.evaluate_table_comparison(
                evaluation_config.data,
                current_stage_history,
            )
        else:
            raise ValueError(
                f"Invalid/unsupported evaluation method: {evaluation_config.method}"
            )

        return result

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
