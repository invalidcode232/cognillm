import os
import json
import logging
from typing import List
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from ...base.ai import Client
from ..types import EvaluationMethods, TableData


class Evaluator:
    """
    Base class for evaluators.
    """

    def __init__(
        self,
        endpoint: str,
        deployment: str,
        api_key: str,
        api_version: str,
        logger: logging.Logger,
    ):
        """
        Initialize the Evaluator.

        Args:
            endpoint (str): The endpoint of the AI client.
            deployment (str): The deployment of the AI client.
            api_key (str): The API key of the AI client.
            api_version (str): The API version of the AI client.
            logger (logging.Logger): The logger to use for logging.
        """
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts")
        prompt_names = [
            EvaluationMethods.OBJECTIVE_COMPLETION,
            EvaluationMethods.TABLE_COMPARISON,
        ]
        self.ai_clients: dict[EvaluationMethods, Client] = {}

        # Validate and load prompts
        for prompt_name in prompt_names:
            prompt_path = os.path.join(prompt_path, f"{prompt_name.value}.txt")
            if not os.path.exists(prompt_path):
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
            with open(prompt_path, "r") as file:
                self.ai_clients[prompt_name] = Client(
                    system_prompt=file.read(),
                    endpoint=endpoint,
                    deployment=deployment,
                    api_key=api_key,
                    api_version=api_version,
                )

        self.logger = logger

        self.logger.info(f"Initialized <Evaluator> with {len(self.ai_clients)} clients")

    def _get_eval_result(self, response: str) -> bool | None:
        """
        Validate the response from the AI.

        Args:
            response (str): The response from the AI.

        Returns:
            bool | None: True if the objective is completed, False otherwise, will return None if the response is invalid.
        """
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON response: {response}")
            return None

        if not response.get("chain_of_thought"):
            self.logger.warning(f"No chain of thought in response: {response}")

        if not response.get("result"):
            self.logger.error(f"No result in response: {response}")
            return None

        return response.get("result")

    def evaluate_objective_completion(
        self, objective: List[str], history: list[ChatCompletionMessageParam]
    ) -> bool | None:
        """
        Evaluate the completion of a stage using the Objective Completion method.

        Args:
            objective (str): The objective to evaluate.
            history (list[ChatCompletionMessageParam]): The history of the conversation.

        Returns:
            bool | None: True if the objective is completed, False otherwise, will return None if the response is invalid.
        """

        try:
            objective_str = json.dumps(objective)
            history_str = json.dumps(history)
        except Exception as e:
            self.logger.error(f"Error converting objective or history to string: {e}")
            raise ValueError(f"Error converting objective or history to string: {e}")

        eval_prompt = f"""
        Objective: {objective_str}
        History: {history_str}
        """

        # self.logger.info(f"Evaluating objective completion: {eval_prompt}")

        ai_client = self.ai_clients[EvaluationMethods.OBJECTIVE_COMPLETION]
        response, _ = ai_client.send_message(eval_prompt)

        eval_result = self._get_eval_result(response)

        # INFO: We are resetting this conversation in order to save context tokens
        # Since the clients are merely summarizers for the relevant message array(s),
        # we will not need the history anymore.
        ai_client.reset_conversation()

        return eval_result

    def evaluate_table_comparison(
        self, table: TableData, history: list[ChatCompletionMessageParam]
    ) -> bool | None:
        """
        Evaluate the completion of a stage using the Table Comparison method.
        """
        raise NotImplementedError("Table Comparison method not implemented")
