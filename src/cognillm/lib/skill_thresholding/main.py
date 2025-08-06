import yaml
import numpy as np
from typing import Dict, Tuple
import os


class SkillThresholding:
    """
    A class to handle skill evaluation and motivational interviewing (MI) session initiation.

    This class loads skill definitions and importance scores from a YAML configuration file,
    normalizes skill counts using log-transformation and various normalization methods,
    calculates weighted skill scores based on MI stage, and determines whether to start
    an MI session based on configurable thresholds.

    Attributes:
        skills (Dict[str, Dict]): Skill definitions from YAML file containing name, id, importance
        importance_scores (Dict[str, np.ndarray]): Stage-specific importance scores for each skill
        max_skill_id (int): Maximum skill ID for array sizing
        num_skills (int): Total number of skills loaded from configuration
        skill_references_mean (Optional[np.ndarray]): Mean reference values for per-skill normalization
        skill_references_std (Optional[np.ndarray]): Std reference values for per-skill normalization
    """

    def _load_yaml_skills(self, file_path: str) -> Tuple[Dict, Dict]:
        """
        Load skills and importance scores from a YAML configuration file.

        The YAML file should contain:
        - SKILLS: Dictionary of skill definitions with id, name, and importance arrays
        - IMPORTANCE_SCORES: Stage-specific importance weights for each skill

        Args:
            file_path (str): Path to the YAML configuration file

        Returns:
            Tuple[Dict, Dict]: (skills_dict, importance_scores_dict)

        Raises:
            FileNotFoundError: If the YAML file cannot be found
            yaml.YAMLError: If the YAML file is malformed
            ValueError: If required sections are missing
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("Invalid file path")

        with open(file_path, "r") as file:
            yaml_data = yaml.safe_load(file)

        if not yaml_data or not isinstance(yaml_data, dict):
            raise ValueError("Invalid YAML structure")

        if "SKILLS" not in yaml_data or "IMPORTANCE_SCORES" not in yaml_data:
            raise ValueError("Missing required YAML sections")

        skills_dict = yaml_data["SKILLS"]
        importance_scores_dict = yaml_data["IMPORTANCE_SCORES"]

        return skills_dict, importance_scores_dict

    def __init__(self, file_path: str | None = None):
        """
        Initialize the SkillHandle with skills from a YAML file.

        Args:
            file_path (str): Path to the YAML configuration file (default: 'skills.yaml')
        """
        if file_path is None:
            skill_path = os.path.join(os.path.dirname(__file__), "skills.yaml")
        else:
            skill_path = file_path

        # Load configuration from YAML file
        self.skills, self.importance_scores = self._load_yaml_skills(skill_path)

        # Cache frequently used values for performance
        self.max_skill_id = max(skill_data["id"] for skill_data in self.skills.values())
        self.num_skills = len(self.skills)

        # Create skill name to ID mapping for O(1) lookup
        self._skill_name_to_id = {
            skill_data["name"]: skill_data["id"] for skill_data in self.skills.values()
        }

    def get_yaml_params(self) -> Tuple[Dict, Dict]:
        """
        Get the loaded YAML parameters (skills and importance scores).

        Returns:
            Tuple[Dict, Dict]: (skills, importance_scores) from the YAML configuration
        """
        return self.skills, self.importance_scores

    def _log_transform_counts(self, counts: np.ndarray) -> np.ndarray:
        """
        Apply log transformation to skill counts: log(1 + count).

        Args:
            counts (np.ndarray): Raw skill counts

        Returns:
            np.ndarray: Log-transformed counts
        """
        return np.log1p(counts)  # log(1 + x) for numerical stability

    def _normalize_by_max(self, log_counts: np.ndarray) -> np.ndarray:
        """
        Normalize log-transformed counts by dividing by maximum value.

        Args:
            log_counts (np.ndarray): Log-transformed skill counts

        Returns:
            np.ndarray: Max-normalized scores in [0, 1] range
        """
        max_count = np.max(log_counts)
        if max_count > 0:
            return log_counts / max_count
        else:
            return np.zeros_like(log_counts)

    def _normalize_skill_count(self, skill_count: Dict[str, int]) -> np.ndarray:
        """
        Normalize skill counts to a 0-1 range using log transformation and normalization.

        This method:
        1. Maps skill names to their corresponding IDs
        2. Creates an array indexed by skill ID
        3. Applies log transformation: log(1 + count)
        4. Normalizes using reference values if available, otherwise max normalization

        Args:
            skill_count (Dict[str, int]): Dictionary mapping skill names to their counts

        Returns:
            np.ndarray: Normalized scores (0-1) for each skill, indexed by skill ID
                       Unknown skills are ignored, missing skills default to 0.0
        """
        if skill_count is None or not isinstance(skill_count, dict):
            raise ValueError("Invalid skill_count")

        # Initialize counts array with zeros for all possible skill IDs
        # Array size is max_skill_id + 1 to accommodate 0-based indexing
        counts_by_id = np.zeros(self.max_skill_id + 1, dtype=np.float64)

        # Map skill names to counts using pre-computed name-to-id mapping
        # Only process skills that exist in our configuration
        for skill_name, count in skill_count.items():
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"Invalid count for skill {skill_name}")
            if skill_name in self._skill_name_to_id:
                skill_id = self._skill_name_to_id[skill_name]
                counts_by_id[skill_id] = float(count)

        # Apply log transformation
        log_counts = self._log_transform_counts(counts_by_id)

        normalized_scores = self._normalize_by_max(log_counts)

        return normalized_scores

    def _calculate_skill_score(
        self, stage: int, normalized_scores: np.ndarray
    ) -> float:
        """
        Calculate weighted skill score using stage-specific importance weights.

        This method computes the dot product of normalized skill scores and their
        corresponding importance weights for the specified MI stage.

        Args:
            stage (int): The motivational interviewing stage (1, 2, or 3)
            normalized_scores (np.ndarray): Normalized skill scores (0-1) indexed by skill ID

        Returns:
            float: Raw weighted skill score (can be negative due to negative importance weights)

        Raises:
            ValueError: If normalized_scores length doesn't match importance_scores length
            KeyError: If the specified stage doesn't exist in importance_scores
        """
        stage_key = f"stage_{stage}"

        if stage_key not in self.importance_scores:
            raise KeyError(f"Stage {stage} not found in importance scores")

        stage_importance_weights = np.array(self.importance_scores[stage_key])

        if len(normalized_scores) != len(stage_importance_weights):
            raise ValueError(
                f"Normalized scores length ({len(normalized_scores)}) must match "
                f"importance scores length ({len(stage_importance_weights)}) for stage {stage}"
            )

        # Calculate weighted sum using numpy dot product
        raw_skill_score = np.dot(normalized_scores, stage_importance_weights)

        return float(raw_skill_score)

    def _normalize_score_to_unit_range(self, raw_score: float, stage: int) -> float:
        """
        Normalize the raw skill score to 0-1 range using theoretical min/max bounds.

        This method calculates the theoretical minimum and maximum possible scores
        for the given stage and maps the raw score to a 0-1 range.

        Args:
            raw_score (float): Raw weighted skill score
            stage (int): The motivational interviewing stage

        Returns:
            float: Normalized score in 0-1 range
        """
        stage_key = f"stage_{stage}"
        importance_weights = np.array(self.importance_scores[stage_key])

        # Calculate theoretical bounds
        theoretical_max = np.sum(importance_weights[importance_weights > 0])
        theoretical_min = np.sum(importance_weights[importance_weights < 0])

        # Handle edge case where all weights are zero
        if theoretical_max == theoretical_min:
            return 0.5

        # Map raw score from [theoretical_min, theoretical_max] to [0, 1]
        normalized_score = (raw_score - theoretical_min) / (
            theoretical_max - theoretical_min
        )

        # Clamp to [0, 1] range to handle any numerical edge cases
        normalized_score = np.clip(normalized_score, 0.0, 1.0)

        return float(normalized_score)

    def signal_mi_start(
        self, skill_count: Dict[str, int], stage: int = 1, threshold: float = 0.5
    ) -> Tuple[float, bool]:
        """
        Determine whether to start a motivational interviewing session.

        This method processes skill counts through the following pipeline:
        1. Normalize skill counts to 0-1 range using reference values (if available) or max normalization
        2. Calculate weighted score using stage-specific importance weights
        3. Normalize the weighted score to 0-1 range using theoretical bounds
        4. Compare normalized score against threshold to make decision

        Args:
            skill_count (Dict[str, int]): Dictionary mapping skill names to their counts
            stage (int): MI stage (1=engagement, 2=focusing, 3=evoking) (default: 1)
            threshold (float): Decision threshold in 0-1 range (default: 0.5)

        Returns:
            Tuple[float, bool]: (normalized_score, should_start_mi)
                - normalized_score: Final score in 0-1 range
                - should_start_mi: True if score >= threshold, False otherwise

        Example:
            >>> handler = SkillHandle()
            >>> # Optionally set reference values for better normalization
            >>> handler.set_skill_references(mean_values, std_values)
            >>> skills = {"Active Listening": 2, "Empathy": 3, "Reflecting": 1}
            >>> score, start = handler.signal_mi_start(skills, stage=2, threshold=0.6)
            >>> print(f"Score: {score:.3f}, Start MI: {start}")
        """
        if not isinstance(stage, int) or stage not in [1, 2, 3]:
            raise ValueError("Invalid stage")

        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            raise ValueError("Invalid threshold")

        # Step 1: Normalize skill counts to 0-1 range
        normalized_skill_scores = self._normalize_skill_count(skill_count)

        # Step 2: Calculate raw weighted score using stage-specific importance weights
        raw_weighted_score = self._calculate_skill_score(stage, normalized_skill_scores)

        # Step 3: Normalize the weighted score to 0-1 range
        final_normalized_score = self._normalize_score_to_unit_range(
            raw_weighted_score, stage
        )

        # Step 4: Make decision based on threshold
        should_start_mi = final_normalized_score >= threshold

        return final_normalized_score, should_start_mi
