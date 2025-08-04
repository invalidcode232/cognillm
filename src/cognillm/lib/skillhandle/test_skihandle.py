import json
from skihandle import SkillHandle
import os


def load_test_data(file_path: str | None = None):
    """Load test skill count data from JSON file."""

    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), "temp.json")

    with open(file_path, "r") as file:
        return json.load(file)


def test_skill_handle():
    """Test the SkillHandle class functionality."""
    print("=" * 60)
    print("Testing SkillHandle Class")
    print("=" * 60)

    # Initialize SkillHandle
    try:
        skill_handler = SkillHandle()
        print("✓ SkillHandle initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize SkillHandle: {e}")
        return

    # Test YAML loading
    try:
        skills, importance_scores = skill_handler.get_yaml_params()
        print(f"✓ Loaded {len(skills)} skills from YAML")
        print(f"✓ Loaded importance scores for {len(importance_scores)} stages")
    except Exception as e:
        print(f"✗ Failed to load YAML parameters: {e}")
        return

    # Load test data
    try:
        test_skill_count = load_test_data()
        print(f"✓ Loaded test data with {len(test_skill_count)} skill counts")
    except Exception as e:
        print(f"✗ Failed to load test data: {e}")
        return

    print("\n" + "-" * 40)
    print("Test Data:")
    print("-" * 40)
    for skill, count in test_skill_count.items():
        if count > 0:  # Only show skills with counts > 0
            print(f"  {skill}: {count}")

    # Test normalization
    print("\n" + "-" * 40)
    print("Testing Normalization:")
    print("-" * 40)
    try:
        normalized_scores = skill_handler._normalize_skill_count(test_skill_count)
        print(f"✓ Normalized scores calculated (length: {len(normalized_scores)})")

        # Show normalized scores for skills with counts > 0
        skill_names = [skill_data["name"] for skill_data in skills.values()]
        print("\nNormalized scores for skills with counts > 0:")
        for skill_name, count in test_skill_count.items():
            if count > 0 and skill_name in skill_names:
                skill_id = next(
                    skill_data["id"]
                    for skill_data in skills.values()
                    if skill_data["name"] == skill_name
                )
                print(
                    f"  {skill_name} (ID {skill_id}): {normalized_scores[skill_id]:.3f}"
                )

    except Exception as e:
        print(f"✗ Failed to normalize skill counts: {e}")
        return

    # Test skill score calculation for each stage
    print("\n" + "-" * 40)
    print("Testing Skill Score Calculation:")
    print("-" * 40)

    for stage in [1, 2, 3]:
        try:
            score, should_start = skill_handler.signal_mi_start(
                test_skill_count, stage=stage, threshold=0.5
            )
            print(f"✓ Stage {stage}: Score = {score:.3f}, Start MI = {should_start}")
        except Exception as e:
            print(f"✗ Failed to calculate score for stage {stage}: {e}")

    # Test with different thresholds
    print("\n" + "-" * 40)
    print("Testing Different Thresholds (Stage 2):")
    print("-" * 40)

    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
    for threshold in thresholds:
        try:
            score, should_start = skill_handler.signal_mi_start(
                test_skill_count, stage=2, threshold=threshold
            )
            print(
                f"  Threshold {threshold:.2f}: Score = {score:.3f}, Start MI = {should_start}"
            )
        except Exception as e:
            print(f"✗ Failed to test threshold {threshold}: {e}")

    # Test edge cases
    print("\n" + "-" * 40)
    print("Testing Edge Cases:")
    print("-" * 40)

    # Empty skill count
    try:
        empty_skill_count = {}
        score, should_start = skill_handler.signal_mi_start(empty_skill_count, stage=1)
        print(f"✓ Empty skill count: Score = {score:.3f}, Start MI = {should_start}")
    except Exception as e:
        print(f"✗ Failed with empty skill count: {e}")

    # Unknown skill names
    try:
        unknown_skills = {"Unknown Skill": 5, "Another Unknown": 3}
        score, should_start = skill_handler.signal_mi_start(unknown_skills, stage=1)
        print(f"✓ Unknown skills: Score = {score:.3f}, Start MI = {should_start}")
    except Exception as e:
        print(f"✗ Failed with unknown skills: {e}")

    # Mixed known and unknown skills
    try:
        mixed_skills = {
            "Active Listening": 2,
            "Unknown Skill": 5,
            "Empathy": 3,
            "Another Unknown": 1,
        }
        score, should_start = skill_handler.signal_mi_start(mixed_skills, stage=2)
        print(f"✓ Mixed skills: Score = {score:.3f}, Start MI = {should_start}")
    except Exception as e:
        print(f"✗ Failed with mixed skills: {e}")


def detailed_analysis():
    """Provide detailed analysis of the test data."""
    print("\n" + "=" * 60)
    print("Detailed Analysis")
    print("=" * 60)

    skill_handler = SkillHandle()

    test_skill_count = load_test_data()

    print("\nSkill Importance by Stage:")
    print("-" * 30)

    skills, importance_scores = skill_handler.get_yaml_params()

    for stage in [1, 2, 3]:
        print(f"\nStage {stage}:")
        stage_scores = importance_scores[f"stage_{stage}"]

        # Get skills sorted by importance for this stage
        skill_importance = []
        for skill_key, skill_data in skills.items():
            skill_id = skill_data["id"]
            skill_name = skill_data["name"]
            importance = stage_scores[skill_id]
            count = test_skill_count.get(skill_name, 0)
            skill_importance.append((skill_name, importance, count, skill_id))

        # Sort by importance (descending) then by count (descending)
        skill_importance.sort(key=lambda x: (-x[1], -x[2]))

        print("  Top positive importance skills:")
        for skill_name, importance, count, skill_id in skill_importance[:10]:
            if importance > 0:
                print(f"    {skill_name}: importance={importance}, count={count}")

        print("  Negative importance skills with counts:")
        for skill_name, importance, count, skill_id in skill_importance:
            if importance < 0 and count > 0:
                print(f"    {skill_name}: importance={importance}, count={count}")


if __name__ == "__main__":
    test_skill_handle()
    detailed_analysis()
