# SkillHandle: Motivational Interviewing Skill Evaluation System

SkillHandle is a Python class designed to evaluate conversational skills and determine when to initiate Motivational Interviewing (MI) sessions. It processes skill count data through normalization, weighted scoring, and threshold-based decision making across three MI stages.

## File Structure

### Core Files

1. **`skihandle.py`** - Main class implementation

   - Contains the `SkillHandle` class with all core functionality
   - Handles YAML loading, skill normalization, and MI decision logic

2. **`skills.yaml`** - Configuration file (REQUIRED)

   - Defines skill definitions with IDs and names
   - Contains stage-specific importance scores
   - Must be paired with `skihandle.py`

3. **`temp`** - Test data file

   - JSON file containing sample skill count data
   - Used for testing and demonstration purposes
   - Format: `{"Skill Name": count, ...}`

4. **`test_skihandle.py`** - Testing script
   - Comprehensive test suite for `SkillHandle` functionality
   - Outputs normalized scores and final scores for each MI stage
   - Demonstrates edge case handling

## Usage

### Basic Usage

```python
from skihandle import SkillHandle

# Initialize with default skills.yaml
skill_handler = SkillHandle()

# Or specify custom YAML file
skill_handler = SkillHandle('path/to/custom_skills.yaml')

# Evaluate skills for MI decision
skill_counts = {
    "Active Listening": 3,
    "Empathy": 2,
    "Reflecting": 1,
    "Closed-Ended Questions": 4
}

# Get decision for stage 2 with threshold 0.6
score, should_start = skill_handler.signal_mi_start(
    skill_counts,
    stage=2,
    threshold=0.6
)

print(f"Score: {score:.3f}, Start MI: {should_start}")
```

### Critical Requirements

Skill names in the input dictionary must match exactly with names in `skills.yaml`

- Case-sensitive matching
- Exact spelling required
- Unknown skills are silently ignored

## Step-by-Step Formula

The system processes skill counts through the following pipeline:

### Step 1: Skill Count Normalization

```
1. Map skill names to IDs using YAML configuration
2. Create ID-indexed array: counts_by_id[skill_id] = count
3. Apply log transformation: log_counts = log(1 + counts)
4. Normalize by maximum: normalized = log_counts / max(log_counts)
```

### Step 2: Weighted Score Calculation

```
raw_score = dot_product(normalized_scores, stage_importance_weights)
```

### Step 3: Score Normalization to [0,1] Range

```
theoretical_max = sum(positive_weights)
theoretical_min = sum(negative_weights)
final_score = (raw_score - theoretical_min) / (theoretical_max - theoretical_min)
final_score = clip(final_score, 0.0, 1.0)
```

### Step 4: Decision Making

```
should_start_MI = final_score >= threshold
```

## MI Stages

- **Stage 1**: Engagement - Building rapport and trust
- **Stage 2**: Focusing - Identifying areas for change
- **Stage 3**: Evoking - Eliciting motivation for change

Each stage has different importance weights for skills, reflecting their relevance at that stage.

## Modifying skills.yaml

### Structure Requirements

```yaml
SKILLS:
  Skill_Key_Name:
    id: unique_integer
    name: "Exact Skill Name"
    importance: [stage1, stage2, stage3] # Not used in current implementation

IMPORTANCE_SCORES:
  stage_1: [weight_for_id_0, weight_for_id_1, ..., weight_for_id_N]
  stage_2: [weight_for_id_0, weight_for_id_1, ..., weight_for_id_N]
  stage_3: [weight_for_id_0, weight_for_id_1, ..., weight_for_id_N]
```

### Modification Guidelines

1. **DO NOT modify skill IDs** - IDs are used for array indexing
2. **DO NOT reorder skills** - Maintain existing skill definitions
3. **Skill names can be modified** - But update all references accordingly
4. **Importance scores order MUST correspond to skill ID order**
   - Index 0 in importance array = skill with ID 0
   - Index 1 in importance array = skill with ID 1
   - etc.

### Adding New Skills

```yaml
SKILLS:
  # Existing skills...

  New_Skill:
    id: 22 # Next available ID
    name: "New Skill Name"
    importance: [1, 0, -1] # Optional, not used

IMPORTANCE_SCORES:
  stage_1: [existing_weights..., new_weight] # Add weight at end
  stage_2: [existing_weights..., new_weight] # Add weight at end
  stage_3: [existing_weights..., new_weight] # Add weight at end
```

### Weight Guidelines

- **Positive weights**: Skills that support MI at this stage
- **Negative weights**: Skills that hinder MI at this stage
- **Zero weights**: Neutral skills for this stage
- **Typical range**: -1 to +1, but any float value is valid

## Testing

Run the test suite to verify functionality:

```bash
python test_skihandle.py
```

The test will output:

- Skill loading verification
- Normalization results for test data
- Scores and decisions for all stages
- Edge case handling (empty inputs, unknown skills)
- Detailed analysis by stage

## Example Output

```
Testing SkillHandle Class
============================================================
✓ SkillHandle initialized successfully
✓ Loaded 22 skills from YAML
✓ Loaded importance scores for 3 stages

Testing Skill Score Calculation:
----------------------------------------
✓ Stage 1: Score = 0.425, Start MI = False
✓ Stage 2: Score = 0.687, Start MI = True
✓ Stage 3: Score = 0.534, Start MI = True
```

## Error Handling

The system validates:

- Input dictionary format and types
- Skill count values (must be non-negative integers)
- Stage parameters (1, 2, or 3)
- Threshold values (0.0 to 1.0)
- YAML file structure and required sections

Unknown skills in input are silently ignored, allowing for flexible input sources.
