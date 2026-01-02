import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts.compose_ingame_prompt import (
    generate_agent_prompt,
    generate_agent_prompt_with_summary,
)


def _base_info():
    return {
        "manual": "Manual text about reaching the exit.",
        "obs": "Agent at center; goal north; key east.",
    }


def test_visualization_prompt_basic_mode_includes_map_instructions():
    prompt = generate_agent_prompt(
        info=_base_info(),
        trajectory="Past events show agent scouting.",
        model_type="ollama",
        use_visualization_of_thought=True,
    )

    assert "Visualization-of-Thought Protocol" in prompt
    assert "Past events show agent scouting." in prompt
    assert "Past trajectory" in prompt


def test_visualization_prompt_with_rolling_summary_uses_summary_text():
    prompt = generate_agent_prompt_with_summary(
        info=_base_info(),
        rolling_summary="Scouted south; need key.",
        model_type="ollama",
        use_visualization_of_thought=True,
    )

    assert "Visualization-of-Thought Protocol" in prompt
    assert "Summary of past actions" in prompt
    assert "Scouted south; need key." in prompt
