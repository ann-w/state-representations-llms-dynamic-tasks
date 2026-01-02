import logging
import re
from typing import Any, Callable, Dict, List, Tuple, Union

from scripts.utils.prompt_constants import PromptPaths
from scripts.utils.environment_utils import extract_summary_from_answer


def compose_ingame_prompt_with_summary(
    info: Dict[str, Any],
    rolling_summary: str,
    template_path: str,
    model_type: str = "openai",
) -> Union[List[Dict[str, str]], str]:
    """
    Compose a prompt using rolling summary instead of trajectory.
    """
    # Load the template as a string
    template = load_prompt_template(template_path)

    # Extract and optionally clean manual / obs for vision envs
    manual_text = info.get("manual", "No manual available")
    obs_text = info.get("obs", "No observation available")

    # Build the dictionary for formatting the template
    format_dict = {
        "manual": manual_text,
        "rolling_summary": rolling_summary,
        "obs": obs_text,
        "question": "What is the next action to take, let's think step by step",
    }

    # Render the prompt using string formatting
    prompt = template.format(**format_dict)

    if model_type not in [
        "openai",
        "huggingface",
        "local_huggingface",
        "ollama",
        "deepseek",
    ]:
        raise ValueError(
            "Unsupported model type. Choose 'openai', 'huggingface', 'local_huggingface', 'ollama', or 'deepseek'."
        )

    if model_type == "openai":
        messages = [{"role": "system", "content": prompt}]
        return messages
    else:
        return prompt


def generate_rolling_summary_refresh(
    info: Dict[str, Any],
    trajectory: List,
    previous_summary: str,
    query_model: Callable,
):
    """Call the LLM to compress recent trajectory into a fresh rolling summary."""

    if not trajectory:
        return previous_summary or "Start of game"

    template = load_prompt_template(PromptPaths.ROLLING_SUMMARY_REFRESH)

    manual_text = info.get("manual", "No manual available")
    formatted_history = format_history(trajectory)
    prev_summary_text = previous_summary or "Start of game"

    prompt = template.format(
        manual=manual_text,
        recent_history=formatted_history,
        previous_summary=prev_summary_text,
    )

    summary_response = query_model(prompt)
    if isinstance(summary_response, (list, tuple)):
        summary_text = summary_response[0]
    else:
        summary_text = summary_response

    if isinstance(summary_text, dict):
        summary_text = summary_text.get("content", "")

    # Prefer structured "Summary: ..." output
    summary_candidate = extract_summary_from_answer(summary_text)
    if summary_candidate and summary_candidate != "Game progressing":
        return summary_candidate

    cleaned = summary_text.strip()
    return cleaned if cleaned else prev_summary_text


def generate_agent_prompt_with_summary(
    info: Dict[str, Any],
    rolling_summary: str,
    model_type: str,
    use_visualization_of_thought: bool = False,
) -> str:
    """
    Generate agent prompt using rolling summary instead of full trajectory.
    """

    if use_visualization_of_thought:
        template_path = PromptPaths.AGENT_VISUALIZATION_OF_THOUGHT
    else:
        template_path = PromptPaths.AGENT_ROLLING_SUMMARY

    # Load template
    template = load_prompt_template(template_path)

    manual_text = info.get("manual", "No manual available")
    obs_text = info.get("obs", "No observation available")

    # Format prompt with rolling summary or trajectory based on template
    summary_text = rolling_summary or "No rolling summary yet."

    format_dict = {
        "manual": manual_text,
        "rolling_summary": summary_text,
        "trajectory": summary_text,
        "obs": obs_text,
        "question": "What is the next action to take?",
        "history_header": "Summary of past actions",
        "history_body": summary_text,
    }

    prompt = template.format(**format_dict)

    if model_type == "openai":
        return [{"role": "system", "content": prompt}]
    else:
        return prompt


def remove_thinking_sections(text: str) -> str:
    """
    Removes any text between <think> and </think> tags from model outputs.

    Args:
        text (str): The text potentially containing thinking sections.

    Returns:
        str: The cleaned text with thinking sections removed.
    """
    if text is None:
        return ""

    # Remove all content between <think> and </think> tags (including the tags)
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()


def format_history(history: List) -> str:
    """This function formats the history of the game in line with the original SmartPlay paper.

    Instead of a a list of tuples like in RL (s, a, r, s', d)
    ('A new round begins.', 'Pull slot machine 1.', -1, 'You pulled slot machine 1, you received reward -1.', False),

    The history is formatted like below in the paper:

    Player Observation Step {step number}: {observation}, you received reward {reward}

    """

    formatted_history = []
    for i, (observation, action, reward, next_state, done) in enumerate(history):
        formatted_history.append(f"Player Observation Step {i + 1}:\n{observation}")
    return "\n\n".join(formatted_history)


def load_prompt_template(template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as file:
        template = file.read()
    return template


def compose_ingame_prompt(
    info: Dict[str, Any],
    trajectory: List,
    template_path: str,
    model_type: str = "openai",
    use_history_format_paper: bool = True,
) -> Union[List[Dict[str, str]], str]:
    """
    Composes a prompt for the LLM to answer a question in the context of a game.

    Args:
        info (Dict[str, Any]): The game information.
        trajectory (List[Tuple[str, int, int, str, int]]): The trajectory of the game.
        template_path (str): The path to the template file.
        model_type (str): The type of model (e.g., "openai" or "huggingface").
        format_history (bool): Whether to format the history like in the SmartPlay paper.

    Returns:
        Union[List[Dict[str, str]], str]: The prompt as a list of messages for OpenAI API or as a string for Hugging Face API

    """

    # Whether to format the history like in the SmartPlay paper
    if use_history_format_paper:
        trajectory = format_history(trajectory)

    # Load the template as a string
    template = load_prompt_template(template_path)

    manual_text = info.get("manual", "No manual available")
    obs_text = info.get("obs", "No observation available")

    # Build the dictionary for formatting the template
    format_dict = {
        "manual": manual_text,
        "trajectory": trajectory,
        "obs": obs_text,
        "question": "What is the next action to take, let's think step by step",
    }

    # Render the prompt using string formatting
    prompt = template.format(**format_dict)

    if model_type not in [
        "openai",
        "huggingface",
        "local_huggingface",
        "ollama",
        "deepseek",
    ]:
        raise ValueError(
            "Unsupported model type. Choose 'openai', 'huggingface', 'local_huggingface', 'ollama', or 'deepseek'."
        )

    if model_type == "openai":
        messages = [{"role": "system", "content": prompt}]
        return messages

    else:
        return prompt


def generate_agent_prompt(
    info: Dict[str, Any],
    trajectory: Union[str, List],
    model_type: str,
    use_visualization_of_thought: bool = False,
) -> str:
    """
    Generate agent prompt using full trajectory.
    """

    if use_visualization_of_thought:
        template_path = PromptPaths.AGENT_VISUALIZATION_OF_THOUGHT
    else:
        template_path = PromptPaths.AGENT_BASIC

    # Load template
    template = load_prompt_template(template_path)

    manual_text = info.get("manual", "No manual available")
    obs_text = info.get("obs", "No observation available")

    # Format prompt
    rolling_summary_fallback = (
        "Rolling summary not enabled (full trajectory mode)."
    )

    if isinstance(trajectory, list):
        formatted_trajectory = format_history(trajectory)
    else:
        formatted_trajectory = trajectory or "No trajectory available."

    format_dict = {
        "manual": manual_text,
        "obs": obs_text,
        "rolling_summary": rolling_summary_fallback,
        "trajectory": formatted_trajectory,
        "question": "What is the next action to take?",
        "history_header": "Past trajectory",
        "history_body": formatted_trajectory,
    }

    prompt = template.format(**format_dict)

    if model_type == "openai":
        return [{"role": "system", "content": prompt}]
    else:
        return prompt
