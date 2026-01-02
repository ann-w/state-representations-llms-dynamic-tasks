import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from scripts.episode_runner import run_single_episode
from scripts.memory import Memory
from scripts.utils.environment_utils import initialize_environment

# Set up logging
logging.basicConfig(level=logging.INFO)


def run_single_episode_with_logging(
    env,
    env_name,
    env_steps,
    query_model,
    model_type,
    global_step,
    iteration,
    memory,
    count_token_usage,
    use_rolling_summary=False,
    use_oracle_summary=False,
    vision: bool = False,
    vision_log_frames: bool = True,
    use_visualization_of_thought: bool = False,
):
    normalized_score, episode_rewards, score = run_single_episode(
        env=env,
        env_name=env_name,
        env_steps=env_steps,
        query_model=query_model,
        model_type=model_type,
        global_step=global_step,
        episode_number=iteration,
        memory=memory,
        count_token_usage=count_token_usage,
        use_rolling_summary=use_rolling_summary,
        use_oracle_summary=use_oracle_summary,
        vision=vision,
        vision_log_frames=vision_log_frames,
        use_visualization_of_thought=use_visualization_of_thought,
    )
    return normalized_score, episode_rewards, score


def run_agent_on_environment(
    env_name: str,
    query_model: Callable,
    env_steps: int = None,
    num_iter: int = None,
    model_type: str = "huggingface",
    memory_capacity: int = 10000,
    count_token_usage: bool = True,
    use_rolling_summary: bool = False,  # Use rolling summary instead of trajectory
    use_oracle_summary: bool = False,  # Use ground-truth oracle summary instead of LLM-generated
    hanoi_state_representation: str | None = None,
    vision: bool = False,
    vision_log_frames: bool = True,
    one_shot_example: bool = False,
    use_visualization_of_thought: bool = False,
) -> Tuple[float, List[float], List[str]]:

    # Initialize variables
    normalized_scores = []
    rewards = []
    scores = []
    global_step = 0

    # Initialize the environment
    env, env_steps, num_iter = initialize_environment(
        env_name,
        env_steps,
        num_iter,
        include_one_shot_example=one_shot_example,
    )
    base_env = getattr(env, "unwrapped", env)

    # If Hanoi and an explicit representation is requested, set it
    try:
        if env_name.startswith("Hanoi") and hanoi_state_representation:
            base_env.set_state_representation(hanoi_state_representation)
    except Exception:
        pass

    for iteration in tqdm(range(num_iter), desc=f"Running {env_name}"):
        memory = Memory(capacity=memory_capacity)

        logging.info(f"Running episode {iteration}.")
        normalized_score, episode_rewards, score = run_single_episode_with_logging(
            env,
            env_name,
            env_steps,
            query_model,
            model_type,
            global_step,
            iteration,
            memory,
            count_token_usage,
            use_rolling_summary,
            use_oracle_summary,
            vision,
            vision_log_frames,
            use_visualization_of_thought,
        )
        normalized_scores.append(normalized_score)
        rewards.append(episode_rewards)
        scores.append(score)
        global_step += env_steps

    return normalized_scores, rewards, scores
