import logging
import os
from typing import Optional, Union

import numpy as np
import wandb
from smartplay.utils.recording import save_gif_from_paths


def initialize_wandb_experiment(env_name: str, config: dict):
    """Initialize a W&B run with a consistent, explicit naming scheme."""

    vision_enabled = bool(config.get("vision"))
    use_visualization_of_thought = bool(config.get("use_visualization_of_thought"))
    use_oracle_summary = bool(config.get("use_oracle_summary"))
    use_rolling_summary = bool(config.get("use_rolling_summary"))

    # Derive display env name with Vision suffix for specific classic envs (non *Image variants)
    display_env_name = env_name
    if vision_enabled:
        # Classic 3-disk Hanoi (avoid Image variant which already implies vision)
        if env_name.startswith("Hanoi3Disk") and not env_name.endswith("Image"):
            display_env_name = f"{env_name}Vision"
        # MessengerL1 variants (Matrix/Symbolic/NaturalLanguage etc.) excluding explicit Image variants
        elif env_name.lower().startswith("messengerl1") and "image" not in env_name.lower():
            display_env_name = f"{env_name}Vision"

    # Base name
    if use_visualization_of_thought:
        name = f"{display_env_name}_vot"
    else:
        name = f"{display_env_name}_default"

    # Oracle summary suffix (ablation study mode)
    if use_oracle_summary:
        name = f"{name}_oracle-summary"
    elif use_rolling_summary:
        name = f"{name}_rolling-summary"

    logging.info(
        "[W&B Naming] env_name=%s display_env_name=%s vision=%s vot=%s oracle_summary=%s rolling_summary=%s -> run_name=%s",
        env_name,
        display_env_name,
        vision_enabled,
        use_visualization_of_thought,
        use_oracle_summary,
        use_rolling_summary,
        name,
    )

    wandb.init(project="LLM_Agent", name=name, config=config)
    log_artifacts()


def log_artifacts(prompts_dir: str = "src/prompts"):
    run = wandb.run

    # Create a new artifact
    artifact = wandb.Artifact(name="prompts_artifact", type="dataset")

    # Add files to the artifact
    for filename in os.listdir(prompts_dir):
        file_path = os.path.join(prompts_dir, filename)
        if os.path.isfile(file_path):
            artifact.add_file(file_path, name=filename)

    # Log the artifact
    run.log_artifact(artifact)


def create_episode_table():
    """
    Create a new WandB table for an episode.
    """
    columns = [
        "step",
        "prompt",
        "observation",
        "score",
        "reward",
        "total_reward",
        "answer",
        "action_index",
        "action",
        "invalid_action",
        "input_tokens",
        "output_tokens",
        # New vision + full prompt columns appended for backward compatibility
        "full_prompt",
        "vision_images",
    ]
    return wandb.Table(columns=columns)


def log_episode_metrics(
    step: int,
    prompt: str,
    full_prompt: str | None,
    observation: str,
    score: float,
    reward: float,
    total_reward: float,
    answer: str,
    action_index: int,
    action: str,
    invalid_action: bool,
    wandb_table: wandb.Table,
    input_tokens: Optional[Union[int, str]] = None,
    output_tokens: Optional[Union[int, str]] = None,
    model_type: Optional[str] = None,
    vision_image_paths: Optional[list[str]] = None,
):
    """
    Log metrics for each step of the episode.
    """

    # Clean up for openai
    if model_type == "openai":
        prompt = prompt[0].get("content")

    # Store vision images as a semicolon-separated path string to avoid mixed-type column issues in WandB Tables.
    vision_images_cell: Union[str, list] = ""
    if vision_image_paths:
        existing_paths = [p for p in vision_image_paths if isinstance(p, str) and os.path.exists(p)]
        if existing_paths:
            vision_images_cell = ";".join(existing_paths)

    new_row = [
        step,
        prompt,
        observation,
        score,
        reward,
        total_reward,
        answer,
        action_index,
        action,
        invalid_action,
        str(input_tokens) if input_tokens is not None else "",
        str(output_tokens) if output_tokens is not None else "",
        full_prompt if full_prompt is not None else prompt,
        vision_images_cell,
    ]
    wandb_table.add_data(*new_row)


def plot_rewards_to_wandb(
    rewards: list,
    plot_id: str = "rewards_plot",
    title: str = "Rewards vs Timesteps",
    folder_path: str = None,
):
    """
    Plots a list of rewards to Weights and Biases (wandb) in a specified folder structure.

    Args:
        rewards (list): A list of rewards to plot.
        eps (int): The episode number.
        plot_id (str): The identifier for the plot in wandb.
        title (str): The title of the plot.
    """
    # Create x-values (e.g., timesteps or indices)
    x_values = list(range(len(rewards)))

    # Pair x-values with rewards
    data = [[x, y] for (x, y) in zip(x_values, rewards)]

    # Create a wandb.Table
    table = wandb.Table(data=data, columns=["x", "y"])

    # Determine the folder path
    folder_path = folder_path if folder_path else ""

    # Log the line plot to wandb
    wandb.log(
        {f"{folder_path}{plot_id}": wandb.plot.line(table, "x", "y", title=title)}
    )


def finalize_episode_logging(
    episode_number: int, wandb_table: wandb.Table, rewards: list
):
    """
    Finalize logging for an episode, logging the table and summary graphs.
    """
    logging.info(f"Logging episode {episode_number} to WandB")

    wandb.log({f"episode_{episode_number}/rollout_table": wandb_table})

    plot_rewards_to_wandb(rewards, episode_number)


def generate_and_log_episode_gif(
    episode_number: int,
    frame_paths: list[str],
    env_name: str,
    gif_duration: float = 0.2,
    record_dir: str | None = None,
    tag: str = "vision",
) -> str | None:
    """Generate a GIF from provided frame paths and log it to WandB.

    Args:
        episode_number: Episode index.
        frame_paths: Ordered list of frame image paths.
        env_name: Environment name (used for naming).
        gif_duration: Frame duration.
        record_dir: Optional output directory (default under process_results/data/renders/<env>). 
        tag: Additional tag segment for filename grouping (e.g. 'vision').

    Returns:
        Absolute path to generated GIF or None if generation skipped/failed.
    """
    if not frame_paths:
        return None
    try:
        base_dir = record_dir or os.path.join("process_results", "data", "renders", env_name.lower())
        os.makedirs(base_dir, exist_ok=True)
        gif_name = f"{env_name}_episode_{episode_number}_{tag}.gif"
        out_path = os.path.join(base_dir, gif_name)
        save_gif_from_paths(frame_paths, out_path, duration=gif_duration)
        abs_path = os.path.abspath(out_path)
        # Log to wandb as video + artifact
        try:
            wandb.log({f"episode_{episode_number}/gif": wandb.Video(abs_path, fps=int(1/max(gif_duration,1e-3)), format="gif")})
        except Exception as e:
            logging.warning(f"Failed wandb video log for GIF {abs_path}: {e}")
        try:
            artifact = wandb.Artifact(name=f"episode_{episode_number}_gif", type="episode_gif")
            artifact.add_file(abs_path, name=gif_name)
            wandb.log_artifact(artifact)
        except Exception as e:
            logging.warning(f"Failed wandb artifact log for GIF {abs_path}: {e}")
        logging.info(f"[GIF] Saved & logged episode GIF: {abs_path}")
        return abs_path
    except Exception as e:
        logging.warning(f"Failed to generate episode GIF (episode={episode_number} env={env_name}): {e}")
        return None


def finalize_experiment_logging(
    normalized_scores, rewards
):
    """
    Finalizes the WandB run by plotting various metrics.

    Args:
        normalized_scores (list or np.array): Normalized scores per episode.
        rewards (list or np.array): Rewards per episode.
    """
    # Plot normalized scores
    plot_rewards_to_wandb(
        normalized_scores,
        plot_id="overall_normalized_scores",
        title="Normalized score per episode",
        folder_path="summary/",
    )

    # Plot cumulative rewards
    plot_rewards_to_wandb(
        np.cumsum(rewards),
        plot_id="cumulative_rewards",
        title="Cumulative rewards over episodes",
        folder_path="summary/",
    )

    # Plot rewards per episode
    plot_rewards_to_wandb(
        rewards,
        plot_id="episode_rewards",
        title="Rewards over episodes",
        folder_path="summary/",
    )

    # Finalize experiment logging
    wandb.finish()
