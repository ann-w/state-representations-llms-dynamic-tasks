import argparse
import logging
import os

import numpy as np
import smartplay
from dotenv import load_dotenv

from scripts.agent_runner import run_agent_on_environment
from scripts.utils.environment_utils import initialize_model
from scripts.utils.wandb_logging import (
    finalize_experiment_logging,
    initialize_wandb_experiment,
)
from scripts.utils.yaml_utils import load_experiment_settings, log_experiment_settings
from scripts.utils.summarize_results import (
    calculate_statistics,
    initialize_results_dicts,
    update_results,
)

"""This script is the entry point for running the agent on the environment. 
It loads the experiment settings from a YAML file, parses the command-line arguments, and runs the agent on the environment. 
The script also initializes the logging and the Weights and Biases (WandB) experiment."""


def str_to_bool(value):
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'yes', '1', 'on'):
        return True
    elif value.lower() in ('false', 'no', '0', 'off'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def parse_experiment_settings(experiment_settings: dict) -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=experiment_settings.get(
            "model_name", "microsoft/Phi-3-mini-4k-instruct"
        ),
        help="Name of the LLM",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=experiment_settings.get("api_type", "ollama"),
        choices=[
            "huggingface",
            "local_huggingface",
            "openai",
            "ollama",
            "deepseek",
        ],
        help="Type of the LLM API",
    )
    parser.add_argument(
        "--env_names",
        type=str,
        default=experiment_settings.get("env_names", "Hanoi3Disk"),
        help=("Comma separated list of environments to run"),
    )
    parser.add_argument(
        "--env_steps",
        type=int,
        default=experiment_settings.get("env_steps", None),
        help="Number of steps to run the environment for. If not provided, the default number of steps for the environment will be used.",
    )
    parser.add_argument(
        "--env_iter",
        type=int,
        default=experiment_settings.get("env_iter", None),
        help="Number of iterations/episodes to run the environment for. If not provided, the default number of iterations for the environment will be used.",
    )

    parser.add_argument(
        "--count_token_usage",
        default=experiment_settings.get("count_token_usage", False),
        action="store_true",  # by default False
        help="Count number of tokens used in each query",
    )
    parser.add_argument(
        "--use_oracle_summary",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=experiment_settings.get("use_oracle_summary", False),
        help="Use oracle (ground-truth) state summary instead of LLM-generated summary. "
             "This provides a perfect state representation for ablation studies.",
    )
    parser.add_argument(
        "--use_rolling_summary",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=experiment_settings.get("use_rolling_summary", False),
        help="Use rolling summary instead of full trajectory",
    )
    parser.add_argument(
        "--use_visualization_of_thought",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=experiment_settings.get("use_visualization_of_thought", False),
        help=(
            "Enable visualization-of-thought prompting so agents draw/update ASCII maps before choosing actions."
        ),
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        default=experiment_settings.get("vision", False),
        help=(
            "Enable vision-language mode (only supported for Ollama vision models like llava). When enabled, environment frames are captured and sent with prompts."
        ),
    )
    parser.add_argument(
        "--vision_log_frames",
        action="store_true",
        default=experiment_settings.get("vision_log_frames", True),
        help=(
            "Persist per-step vision frames & GIFs (default). Disable to keep only the latest frame on disk to reduce storage usage."
        ),
    )
    parser.add_argument(
        "--one_shot_example",
        action="store_true",
        default=experiment_settings.get("one_shot_example", False),
        help=(
            "Include an illustrative one-shot example in environment manuals when supported."
        ),
    )
    return parser.parse_args()


def set_default_values(args):
    if args.env_names is None:
        args.env_names = ",".join(smartplay.benchmark_games_v0)
    return args


def main():
    load_dotenv()

    # Set the working directory to the parent folder that is two levels up
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    os.chdir(parent_dir)

    # Load experiment settings
    experiment_settings = load_experiment_settings(
        "src/config/experiment_settings/experiment_settings.yaml"
    )

    logging.info("Loaded Experiment Settings:")
    log_experiment_settings(experiment_settings)

    # Parse arguments again with the loaded experiment settings
    args = parse_experiment_settings(experiment_settings)
    args = set_default_values(args)

    # initialize dicts
    results_dict, score_dict = initialize_results_dicts()

    # initialize model
    query_model = initialize_model(args.model_name, args.model_type, vision=args.vision)

    for env_name in args.env_names.split(","):
        env_name = env_name.strip()

        # Initialize WandB
        initialize_wandb_experiment(env_name=env_name, config=experiment_settings)

        normalized_scores, rewards, scores = run_agent_on_environment(
            env_name=env_name,
            query_model=query_model,
            env_steps=args.env_steps,
            num_iter=args.env_iter,
            model_type=args.model_type,
            count_token_usage=args.count_token_usage,
            use_rolling_summary=args.use_rolling_summary,
            use_oracle_summary=args.use_oracle_summary,
            use_visualization_of_thought=args.use_visualization_of_thought,
            vision=args.vision,
            vision_log_frames=args.vision_log_frames,
            one_shot_example=args.one_shot_example,
        )

        logging.info(
            f"[COMPLETED ENVIRONMENT EPISODES] env: {env_name}, average scores: mean: {np.mean(scores)} and std: {np.std(scores)}"
        )

        update_results(
            env_name, normalized_scores, rewards, scores, results_dict, score_dict
        )

        finalize_experiment_logging(
            normalized_scores, rewards
        )

    # Evaluate the capability scores of the LLM based on a given dictionary of scores for different environments
    # It returns a dictionary of capability scores for each environment
    # logging.info(f"Capability scores of the LLM:", smartplay.analyze_capabilities(score_dict))

    statistics = calculate_statistics(results_dict)
    for env_name, stats in statistics.items():
        logging.info(f"Statistics for {env_name}: {stats}")


if __name__ == "__main__":
    main()
