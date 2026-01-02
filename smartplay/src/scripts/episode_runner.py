import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
from smartplay.eval import normalize_score

from scripts.compose_ingame_prompt import (
    generate_agent_prompt,
    generate_agent_prompt_with_summary,
    generate_rolling_summary_refresh,
)
from scripts.constants import ENV_TRAJECTORY_SAMPLES
from scripts.memory import Memory
from scripts.utils.environment_utils import (
    convert_parsed_action_to_valid_index,
    extract_summary_from_answer,
    parse_action_number,
)
from scripts.utils.timeit import timeit
from scripts.utils.vision_utils import (
    initialize_episode_vision,
    capture_post_step_frame,
    finalize_episode_vision,
)

from scripts.utils.wandb_logging import (
    create_episode_table,
    finalize_episode_logging,
    log_episode_metrics,
)


@timeit
def run_single_episode(
    env,
    env_name: str,
    env_steps: int,
    query_model: Callable,
    model_type: str,
    memory: Memory,
    global_step: int,
    episode_number: Optional[int] = None,
    count_token_usage: bool = False,
    use_rolling_summary: bool = False,
    use_oracle_summary: bool = False,
    vision: bool = False,
    vision_log_frames: bool = True,
    use_visualization_of_thought: bool = False,
) -> Tuple[float, List[float]]:

    rewards = []
    _, info = env.reset()

    action = None

    # Initialize the memory if it is not already initialized
    # Oracle summary mode uses rolling summary template but with ground-truth state
    if use_rolling_summary or use_oracle_summary:
        memory.reset_rolling_summary()

    # Get the last n transitions from the memory
    if env_name not in ENV_TRAJECTORY_SAMPLES:
        raise ValueError(f"Environment {env_name} not found in env_history_samples")

    # Get the number of trajectory samples from the dictionary
    n_trajectory_samples = ENV_TRAJECTORY_SAMPLES[env_name]

    episode_table = create_episode_table()

    logging.info("n_trajectory_samples for history: %s", n_trajectory_samples)

    # Initialize vision episode lifecycle (creates directories & captures initial frame)
    episode_render_dir, collected_vision_frames, image_paths = initialize_episode_vision(
        env=env,
        env_name=env_name,
        vision=vision,
        episode_number=episode_number,
        initial_state=info["obs"],
    )

    for step_number in range(env_steps):
        agent_prompt = ""
        answer = ""
        num_input_tokens = 0
        num_output_tokens = 0
        action = None
        invalid_action = False

        state = info["obs"]

        trajectory = memory.get_transitions(n=n_trajectory_samples)

        if use_oracle_summary:
            # Use ground-truth oracle summary from the environment
            # This bypasses LLM summarization for ablation studies
            base_env = getattr(env, "unwrapped", env)
            if hasattr(base_env, "get_oracle_summary"):
                rolling_summary = base_env.get_oracle_summary()
                memory.update_rolling_summary(rolling_summary)
            else:
                logging.warning(
                    f"Oracle summary requested but {env_name} does not support get_oracle_summary(). "
                    "Falling back to 'Start of game'."
                )
                rolling_summary = memory.get_rolling_summary()
        elif use_rolling_summary:
            # Refresh rolling summary using actual history before constructing prompts
            if trajectory:
                refreshed_summary = generate_rolling_summary_refresh(
                    info=info,
                    trajectory=trajectory,
                    previous_summary=memory.get_rolling_summary(),
                    query_model=query_model,
                )
                if refreshed_summary:
                    memory.update_rolling_summary(refreshed_summary)

            rolling_summary = memory.get_rolling_summary()
        else:
            rolling_summary = None

        # For oracle summary, we use the rolling summary template but skip LLM-based summary updates
        use_summary_mode = use_rolling_summary or use_oracle_summary

        # Standard agent execution path
        if use_summary_mode:
            agent_prompt = generate_agent_prompt_with_summary(
                info=info,
                rolling_summary=rolling_summary,
                model_type=model_type,
                use_visualization_of_thought=use_visualization_of_thought,
            )
        else:
            agent_prompt = generate_agent_prompt(
                info=info,
                trajectory=trajectory,
                model_type=model_type,
                use_visualization_of_thought=use_visualization_of_thought,
            )

        # Query the model
        try:
            if image_paths:
                answer, num_input_tokens, num_output_tokens = query_model(
                    agent_prompt, count_token_usage, images=image_paths
                )
            else:
                answer, num_input_tokens, num_output_tokens = query_model(
                    agent_prompt, count_token_usage
                )
        except TypeError:
            # Backward compatibility: older query_model without images param
            answer, num_input_tokens, num_output_tokens = query_model(
                agent_prompt, count_token_usage
            )

        # Parse action using existing function
        parsed_action_number = parse_action_number(answer, environment=env_name)
        action_index, invalid_action = convert_parsed_action_to_valid_index(
            parsed_action_number, env
        )

        # If using rolling summary (but NOT oracle summary), extract and update the summary
        if use_rolling_summary and not use_oracle_summary:
            new_summary = extract_summary_from_answer(answer)
            if (
                new_summary
                and new_summary.strip()
                and new_summary not in {"Game progressing", "Game in progress"}
            ):
                memory.update_rolling_summary(new_summary)
            else:
                logging.info(
                    "Agent response summary not applied (empty or placeholder)."
                )

        action = env.action_list[action_index]

        # Step environment
        _, reward, done, info = env.step(action_index)
        score = info["score"]

        next_state = info["obs"]
        done_bool = True if done == 1 else False

        memory.store_transition((state, action, reward, next_state, done_bool))
        rewards.append(reward)

        log_episode_metrics(
            step=step_number,
            prompt=agent_prompt,
            full_prompt=agent_prompt,
            observation=state,
            score=score,
            reward=reward,
            total_reward=sum(rewards),
            answer=answer,
            action_index=action_index,
            action=action,
            invalid_action=invalid_action,
            wandb_table=episode_table,
            input_tokens=num_input_tokens if num_input_tokens else "",
            output_tokens=num_output_tokens if num_output_tokens else "",
            model_type=model_type,
            vision_image_paths=image_paths if vision else None,
        )

        # Post-step vision frame capture to reflect movement & updated HUD (step_number+1)
        if vision:
            post_step_paths = capture_post_step_frame(
                env=env,
                env_name=env_name,
                state=info["obs"],
                episode_number=episode_number,
                step_number=step_number + 1,
                episode_render_dir=episode_render_dir,
                collected_frames=collected_vision_frames,
            )
            if post_step_paths:
                image_paths = post_step_paths

        if done:
            # Ensure final frame already captured above
            break

    # Normalizes the score between 0 and 1 based on predefined minum and human scores for the game
    normalized_score = normalize_score(env_name, score)

    finalize_episode_logging(episode_number, episode_table, rewards)

    # Finalize vision (central logging + local GIF)
    finalize_episode_vision(
        env=env,
        env_name=env_name,
        episode_number=episode_number,
        collected_vision_frames=collected_vision_frames,
        vision=vision,
        episode_render_dir=episode_render_dir,
        tag="vision",
        create_local_gif=vision_log_frames,
    )

    # Post-upload pruning: after WandB table & (optional) GIF are created, reclaim disk if disabled
    if vision and not vision_log_frames and collected_vision_frames:
        try:
            import os
            # Keep only the final frame for minimal footprint
            keep = collected_vision_frames[-1]
            for fp in collected_vision_frames[:-1]:
                if fp != keep and os.path.exists(fp):
                    os.remove(fp)
            collected_vision_frames[:] = [keep]
            logging.info("[VisionPrune] Pruned earlier frames; retained last frame: %s", keep)
        except Exception as e:
            logging.debug(f"[VisionPrune] Frame pruning failed: {e}")

    logging.info(f"Episode statistics")
    logging.info(
        f"Total reward: {np.sum(rewards)}, Normalized score: {round(normalized_score, 2)}, Score: {score}"
    )

    return normalized_score, np.sum(rewards), score
