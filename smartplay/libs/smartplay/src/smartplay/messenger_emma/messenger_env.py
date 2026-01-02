"""Messenger environment wrapper (clean version) with multi-representations & GIF recording.

Adds:
 - force_start_without_message: optional reset guard
 - print_actions: per-step debug logging (positions & actions)
 - termination reason tagging (max_steps | success | fail)
 - cumulative score tracking for score overlay (renderer already draws Score)
"""

from pathlib import Path
import numpy as np
import gym
import messenger
import messenger.envs
from ..utils import HistoryTracker, describe_act

# ---------- Helper descriptions ----------
id_map = {e.id: e.name for e in messenger.envs.config.ALL_ENTITIES}
id_map[0] = "  "
id_map[15] = "you (agent) without the message"
id_map[16] = "you (agent) with the message"


def _describe_frame(obs):
    avatar = obs["avatar"]
    if 15 in np.unique(avatar):
        agent_id = 15; status = "You (agent) don't have the message."
    elif 16 in np.unique(avatar):
        agent_id = 16; status = "You (agent) already have the message."
    else:
        raise RuntimeError("Agent sprite missing")
    ent_map = obs["entities"]
    # Collapse depth if present (H,W,C) -> (H,W)
    if ent_map.ndim == 3:
        ent_2d = ent_map.max(axis=2)
    else:
        ent_2d = ent_map
    items = []
    for eid in np.unique(ent_2d):
        if eid in (0, 15, 16):
            continue
        ys, xs = np.where(ent_2d == eid)
        if ys.size == 0:
            continue
        ayx = np.where(avatar == agent_id)
        ay, ax = int(ayx[0][0]), int(ayx[1][0])
        dist = abs(int(ys[0]) - ay) + abs(int(xs[0]) - ax)
        items.append((id_map.get(eid, f"entity_{eid}"), dist))
    if items:
        listing = "You see:\n" + "\n".join([(f"- {n} {d} steps away" if d > 0 else f"- {n} 0 steps with you") for n, d in items])
    else:
        listing = "You see nothing away from you."
    return (status + "\n\n" + listing).strip()


class MessengerEnv(gym.Env):
    default_iter = 20
    default_steps = None

    @property
    def use_shaping(self):
        return getattr(self._env, "use_shaping", False)

    @property
    def use_text_substitution(self):
        return getattr(self._env, "use_text_substitution", True)

    def __init__(self, lvl: int = 1, max_steps: int = 10, env_noise: int = 0, representation: str = "Default",
                 record_gif: bool = False, record_dir=None, gif_duration: float = 0.2,
                 include_one_shot_example: bool = False, **kwargs):
        super().__init__()
        self.lvl = lvl
        self.env_noise = env_noise
        self.representation = representation
        self.include_one_shot_example = include_one_shot_example
        # Recording
        self.record_gif = record_gif
        self.record_dir = record_dir
        self.gif_duration = gif_duration
        self._frame_paths = []
        self._frame_output_dir = None
        self._episode_dir = None
        self._frame_index = 0
        self._last_image_path = None
        # Visual state
        self.enable_flashes = True
        self._flash_queue = []
        self._prev_agent_has_message = False
        self._enemy_collision_reveal = False
        self.debug_assets = kwargs.get("debug_assets", False)
        self._missing_variants = set()
        self._last_action = None
        self._current_score = 0.0
        self._last_reward = 0.0
        self._last_termination_reason = None
        self._show_message_persistent = False
        self.overlay_debug = kwargs.get("overlay_debug", False)
        self.force_start_without_message = kwargs.get("force_start_without_message", False)
        self.print_actions = kwargs.get("print_actions", False)
        self._prev_agent_xy = None
        self._just_done = False
        # Rendering safety
        self.max_render_pixels = kwargs.get("max_render_pixels", 40_000_000)
        self._force_tile_size = kwargs.get("tile_size")
        self.max_tile_pixels = kwargs.get("max_tile_pixels", 12_000_000)
        # Default steps per level (fallback)
        self.default_steps = [10, 64, 128][self.lvl - 1]
        # Base env
        env_id = f"msgr-test-v{self.lvl}"
        if self.lvl == 1:
            self._env = gym.make(
                env_id,
                use_shaping=kwargs.get("use_shaping", False),
                use_text_substitution=kwargs.get("use_text_substitution", True),
            )
        else:
            self._env = gym.make(env_id)
        # Actions
        self.action_list = ["Move North", "Move South", "Move West", "Move East", "Do Nothing"]
        # History / limits
        self.history = HistoryTracker(max_steps)
        self.step_limit = max_steps
        self._action_steps = 0
        # Context text
        self.game_context = (
            "In the game, MESSENGER, each entity can take on one of three roles: an enemy, message, or goal. "
            "The agent’s objective is to bring the message to the goal while avoiding the enemies. If the agent encounters an enemy at any point in the game, or the goal without first obtaining the message, it loses the game and obtains a reward of −1."
        )
        self.advice = (
            "To solve a game, you may find it helpful to list the objects that you see. Then for each object, match it with an entity description, and identify whether it is good or bad to interact with the object.\n"
            "The name specifications of in-game objects may not be exact matches. Please try identifying with synonyms."
        )

    # ---------------- Manual / Description Helpers ----------------
    def _update_manual(self, manual):
        sections = [
            self.game_context,
            "\n".join(manual),
            self.advice,
        ]
        if self.include_one_shot_example:
            sections.append(self._build_one_shot_example_text())
        sections.append(describe_act(self.action_list))
        self.desc = ("\n\n".join(sections)).strip()

    def _build_one_shot_example_text(self) -> str:
        """Return a static one-shot example to guide the agent."""

        return (
            "# One-shot example \n\n"
            "Game Description:\n"
            "the secret document is on the immovable saber.\n"
            "there is one place with an eagle that is immobile has the enemy that is dangerous within it.\n"
            "the baleen is holding the essential goal.\n\n"
            "Current observation:\n"
            "COORDINATE SYSTEM:\n"
            "Agent: (4, 4)\n"
            "Entities:\n"
            "  bird_0: (4, 2)\n"
            "  fish_0: (2, 4)\n"
            "  sword_0: (6, 4)\n\n"
            "Original View:\n"
            "You (agent) don't have the message.\n\n"
            "You see:\n"
            "- bird 2 steps away\n"
            "- fish 2 steps away\n"
            "- sword 2 steps away\n\n"
            "Action: 4. Reason: Move East toward the sword (saber holds the message) while avoiding the bird (enemy); "
            "getting the message is required before going to the fish (goal)."
        )

    def describe(self, obs, action=None):
        base_description = _describe_frame(obs)
        action_text = f"You took action {self.action_list[action]}.\n\n" if action is not None else ""
        if self.representation == "Default":
            return action_text + base_description
        return action_text + self._format_observation_by_representation(obs, base_description)

    # ---------------- Representation Formatting ----------------
    def _format_observation_by_representation(self, obs, base_description):
        k = self.representation
        if k == "Matrix":
            return self._format_matrix_representation(obs, base_description)
        if k == "NaturalLanguage":
            return self._format_natural_language_representation(obs, base_description)
        if k == "Symbolic":
            return self._format_symbolic_representation(obs, base_description)
        if k == "Image":
            return self._format_image_representation(obs, base_description)
        if k == "ImageLanguage":
            try:
                out_path = self._render_image_grid(obs, image_language_mode=True)
                self._last_image_path = out_path
                if self.record_gif:
                    self._frame_paths.append(out_path)
            except Exception:
                pass
            return base_description
        return base_description

    def _get_agent_position(self, obs):
        a15 = np.where(obs["avatar"] == 15)
        a16 = np.where(obs["avatar"] == 16)
        if len(a15[0]) > 0:
            return (int(a15[0][0]), int(a15[1][0]))
        if len(a16[0]) > 0:
            return (int(a16[0][0]), int(a16[1][0]))
        return (0, 0)

    def _get_entity_positions(self, obs):
        entities = {}
        raw = obs["entities"]
        info2d = raw.max(axis=2) if raw.ndim == 3 else raw
        for eid in np.unique(info2d):
            if eid in (0, 15, 16):
                continue
            ys, xs = np.where(info2d == eid)
            if ys.size == 0:
                continue
            name = id_map.get(eid, f"entity_{eid}")
            entities[name] = [(int(ys[i]), int(xs[i])) for i in range(len(ys))]
        return entities

    def _format_matrix_representation(self, obs, base_description):
        agent_pos = self._get_agent_position(obs)
        entities = self._get_entity_positions(obs)
        lines = ["COORDINATE SYSTEM:", f"Agent: ({agent_pos[0]}, {agent_pos[1]})"]
        if entities:
            lines.append("Entities:")
            for et, positions in entities.items():
                for i, p in enumerate(positions):
                    lines.append(f"  {et}_{i}: ({p[0]}, {p[1]})")
        lines.append("\nOriginal View:\n" + base_description)
        return "\n".join(lines)

    def _get_direction_text(self, a, b):
        dy = b[0] - a[0]; dx = b[1] - a[1]
        parts = []
        if dy < 0: parts.append("north")
        elif dy > 0: parts.append("south")
        if dx < 0: parts.append("west")
        elif dx > 0: parts.append("east")
        return "-".join(parts) if parts else "here"

    def _format_natural_language_representation(self, obs, base_description):
        agent_pos = self._get_agent_position(obs)
        entities = self._get_entity_positions(obs)
        avatar_unique = np.unique(obs["avatar"])
        if 15 in avatar_unique:
            agent_state = "You are an agent without the message."
        elif 16 in avatar_unique:
            agent_state = "You are an agent with the message."
        else:
            agent_state = "You are an agent."
        parts = [f"{agent_state} You are currently in position {agent_pos[0]}, {agent_pos[1]}. "]
        if entities:
            here, others = [], []
            for et, poss in entities.items():
                for p in poss:
                    if p == agent_pos:
                        here.append(et)
                    else:
                        dist = abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1])
                        others.append(f"a {et} {dist} steps {self._get_direction_text(agent_pos, p)}")
            if here:
                parts.append(f"In this position you can see: {', '.join(here)}. ")
            if others:
                parts.append(f"You can see {', '.join(others)}. ")
        return "".join(parts)

    def _format_symbolic_representation(self, obs, base_description):
        entities_map = obs["entities"]
        if entities_map.ndim == 3:
            entities_map = entities_map.squeeze()
            if entities_map.ndim == 3:
                entities_map = entities_map[0]
        if entities_map.ndim != 2:
            raise ValueError("Unsupported entities map shape")
        H, W = entities_map.shape
        grid = [["." for _ in range(W)] for _ in range(H)]
        entity_ids = np.unique(entities_map)
        entity_ids = entity_ids[entity_ids > 0]
        letters = ["E", "M", "G", "X", "Y", "Z"]
        char_map = {}
        for i, eid in enumerate(sorted(entity_ids)):
            char_map[eid] = letters[i] if i < len(letters) else str(eid)
        for y in range(H):
            for x in range(W):
                eid = entities_map[y, x]
                if eid in char_map:
                    grid[y][x] = char_map[eid]
        ay, ax = self._get_agent_position(obs)
        avatar_unique = np.unique(obs["avatar"])
        agent_char = "A" if 15 in avatar_unique else ("P" if 16 in avatar_unique else "?")
        # Bounds check to handle mismatched array dimensions
        if 0 <= ay < H and 0 <= ax < W:
            grid[ay][ax] = agent_char
        rows = ["".join(r) for r in grid]
        legend = ["", "Legend: A=agent(no msg) P=agent(with msg) .=empty", "Entities"]
        for eid, ch in char_map.items():
            legend.append(f"  {ch}={id_map.get(eid, f'entity_{eid}')}")
        return "\n".join(rows + legend)

    def _format_image_representation(self, obs, base_description):
        try:
            out_path = self._render_image_grid(obs)
            self._last_image_path = out_path
            if self.record_gif:
                self._frame_paths.append(out_path)
            # Return only the textual base description; image path is exposed separately via info['image_path']
            return base_description
        except Exception as e:
            return base_description + f"\n\n[Image render failed: {e}]"

    # ---------------- Rendering Bridge ----------------
    def _render_image_grid(self, obs, image_language_mode: bool = False, transitional_prev=None, transitional_stage: bool = False) -> str:
        from .image_renderer import render_image_grid
        return render_image_grid(self, obs, image_language_mode=image_language_mode, transitional_prev=transitional_prev, transitional_stage=transitional_stage)

    # ---------------- Core API ----------------
    def reset(self):
        obs, manual = self._env.reset()
        if self.force_start_without_message:
            safety = 0
            while 16 in np.unique(obs["avatar"]) and safety < 50:
                obs, manual = self._env.reset(); safety += 1
        self._update_manual(manual)
        self.history.reset()
        self._frame_paths = []
        self._frame_index = 0
        self._action_steps = 0
        self._just_done = False
        self._current_score = 0.0
        self._last_reward = 0.0
        self._last_termination_reason = None
        # Episode directory for GIF
        if self.record_gif:
            from time import time
            try:
                repo_root = Path(__file__).resolve().parents[5]
            except Exception:
                repo_root = Path.cwd()
            base_dir = (Path(self.record_dir) if self.record_dir else repo_root / "process_results" / "data" / "renders" / "messenger")
            episode_id = f"episode_{int(time())}_{np.random.randint(1_000_000):06d}"
            self._episode_dir = base_dir / episode_id
            self._frame_output_dir = self._episode_dir / "frames"
            try:
                self._frame_output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        # Per-episode state
        self._flash_queue = []
        self._enemy_collision_reveal = False
        self._last_action = None
        self._show_message_persistent = False
        try:
            self._prev_agent_xy = self._get_agent_position(obs)
        except Exception:
            self._prev_agent_xy = None
        self._prev_agent_has_message = 16 in np.unique(obs["avatar"])
        # Debug mapping of roles
        try:
            game = getattr(self._env, "game", None)
            if game is not None:
                print(f"[MessengerEnv][RESET] Roles: enemy=({getattr(game.enemy,'name','?')},{getattr(game.enemy,'id','?')}) message=({getattr(game.message,'name','?')},{getattr(game.message,'id','?')}) goal=({getattr(game.goal,'name','?')},{getattr(game.goal,'id','?')})")
        except Exception:
            pass
        if self.record_gif and self.representation in ("Image", "ImageLanguage"):
            try:
                p = self._render_image_grid(obs, image_language_mode=(self.representation == "ImageLanguage"))
                self._last_image_path = p
                self._frame_paths.append(p)
            except Exception as e:
                print(f"[MessengerEnv][RESET] Initial frame render failed: {e}")
        info = {"obs": self.describe(obs), "manual": self.desc, "history": self.history.describe(), "score": 0, "completed": 0}
        if self.representation in ("Image", "ImageLanguage") and self._last_image_path:
            info["image_path"] = self._last_image_path
        self.history.step(info)
        self.last_obs = obs
        return obs, info

    def step(self, action: int):
        prior_has_message = self._prev_agent_has_message
        prev_pos = None
        try:
            prev_pos = self._get_agent_position(self.last_obs) if hasattr(self, 'last_obs') else None
        except Exception:
            prev_pos = None
        obs, reward, done, _ = self._env.step(action)
        termination_reason = None
        now_has_message = 16 in np.unique(obs["avatar"])
        self._last_action = action
        self._current_score += reward
        self._last_reward = reward
        moved = False
        try:
            new_pos = self._get_agent_position(obs)
            if self._prev_agent_xy and new_pos != self._prev_agent_xy:
                moved = True
        except Exception:
            new_pos = None
        self._just_done = done
        # Pickup handling: freeze original message tile + flash directly on that tile
        if (not prior_has_message) and now_has_message:
            # Record original message tile from last_obs (before pickup)
            try:
                base_obs = getattr(self, 'last_obs', None)
                if base_obs is not None:
                    game = getattr(self._env, 'game', None)
                    msg_id = getattr(getattr(game, 'message', None), 'id', None) if game else None
                    if msg_id is not None:
                        ent_prev = base_obs["entities"]
                        ent_prev2d = ent_prev.max(axis=2) if ent_prev.ndim == 3 else ent_prev
                        ys, xs = np.where(ent_prev2d == msg_id)
                        if ys.size > 0:
                            self._message_origin_pos = (int(ys[0]), int(xs[0]))
                            # Flash should target the original message tile
                            self._flash_pickup_target = self._message_origin_pos
            except Exception:
                pass
            # Force message entity layer to remain at origin position (channel index 1 by construction: enemy,msg,goal)
            try:
                if hasattr(self, '_message_origin_pos') and self._message_origin_pos is not None:
                    game = getattr(self._env, 'game', None)
                    mid = getattr(getattr(game, 'message', None), 'id', None) if game else None
                    ent_cur = obs["entities"]
                    if mid is not None and ent_cur.ndim == 3 and ent_cur.shape[2] >= 2:
                        ent_cur[:, :, 1] = 0  # clear message channel
                        oy, ox = self._message_origin_pos
                        if 0 <= oy < ent_cur.shape[0] and 0 <= ox < ent_cur.shape[1]:
                            ent_cur[oy, ox, 1] = mid
            except Exception:
                pass
            self._show_message_persistent = True
            if self.enable_flashes:
                try:
                    self._queue_message_pickup_flash()
                    # Tag queue type for renderer positioning
                    self._flash_queue_type = 'pickup'
                except Exception: pass
            # Extra immediate frames for pickup flash visibility
            if self.enable_flashes and self.record_gif and self.representation in ("Image", "ImageLanguage"):
                for _ in range(2):
                    try:
                        p = self._render_image_grid(obs, image_language_mode=(self.representation == "ImageLanguage"))
                        self._frame_paths.append(p)
                    except Exception:
                        break
        # Fail flash
        if self.enable_flashes and done and reward < 0:
            self._queue_flash_sequence("fail")
        # Enemy collision reveal (fail, ImageLanguage)
        if self.representation == "ImageLanguage" and done and reward < 0 and not self._enemy_collision_reveal:
            try:
                game = getattr(self._env, "game", None)
                enemy_id = getattr(game.enemy, "id", None) if game else None
                if enemy_id is not None:
                    ents2d = obs["entities"].max(axis=2) if obs["entities"].ndim == 3 else obs["entities"]
                    ay, ax = self._get_agent_position(obs)
                    if 0 <= ay < ents2d.shape[0] and 0 <= ax < ents2d.shape[1] and int(ents2d[ay, ax]) == enemy_id:
                        self._enemy_collision_reveal = True
            except Exception:
                pass
        self._prev_agent_has_message = now_has_message
        if moved:
            self._prev_agent_xy = new_pos
        if done:
            self._enemy_collision_reveal = True
        try:
            desc = self.describe(obs, action)
        except Exception:
            desc = "Environment Error."
        step_info = {
            "obs": desc,
            "manual": self.desc,
            "history": self.history.describe(),
            "score": reward,
            "completed": int(reward == 1 and done),
            "action_index": action,
            "action_name": self.action_list[action] if 0 <= action < len(self.action_list) else str(action),
            "step": self._action_steps,
        }
        if self.representation in ("Image", "ImageLanguage") and self._last_image_path:
            step_info["image_path"] = self._last_image_path
        self.history.step(step_info)
        if self.print_actions:
            print(f"[MessengerEnv][STEP] a={action} ({step_info['action_name']}) prev_pos={prev_pos} new_pos={new_pos} reward={reward} done={done}")
        self._action_steps += 1
        # Termination classification
        if (not done) and self.step_limit is not None and self._action_steps >= self.step_limit:
            done = True; termination_reason = "max_steps"
        if done and termination_reason is None:
            game = getattr(self._env, "game", None)
            goal_id = getattr(getattr(game, "goal", None), "id", None) if game else None
            enemy_id = getattr(getattr(game, "enemy", None), "id", None) if game else None
            ay, ax = self._get_agent_position(obs)
            ents2d = obs["entities"].max(axis=2) if obs["entities"].ndim == 3 else obs["entities"]
            cell_eid = None
            try:
                if 0 <= ay < ents2d.shape[0] and 0 <= ax < ents2d.shape[1]:
                    cell_eid = int(ents2d[ay, ax])
            except Exception:
                cell_eid = None
            if reward > 0:
                termination_reason = "reach_goal_with_message"
            elif reward < 0:
                if enemy_id is not None and cell_eid == enemy_id:
                    termination_reason = "enemy_encounter"
                elif goal_id is not None and cell_eid == goal_id and not now_has_message:
                    termination_reason = "reach_goal_without_message"
                else:
                    termination_reason = "enemy_encounter"
        if done:
            self._last_termination_reason = termination_reason
        # Final frame
        if done and self.record_gif and self.representation in ("Image", "ImageLanguage"):
            try:
                if reward > 0 and self.enable_flashes:
                    try:
                        self._queue_flash_sequence("success")
                    except Exception:
                        pass
                p = self._render_image_grid(obs, image_language_mode=(self.representation == "ImageLanguage"))
                self._frame_paths.append(p)
                self._last_image_path = p
                step_info["image_path"] = p
            except Exception:
                pass
        if done and self.record_gif and self._frame_paths:
            try:
                from smartplay.utils.recording import save_gif_from_paths
                if self._episode_dir is None:
                    try:
                        repo_root = Path(__file__).resolve().parents[5]
                    except Exception:
                        repo_root = Path.cwd()
                    base_dir = repo_root / "process_results" / "data" / "renders" / "messenger"
                    base_dir.mkdir(parents=True, exist_ok=True)
                    self._episode_dir = base_dir / f"episode_{np.random.randint(1_000_000):06d}"
                    self._episode_dir.mkdir(parents=True, exist_ok=True)
                gif_path = self._episode_dir / "episode.gif"
                # Hold final frame longer for readability (duplicate last frame)
                if self._frame_paths:
                    last_frame = self._frame_paths[-1]
                    # Add 12 extra repeats (extended hold for readability)
                    self._frame_paths.extend([last_frame]*12)
                save_gif_from_paths(self._frame_paths, str(gif_path), duration=self.gif_duration)
                step_info["episode_gif"] = str(gif_path.resolve())
                if termination_reason:
                    step_info["terminated_reason"] = termination_reason
            except Exception as e:
                step_info["episode_gif_error"] = str(e)
        self.last_obs = obs
        return obs, reward, done, step_info

    # -------------- Flash Helpers (delegate to image_renderer) --------------
    # Flash helpers delegate to renderer
    def _queue_message_pickup_flash(self):
        try:
            from .image_renderer import queue_message_pickup_flash
            queue_message_pickup_flash(self)
        except Exception:
            pass

    def _load_flash_sequence(self, flash_type: str):
        try:
            from .image_renderer import load_flash_sequence
            return load_flash_sequence(self, flash_type)
        except Exception:
            return []

    def _queue_flash_sequence(self, flash_type: str):
        try:
            from .image_renderer import queue_flash_sequence
            queue_flash_sequence(self, flash_type)
        except Exception:
            pass
