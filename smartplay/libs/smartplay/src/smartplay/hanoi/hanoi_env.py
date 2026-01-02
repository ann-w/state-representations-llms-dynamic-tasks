import itertools
import random

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

from ..utils import HistoryTracker, describe_act
from .state_representations import StateRepresentation, get_representation


# Each integer action (0..5) corresponds to a move (src, dst) meaning:
#   Move top disk from peg src to peg dst
action_to_move = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


class HanoiEnv(gym.Env):
    default_iter = 10
    default_steps = 30

    def __init__(
        self,
        max_steps=5,
        num_disks=4,
        env_noise=0,
        show_valid_actions=False,
        reward_shaping=False,
        state_representation="default",
        record_gif: bool = False,
        record_dir=None,
        gif_duration: float = 0.2,
        include_one_shot_example: bool = False,
    ):
        self.num_disks = num_disks
        self.env_noise = env_noise
        self.show_valid_actions = show_valid_actions
        self.reward_shaping = reward_shaping

        # Initialize state representation
        if isinstance(state_representation, str):
            self.state_formatter = get_representation(state_representation)
        elif isinstance(state_representation, StateRepresentation):
            self.state_formatter = state_representation
        else:
            raise ValueError(
                "state_representation must be a string or StateRepresentation instance"
            )

        # Recording options (used when using image-based state representation)
        self.record_gif = record_gif
        self.record_dir = record_dir
        self.gif_duration = gif_duration
        self.include_one_shot_example = include_one_shot_example
        self._frame_paths = []

        # Basic Gym spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Tuple(self.num_disks * (spaces.Discrete(3),))

        # State & Goal
        self.current_state = None
        self.goal_state = self.num_disks * (2,)
        self.history = HistoryTracker(max_steps)
        self.done = None
        self.ACTION_LOOKUP = {
            0: "(0,1) - top disk of pole 0 to top of pole 1 ",
            1: "(0,2) - top disk of pole 0 to top of pole 2 ",
            2: "(1,0) - top disk of pole 1 to top of pole 0",
            3: "(1,2) - top disk of pole 1 to top of pole 2",
            4: "(2,0) - top disk of pole 2 to top of pole 0",
            5: "(2,1) - top disk of pole 2 to top of pole 1",
        }

        # Action descriptions
        self.action_list = [
            "Move the top disk of rod A to the top of rod B",
            "Move the top disk of rod A to the top of rod C",
            "Move the top disk of rod B to the top of rod A",
            "Move the top disk of rod B to the top of rod C",
            "Move the top disk of rod C to the top of rod A",
            "Move the top disk of rod C to the top of rod B",
        ]

        # Generate description based on representation
        self.desc = self._generate_description()

    def _generate_description(self):
        """Generate description text based on the current state representation"""

        # Common introduction
        base_desc = """The game consists of three rods (A,B,C) and a number of disks of various sizes, which can go onto any rod. 
The game begins with the disks stacked on rod A in order of decreasing size, the smallest at the top (righthand side). 
The objective is to move the entire stack to rod C, obeying the following rules:

 - Only one disk may be moved at a time.
 - Each move consists of taking the top disk from one of the stacks and placing it on top of another stack or on an empty rod.
 - You cannot place a bigger disk on top of a smaller disk.

"""

        # Create example states for demonstration
        # Example: disk 0 on A, disk 1 on B, disk 2 on C
        example_internal_state = (0, 1, 2)  # disk0->pegA, disk1->pegB, disk2->pegC
        start_internal_state = tuple([0] * self.num_disks)  # All disks on peg A
        goal_internal_state = tuple([2] * self.num_disks)  # All disks on peg C

        # Get formatted examples
        example_state = self.state_formatter.from_internal_state(
            example_internal_state, 3
        )
        start_state = self.state_formatter.from_internal_state(
            start_internal_state, self.num_disks
        )
        goal_state = self.state_formatter.from_internal_state(
            goal_internal_state, self.num_disks
        )

        # Get representation-specific examples and explanations
        if self.state_formatter.__class__.__name__ == "DefaultStateRepresentation":
            # Default representation
            example_text = """For example, considering movements from B under the following setting:
- A: |bottom, [0], top|
- B: |bottom, [1], top|
- C: |bottom, [2], top|
You are only allowed to move from B to C but not A, since the top of B (1) is smaller than the top of C (2) but bigger than the top of A (0).

Finally, the starting configuration is:
{}

and the goal configuration is:
{}
with top on the right and bottom on the left""".format(
                self.state_formatter.describe(start_state),
                self.state_formatter.describe(goal_state),
            )

        elif self.state_formatter.__class__.__name__ == "DictListStateRepresentation":
            # Dict list representation
            example_text = """For example, considering movements from B under the following setting:
{}
You are only allowed to move from B to C but not A, since the top of B (1) is smaller than the top of C (2) but bigger than the top of A (0).

Finally, the starting configuration is:
{}

and the goal configuration is:
{}
In this representation, lists show disks from bottom to top (largest to smallest disk IDs).""".format(
                self.state_formatter.describe(example_state),
                self.state_formatter.describe(start_state),
                self.state_formatter.describe(goal_state),
            )

        elif self.state_formatter.__class__.__name__ == "MatrixStateRepresentation":
            # Matrix representation
            example_text = """For example, considering movements from B under the following setting:
{}
You are only allowed to move from B to C but not A, since the top of B (1) is smaller than the top of C (2) but bigger than the top of A (0).

Finally, the starting configuration is:
{}

and the goal configuration is:
{}
In this matrix representation, each row represents a rod (A, B, C), each column represents a position from bottom (left) to top (right), and -1 means empty slot.""".format(
                self.state_formatter.describe(example_state),
                self.state_formatter.describe(start_state),
                self.state_formatter.describe(goal_state),
            )

        elif (
            self.state_formatter.__class__.__name__
            == "NaturalLanguageStateRepresentation"
        ):
            # Natural language representation
            example_text = """For example, considering movements from B under the following setting:
{}
You are only allowed to move from B to C but not A, since the top of B (1) is smaller than the top of C (2) but bigger than the top of A (0).

Finally, the starting configuration is:
{}

and the goal configuration is:
{}
In this representation, the state is described in natural language with disk positions from bottom to top.""".format(
                self.state_formatter.describe(example_state),
                self.state_formatter.describe(start_state),
                self.state_formatter.describe(goal_state),
            )
        elif self.state_formatter.__class__.__name__ == "LuaFunctionStateRepresentation":
            # Lua function representation
            example_text = """For example, considering movements from B under the following setting:
{}
You are only allowed to move from B to C but not A, since the top of B (1) is smaller than the top of C (2) but bigger than the top of A (0).

Finally, the starting configuration is:
{}

and the goal configuration is:
{}
In this representation, the state is described as a Lua function that returns a table with three lists (A, B, C) showing disks from bottom to top.""".format(
                self.state_formatter.describe(example_state),
                self.state_formatter.describe(start_state),
                self.state_formatter.describe(goal_state),
            )
        else:
            # Fallback for custom representations
            example_text = """For example, considering movements from B under the following setting:
{}

Finally, the starting configuration is:
{}

and the goal configuration is:
{}""".format(
                self.state_formatter.describe(example_state),
                self.state_formatter.describe(start_state),
                self.state_formatter.describe(goal_state),
            )

        # Combine base description with representation-specific examples
        one_shot_text = ""
        if self.include_one_shot_example:
            one_shot_text = "\n\n" + self._build_one_shot_example_text(
                example_state, example_internal_state
            )

        full_desc = (
            base_desc + example_text + one_shot_text + "\n\n" + describe_act(self.action_list)
        )

        return full_desc.strip()

    def _build_one_shot_example_text(
        self, formatted_state, internal_state
    ) -> str:
        """Return a formatted one-shot example illustrating a legal move."""

        peg_names = ["A", "B", "C"]
        tops = {}
        for peg_id, peg_name in enumerate(peg_names):
            disks = [disk for disk, location in enumerate(internal_state) if location == peg_id]
            tops[peg_name] = min(disks) if disks else None

        def fmt_top(value):
            return str(value) if value is not None else "empty"

        observation_text = self.state_formatter.describe(formatted_state)

        action_index = 3  # "Move the top disk of rod B to the top of rod C"
        action_number = action_index + 1

        return (
            "# One-shot example\n\n"
            f"Current observation:\n{observation_text}\n\n"
            "Past trajectory:\n[]\n\n"
            f"Action: {action_number}. Reason: Top(B)={fmt_top(tops['B'])} can be placed on C where Top(C)={fmt_top(tops['C'])}, "
            f"but not on A where Top(A)={fmt_top(tops['A'])}; B→C is legal and progresses toward stacking on C."
        )

    def set_state_representation(self, representation):
        """Change the state representation and update description"""
        if isinstance(representation, str):
            self.state_formatter = get_representation(representation)
        elif isinstance(representation, StateRepresentation):
            self.state_formatter = representation
        else:
            raise ValueError(
                "representation must be a string or StateRepresentation instance"
            )

        # Regenerate description with new representation
        self.desc = self._generate_description()

    def get_formatted_state(self):
        """Get the current state in the chosen representation format"""
        if self.current_state is None:
            return None
        return self.state_formatter.from_internal_state(
            self.current_state, self.num_disks
        )

    def set_state_representation(self, representation):
        """Change the state representation"""
        if isinstance(representation, str):
            self.state_formatter = get_representation(representation)
        elif isinstance(representation, StateRepresentation):
            self.state_formatter = representation
        else:
            raise ValueError(
                "representation must be a string or StateRepresentation instance"
            )

    def step(self, action):
        """
        * Inputs:
            - action: integer from 0 to 5 (see ACTION_LOOKUP)
        * Outputs:
            - current_state: state after transition
            - reward: reward from transition
            - done: episode state
            - info: dict of booleans (noisy?/invalid action?)
        0. Check if transition is noisy or not
        1. Transform action (0 to 5 integer) to tuple move - see Lookup
        2. Check if move is allowed
        3. If it is change corresponding entry | If not return same state
        4. Check if episode completed and return
        """
        if self.done:
            raise RuntimeError("Episode finished. Call env.reset() to start a new one.")

        info = {"transition_failure": False, "invalid_action": False}

        # Possibly override chosen action if there's environment noise
        if self.env_noise > 0:
            if random.random() <= self.env_noise:
                action = random.randint(0, self.action_space.n - 1)
                info["transition_failure"] = True
        else:
            info["transition_failure"] = False

        # Check if the action is valid
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            info["invalid_action"] = True

        # We compute reward according to whether it's valid or not
        reward = self.compute_reward(info["invalid_action"])

        # If it's valid, do the move; otherwise we skip
        if not info["invalid_action"]:
            move = action_to_move[action]
            disk_to_move = min(self.disks_on_peg(move[0]))
            moved_state = list(self.current_state)
            moved_state[disk_to_move] = move[1]
            self.current_state = tuple(moved_state)

        # Check completion
        if self.current_state == self.goal_state:
            # Add a large bonus for finishing
            reward += 100
            self.done = True

        # Build info
        info["state"] = (
            self.disks_on_peg(0),
            self.disks_on_peg(1),
            self.disks_on_peg(2),
        )
        info["score"] = len(self.disks_on_peg(2))
        info["manual"] = self.desc
        # Return a textual description that might show valid actions
        info["obs"] = self.describe_state(info, action)
        info["formatted_state"] = self.get_formatted_state()

        # If image-based, collect frame path for potential GIF
        try:
            if self.record_gif and getattr(self.state_formatter, "__class__").__name__ == "ImageStateRepresentation":
                formatted = info.get("formatted_state")
                if isinstance(formatted, dict) and "path" in formatted:
                    self._frame_paths.append(formatted["path"])
        except Exception:
            pass

        # Tracking
        info["completed"] = 1 if self.done else 0
        self.history.step(info)

        # Finalize GIF on completion
        if self.done and self.record_gif and self._frame_paths:
            try:
                from smartplay.utils.recording import save_gif_from_paths
                import os
                base_dir = os.path.join(os.path.dirname(__file__), "images")
                out_dir = self.record_dir or base_dir
                os.makedirs(out_dir, exist_ok=True)
                gif_path = os.path.join(out_dir, f"hanoi_episode_{np.random.randint(1_000_000):06d}.gif")
                save_gif_from_paths(self._frame_paths, gif_path, duration=self.gif_duration)
                info["episode_gif"] = gif_path
            except Exception as e:
                info["episode_gif_error"] = str(e)

        return self.current_state, reward, self.done, info

    def compute_reward(self, invalid_action):
        """
        Applies either the default or shaped reward scheme.
        """
        if not self.reward_shaping:
            # Original scheme:
            # -1 if invalid
            #  0 if valid
            # +100 if goal completed (applied separately)
            return -1 if invalid_action else 0
        else:
            # Reward Shaping Example:
            # -2 if invalid
            # +1 if valid
            # +100 on goal (applied separately in step)
            if invalid_action:
                return -2
            else:
                return +1

    def get_valid_actions(self):
        """
        In the combined scheme, if `show_valid_actions` is True or False,
        we still compute what is valid. The environment penalizes invalid moves.
        """
        valid = []
        for idx, (src, dst) in enumerate(action_to_move):
            if self.move_allowed((src, dst)):
                valid.append(idx)
        return valid

    def describe_state(self, state, action=None):
        """
        Returns a textual representation of rods using the chosen representation.
        """
        # Get formatted state
        formatted_state = self.state_formatter.from_internal_state(
            self.current_state, self.num_disks
        )

        # Show attempted action:
        if action is not None:
            result = f"You tried to {self.action_list[action].lower()}.\n"
        else:
            result = ""

        # Use the formatter's describe method
        result += self.state_formatter.describe(formatted_state)

        # If we combined show_valid_actions = True, let's list them in the text
        if self.show_valid_actions:
            valid_actions = self.get_valid_actions()
            if valid_actions:
                # Convert each valid action to a user-friendly string
                valid_strs = [
                    f"{idx+1}. {self.action_list[idx]}" for idx in valid_actions
                ]
                result += f"\n\nValid actions: {', '.join(valid_strs)}"
            else:
                result += "\n\nValid actions: None"

        return result.strip()

    def move_allowed(self, move):
        """
        Checks if we can move top disk from move[0] to move[1].
        1) The source rod must have at least one disk.
        2) We cannot place a bigger disk on top of a smaller disk.
        """
        disks_from = self.disks_on_peg(move[0])
        disks_to = self.disks_on_peg(move[1])
        if disks_from:
            # If 'to' rod is empty or top disk on 'to' rod is bigger than top disk on 'from'
            return (min(disks_to) > min(disks_from)) if disks_to else True
        else:
            return False

    def disks_on_peg(self, peg):
        """
        Returns a list of disk IDs on the given peg.
        Smaller ID can mean physically smaller or bigger disk
        depending on your naming scheme, but it’s consistent.
        """
        return [
            disk for disk in range(self.num_disks) if self.current_state[disk] == peg
        ]

    def reset(self):
        """
        Reset all disks to rod A (peg 0).
        """
        self.current_state = self.num_disks * (0,)
        self.done = False
        self.history.reset()
        self._frame_paths = []

        info = {
            "state": (
                self.disks_on_peg(0),
                self.disks_on_peg(1),
                self.disks_on_peg(2),
            )
        }
        info["score"] = len(self.disks_on_peg(2))
        info["manual"] = self.desc
        info["obs"] = self.describe_state(info)
        info["formatted_state"] = self.get_formatted_state()
        info["completed"] = 0
        self.history.step(info)
        return self.current_state, info

    def get_oracle_summary(self) -> str:
        """Generate a perfect textual description of current state from ground truth.
        
        This provides an 'oracle' summary that bypasses LLM summarization,
        giving the agent a perfect compressed state representation.
        Useful for ablation studies to isolate summarization ability from reasoning.
        """
        # Get disks on each peg (peg 0=A, 1=B, 2=C)
        peg_names = ['A', 'B', 'C']
        lines = ["Current State:"]
        
        for peg_idx, peg_name in enumerate(peg_names):
            disks = self.disks_on_peg(peg_idx)
            if not disks:
                lines.append(f"  Rod {peg_name}: empty")
            else:
                # Sort disks from bottom (largest) to top (smallest)
                # In this env, smaller disk ID = smaller physical disk
                sorted_disks = sorted(disks, reverse=True)
                if len(sorted_disks) == 1:
                    lines.append(f"  Rod {peg_name}: disk {sorted_disks[0]}")
                else:
                    disk_str = ", ".join(str(d) for d in sorted_disks)
                    lines.append(f"  Rod {peg_name}: disks {disk_str} (bottom to top)")
        
        # Add goal reminder
        lines.append(f"Goal: Move all {self.num_disks} disks to Rod C.")
        
        return "\n".join(lines)

    def render(self, mode="human", close=False):
        """Not used in this text-based environment"""
        pass

    def set_env_parameters(self, num_disks=4, env_noise=0, verbose=True):
        """
        Dynamically change the environment parameters if you like.
        """
        self.num_disks = num_disks
        self.env_noise = env_noise
        self.observation_space = spaces.Tuple(self.num_disks * (spaces.Discrete(3),))
        self.goal_state = self.num_disks * (2,)

        if verbose:
            print("Hanoi Environment parameters updated to:")
            print(f" - Disks: {self.num_disks}")
            print(f" - Noise Probability: {self.env_noise}")

    def get_movability_map(self, fill=False):
        """
        Returns a map of valid moves for all possible states, optional usage.
        """
        mov_map = np.zeros(self.num_disks * (3,) + (6,))

        if fill:
            # List out all permutations of disk placements
            id_list = self.num_disks * [0] + self.num_disks * [1] + self.num_disks * [2]
            states = list(itertools.permutations(id_list, self.num_disks))
            for state in states:
                for action in range(6):
                    move = action_to_move[action]
                    disks_from = []
                    disks_to = []
                    for d in range(self.num_disks):
                        if state[d] == move[0]:
                            disks_from.append(d)
                        elif state[d] == move[1]:
                            disks_to.append(d)

                    if disks_from:
                        valid = (min(disks_to) > min(disks_from)) if disks_to else True
                    else:
                        valid = False

                    if not valid:
                        mov_map[state][action] = -np.inf

        return mov_map


# -------------------------------------------------------------------
# Original classes kept for backward compatibility
# (they default to show_valid_actions=False, reward_shaping=False)
# -------------------------------------------------------------------


class Hanoi3Disk(HanoiEnv):
    """Basic 3 disk Hanoi Environment"""

    def __init__(
        self,
        max_steps=5,
        env_noise=0,
        show_valid_actions=False,
        reward_shaping=False,
        state_representation="default",
        include_one_shot_example: bool = False,
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=3,
            env_noise=env_noise,
            show_valid_actions=show_valid_actions,
            reward_shaping=reward_shaping,
            state_representation=state_representation,
            include_one_shot_example=include_one_shot_example,
        )


class Hanoi4Disk(HanoiEnv):
    """Basic 4 disk Hanoi Environment"""

    def __init__(
        self,
        max_steps=5,
        env_noise=0,
        show_valid_actions=False,
        reward_shaping=False,
        state_representation="default",
        include_one_shot_example: bool = False,
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=4,
            env_noise=env_noise,
            show_valid_actions=show_valid_actions,
            reward_shaping=reward_shaping,
            state_representation=state_representation,
            include_one_shot_example=include_one_shot_example,
        )


# -------------------------------------------------------------------
# NEW specialized classes with 2disks, show_valid_actions and reward_shaping
# -------------------------------------------------------------------


class Hanoi2Disk(HanoiEnv):
    """Basic 2 disk Hanoi Environment"""

    def __init__(
        self,
        max_steps=5,
        env_noise=0,
        show_valid_actions=False,
        reward_shaping=False,
        state_representation="default",
        include_one_shot_example: bool = False,
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=2,
            env_noise=env_noise,
            show_valid_actions=show_valid_actions,
            reward_shaping=reward_shaping,
            state_representation=state_representation,
            include_one_shot_example=include_one_shot_example,
        )


class Hanoi2DShowValid(HanoiEnv):
    """2-disk environment that always shows valid actions"""

    def __init__(
        self,
        max_steps=5,
        env_noise=0,
        state_representation="default",
        include_one_shot_example: bool = False,
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=2,
            env_noise=env_noise,
            show_valid_actions=True,
            reward_shaping=False,
            state_representation=state_representation,
            include_one_shot_example=include_one_shot_example,
        )


class Hanoi2DRewardShaping(HanoiEnv):
    """
    2-disk environment using reward shaping.
    """

    def __init__(
        self,
        max_steps=5,
        env_noise=0,
        state_representation="default",
        include_one_shot_example: bool = False,
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=2,
            env_noise=env_noise,
            show_valid_actions=False,
            reward_shaping=True,
            state_representation=state_representation,
            include_one_shot_example=include_one_shot_example,
        )


class Hanoi2DShowValidRewardShaping(HanoiEnv):
    """
    2-disk environment with valid actions displayed and reward shaping.
    """

    def __init__(
        self,
        max_steps=5,
        env_noise=0,
        state_representation="default",
        include_one_shot_example: bool = False,
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=2,
            env_noise=env_noise,
            show_valid_actions=True,
            reward_shaping=True,
            state_representation=state_representation,
            include_one_shot_example=include_one_shot_example,
        )


class Hanoi3DShowValid(HanoiEnv):
    """
    3-disk environment that always shows valid actions, classic reward scheme.
    """

    def __init__(
        self,
        max_steps=5,
        env_noise=0,
        state_representation="default",
        include_one_shot_example: bool = False,
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=3,
            env_noise=env_noise,
            show_valid_actions=True,
            reward_shaping=False,
            state_representation=state_representation,
            include_one_shot_example=include_one_shot_example,
        )


class Hanoi3DRewardShaping(HanoiEnv):
    """
    3-disk environment using reward shaping.
    """

    def __init__(
        self,
        max_steps=5,
        env_noise=0,
        state_representation="default",
        include_one_shot_example: bool = False,
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=3,
            env_noise=env_noise,
            show_valid_actions=False,
            reward_shaping=True,
            state_representation=state_representation,
            include_one_shot_example=include_one_shot_example,
        )


class Hanoi3DShowValidRewardShaping(HanoiEnv):
    """
    3-disk environment with valid actions displayed and reward shaping.
    """

    def __init__(
        self,
        max_steps=5,
        env_noise=0,
        state_representation="default",
        include_one_shot_example: bool = False,
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=3,
            env_noise=env_noise,
            show_valid_actions=True,
            reward_shaping=True,
            state_representation=state_representation,
            include_one_shot_example=include_one_shot_example,
        )


# -------------------------------------------------------------------
# Add class for different representations of the 3-disk Hanoi
#
# -------------------------------------------------------------------


class Hanoi3DiskDictList(HanoiEnv):
    """3-disk Hanoi Environment using dictionary list representation"""

    def __init__(
        self,
        max_steps=5,
        env_noise=0,
        show_valid_actions=False,
        reward_shaping=False,
        include_one_shot_example: bool = False,
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=3,
            env_noise=env_noise,
            show_valid_actions=show_valid_actions,
            reward_shaping=reward_shaping,
            state_representation="dict_list",
            include_one_shot_example=include_one_shot_example,
        )


class Hanoi3DiskMatrix(HanoiEnv):
    """3-disk Hanoi Environment using matrix representation"""

    def __init__(
        self,
        max_steps=5,
        env_noise=0,
        show_valid_actions=False,
        reward_shaping=False,
        include_one_shot_example: bool = False,
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=3,
            env_noise=env_noise,
            show_valid_actions=show_valid_actions,
            reward_shaping=reward_shaping,
            state_representation="matrix",
            include_one_shot_example=include_one_shot_example,
        )


class Hanoi3DiskNaturalLanguage(HanoiEnv):
    """3-disk Hanoi Environment using natural language representation"""

    def __init__(
        self,
        max_steps=5,
        env_noise=0,
        show_valid_actions=False,
        reward_shaping=False,
        include_one_shot_example: bool = False,
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=3,
            env_noise=env_noise,
            show_valid_actions=show_valid_actions,
            reward_shaping=reward_shaping,
            state_representation="natural_language",
            include_one_shot_example=include_one_shot_example,
        )


class Hanoi3DiskLuaFunction(HanoiEnv):
    """3-disk Hanoi Environment using Lua function representation"""

    def __init__(
        self,
        max_steps=5,
        env_noise=0,
        show_valid_actions=False,
        reward_shaping=False,
        include_one_shot_example: bool = False,
    ):
        super().__init__(
            max_steps=max_steps,
            num_disks=3,
            env_noise=env_noise,
            show_valid_actions=show_valid_actions,
            reward_shaping=reward_shaping,
            state_representation="lua_function",
            include_one_shot_example=include_one_shot_example,
        )
