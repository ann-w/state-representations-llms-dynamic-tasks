"""
Classes that follow a gym-like interface and implement stage one of the Messenger environment.
"""

import json
import random
from collections import namedtuple
from pathlib import Path

import messenger.envs.config as config
import numpy as np
from messenger.envs.base import MessengerEnv, Position
from messenger.envs.manual import TextManual
from messenger.envs.utils import games_from_json

# Used to track sprites in StageOne, where we do not use VGDL to handle sprites.
Sprite = namedtuple("Sprite", ["name", "id", "position"])


class StageOne(MessengerEnv):
    def __init__(
        self,
        split: str,
        message_prob: float = 0.2,
        shuffle_obs: bool = True,
        use_shaping: bool = False,
        use_text_substitution: bool = False,
        end_on_pickup: bool = False,
        pickup_reward: float = 0.0,
    ):
        """Stage one where objects are immovable (no VGDL engine).

        Args:
            split: dataset split (train[_mc|_sc]/val/test)
            message_prob: probability avatar starts WITH the message
            shuffle_obs: shuffle textual descriptions in obs
            use_shaping: enable distance-based reward shaping
            use_text_substitution: use random textual descriptions
            end_on_pickup: legacy terminal pickup behavior if True
            pickup_reward: reward to add on pickup if not terminal
        """
        super().__init__(lvl=1)
        self.message_prob = message_prob
        self.shuffle_obs = shuffle_obs
        self.use_shaping = use_shaping
        self.use_text_substitution = use_text_substitution
        self.end_on_pickup = end_on_pickup
        self.pickup_reward = pickup_reward

        this_folder = Path(__file__).parent
        games_json_path = this_folder.joinpath("games.json")
        if "train" in split and "mc" in split:
            game_split = "train_multi_comb"
            text_json_path = this_folder.joinpath("texts", "text_train.json")
        elif "train" in split and "sc" in split:
            game_split = "train_single_comb"
            text_json_path = this_folder.joinpath("texts", "text_train.json")
        elif "val" in split:
            game_split = "val"
            text_json_path = this_folder.joinpath("texts", "text_val.json")
        elif "test" in split:
            game_split = "test"
            text_json_path = this_folder.joinpath("texts", "text_test.json")
        else:
            raise Exception(f"Split: {split} not understood.")

        self.all_games = games_from_json(json_path=games_json_path, split=game_split)

        with text_json_path.open("r") as f:
            descrip = json.load(f)
        self.descriptors = {}
        for entity in descrip:
            self.descriptors[entity] = {}
            for role in ("enemy", "message", "goal"):
                opts = []
                opts.extend(descrip[entity][role]["immovable"])
                opts.extend(descrip[entity][role]["unknown"])
                self.descriptors[entity][role] = opts

        # fixed possible locations
        self.positions = [
            Position(y=3, x=5),
            Position(y=5, x=3),
            Position(y=5, x=7),
            Position(y=7, x=5),
        ]
        self.avatar_start_pos = Position(y=5, x=5)
        self.avatar = None
        self.enemy = None
        self.message = None
        self.goal = None
        self.message_collected = False
        self.prev_dist_to_message = None
        self.prev_dist_to_goal = None

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _manhattan_distance(self, sprite_1, sprite_2) -> int:
        return abs(sprite_1.position.x - sprite_2.position.x) + abs(
            sprite_1.position.y - sprite_2.position.y
        )

    def _get_manual(self):
        # Return list of three descriptor sentences (enemy, message, goal)
        manual = []
        for sprite, role in ((self.enemy, "enemy"), (self.message, "message"), (self.goal, "goal")):
            if self.use_text_substitution:
                manual.append(random.choice(self.descriptors[sprite.name][role]))
            else:
                manual.append(f"{sprite.name} {role}")
        if self.shuffle_obs:
            manual = random.sample(manual, len(manual))
        return manual

    def _get_obs(self):
        """Observation always includes enemy, message, goal (message not hidden after pickup)."""
        grid_entities = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 3))
        avatar_layer = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))
        # Fixed order: enemy(0), message(1), goal(2)
        for idx, sprite in enumerate([self.enemy, self.message, self.goal]):
            grid_entities[sprite.position.y, sprite.position.x, idx] = sprite.id
        avatar_layer[self.avatar.position.y, self.avatar.position.x, 0] = self.avatar.id
        return {"entities": grid_entities, "avatar": avatar_layer}

    def reset(self):
        self.game = random.choice(self.all_games)
        enemy, message, goal = self.game.enemy, self.game.message, self.game.goal

        shuffled_pos = random.sample(self.positions, 4)
        self.enemy = Sprite(name=enemy.name, id=enemy.id, position=shuffled_pos[0])
        self.message = Sprite(name=message.name, id=message.id, position=shuffled_pos[1])
        self.goal = Sprite(name=goal.name, id=goal.id, position=shuffled_pos[2])
        self.message_collected = False

        if random.random() < self.message_prob:
            self.avatar = Sprite(
                name=config.WITH_MESSAGE.name,
                id=config.WITH_MESSAGE.id,
                position=self.avatar_start_pos,
            )
            self.message_collected = True  # starts with message
        else:
            self.avatar = Sprite(
                name=config.NO_MESSAGE.name,
                id=config.NO_MESSAGE.id,
                position=self.avatar_start_pos,
            )

        self.prev_dist_to_message = self._manhattan_distance(self.avatar, self.message)
        self.prev_dist_to_goal = self._manhattan_distance(self.avatar, self.goal)

        obs = self._get_obs()
        manual = self._get_manual()
        return obs, manual

    def _move_avatar(self, action):
        """
        Updates the agent's position based on the selected action.

        The agent moves within the game grid, avoiding out-of-bounds movements.
        Possible actions:
        - Stay in place
        - Move Up
        - Move Down
        - Move Left
        - Move Right
        """
        # print(f"Before move: {self.avatar.position}, Action: {action}")  # Debugging output

        # Action: Stay in place (No movement)
        if action == config.ACTIONS.stay:
            # print("Action: Stay → No movement")
            return

        elif action == config.ACTIONS.up:
            if self.avatar.position.y <= 0:  # top boundary
                # print("Hit upper boundary. No movement.")
                return
            new_position = Position(
                y=self.avatar.position.y - 1, x=self.avatar.position.x
            )

        elif action == config.ACTIONS.down:
            if self.avatar.position.y >= config.STATE_HEIGHT - 1:  # bottom boundary
                # print("Hit lower boundary. No movement.")
                return
            new_position = Position(
                y=self.avatar.position.y + 1, x=self.avatar.position.x
            )

        elif action == config.ACTIONS.left:
            if self.avatar.position.x <= 0:  # left boundary
                # print("Hit left boundary. No movement.")
                return
            new_position = Position(
                y=self.avatar.position.y, x=self.avatar.position.x - 1
            )

        elif action == config.ACTIONS.right:
            if self.avatar.position.x >= config.STATE_WIDTH - 1:  # right boundary
                # print("Hit right boundary. No movement.")
                return
            new_position = Position(
                y=self.avatar.position.y, x=self.avatar.position.x + 1
            )

        else:
            raise Exception(f"{action} is not a valid action.")

        # Update the avatar's position
        self.avatar = Sprite(
            name=self.avatar.name, id=self.avatar.id, position=new_position
        )
        # print(f"After move: {self.avatar.position}")  # Debugging output

    def _overlap(self, sprite_1, sprite_2):
        overlap = (
            sprite_1.position.x == sprite_2.position.x
            and sprite_1.position.y == sprite_2.position.y
        )
        # if overlap:
        # print(f"Overlap detected: {sprite_1.name} and {sprite_2.name}")
        return overlap

    def _has_message(self):
        return self.avatar.name == config.WITH_MESSAGE.name or self.message_collected

    def step(self, action):
        # Move avatar
        self._move_avatar(action)
        obs = self._get_obs()

        # Distances
        new_dist_to_message = self._manhattan_distance(self.avatar, self.message)
        new_dist_to_goal = self._manhattan_distance(self.avatar, self.goal)

        shaping_reward = 0.0
        if self.use_shaping:
            if not self._has_message():
                delta_msg = self.prev_dist_to_message - new_dist_to_message
                delta_msg = max(-1, min(1, delta_msg))
                shaping_reward += 0.5 * delta_msg
            else:
                delta_goal = self.prev_dist_to_goal - new_dist_to_goal
                delta_goal = max(-1, min(1, delta_goal))
                shaping_reward += 0.5 * delta_goal
            self.prev_dist_to_message = new_dist_to_message
            self.prev_dist_to_goal = new_dist_to_goal

        final_reward = 0.0
        done = False

        if self._overlap(self.avatar, self.enemy):
            final_reward = -1.0
            done = True
        elif self._overlap(self.avatar, self.message):
            # PICKUP: Non-terminal +1.0 (even if shaping enabled). If already holding, no extra reward.
            if not self._has_message():
                self.avatar = Sprite(
                    name=config.WITH_MESSAGE.name,
                    id=config.WITH_MESSAGE.id,
                    position=self.avatar.position,
                )
                self.message_collected = True
                final_reward = 1.0  # spec: reward for pickup
        elif self._overlap(self.avatar, self.goal):
            done = True
            if self._has_message():
                # Delivery reward: larger when shaping enabled
                final_reward = 50.0 if self.use_shaping else 1.0
            else:
                final_reward = -1.0


        total_reward = shaping_reward + final_reward
        return obs, total_reward, done, {}
