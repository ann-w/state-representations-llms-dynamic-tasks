import random
from collections import deque
from typing import List, Tuple, Union


class Memory:
    """
    A class for managing the storage of transitions for an intelligent agent.

    Attributes:
        transition_memory: A deque for storing transitions.
    """

    def __init__(self, capacity: int = None):
        self.transition_memory = deque(maxlen=capacity) if capacity else deque()
        self.rolling_summary = "Start of game"
        self.max_summary_tokens = 100

    def update_rolling_summary(self, new_summary: str):
        """Update the rolling summary with token limit enforcement"""
        if new_summary:
            # Better token estimation: ~4-5 characters per token on average
            max_chars = self.max_summary_tokens * 5  # More generous estimation

            if len(new_summary) > max_chars:
                # Truncate at word boundary, not character boundary
                truncated = new_summary[:max_chars]
                last_space = truncated.rfind(" ")
                if (
                    last_space > max_chars * 0.8
                ):  # If we can find a reasonable word boundary
                    self.rolling_summary = truncated[:last_space]
                else:
                    self.rolling_summary = truncated
            else:
                self.rolling_summary = new_summary

    def get_rolling_summary(self) -> str:
        """Get the current rolling summary"""
        return self.rolling_summary

    def reset_rolling_summary(self):
        """Reset summary for new episode"""
        self.rolling_summary = "Start of game"

    def store_transition(
        self, transition: Tuple[Union[str, float], int, Union[str, float], bool]
    ) -> None:
        """Store a transition in the transition memory.

        Each transition is a tuple of the form (state, action_index, reward, next_state, done).
        """
        self.transition_memory.append(transition)

    def get_transitions(
        self, n: int = None
    ) -> List[Tuple[Union[str, float], int, Union[str, float], bool]]:
        """Return the last n transitions from the transition memory. If n is None, return all."""
        if n is None:
            return list(self.transition_memory)
        return list(self.transition_memory)[-n:]

    def sample_transitions(
        self, batch_size: int
    ) -> List[Tuple[Tuple[Union[str, float], int, Union[str, float], bool]]]:
        """Sample a batch of transitions from the transition memory."""
        if batch_size > len(self.transition_memory):
            raise ValueError(
                "Batch size is larger than the number of stored transitions."
            )
        return random.sample(list(self.transition_memory), batch_size)

    def clear_transitions(self) -> None:
        """Clear all transitions from the transition memory."""
        self.transition_memory.clear()

    def __len__(self) -> int:
        """Return the total number of items in transition memory."""
        return len(self.transition_memory)
