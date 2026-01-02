from collections import deque
from typing import List, Optional, Deque, Dict, Any


ROLLING_SUMMARY_TIMESTEPS = 20


class Message:
    """Represents a conversation message with role, content, and optional attachment."""

    def __init__(self, role: str, content: str, attachment: Optional[object] = None):
        self.role = role  # 'system', 'user', 'assistant'
        self.content = content  # String content of the message
        self.attachment = attachment

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content}, attachment={self.attachment})"


class HistoryPromptBuilder:
    """Builds a prompt with a history of observations, actions, and reasoning.

    Maintains a configurable history of text, images, and chain-of-thought reasoning to
    construct prompt messages for conversational agents.
    """

    def __init__(
        self,
        max_text_history: int = 16,
        max_image_history: int = 1,
        system_prompt: Optional[str] = None,
        max_cot_history: int = 1,
        summary: bool = False,
    ):
        self.max_text_history = max_text_history
        self.max_image_history = max_image_history
        self.max_history = max(max_text_history, max_image_history)
        self.system_prompt = system_prompt
        self._transitions: Deque[Dict[str, Any]] = deque(maxlen=self.max_text_history)
        self._current_state: Optional[str] = None
        self._current_image = None
        self._pending_action: Optional[str] = None
        self._pending_reasoning: Optional[str] = None
        self._pending_reward: Optional[float] = None
        self._last_short_term_obs = None  # To store the latest short-term observation
        self.previous_reasoning = None
        self.max_cot_history = max_cot_history
        self.summary = summary
        self.summary_text: Optional[str] = None
        self.summary_system_prompt = (
            "You compress the recent trajectory for a game-playing agent. "
            "Keep summaries factual, action-ready, and within 25 tokens."
        )
        self.summary_user_template = (
            "History so far:\n{history}\n\n"
            "Return a single line ≤25 tokens highlighting only the most relevant facts."
        )

    def update_instruction_prompt(self, instruction: str):
        """Set the system-level instruction prompt."""
        self.system_prompt = instruction

    def update_observation(self, obs: dict):
        """Add an observation to the prompt history, which can include text, an image, or both."""
        long_term_context = obs["text"].get("long_term_context", "")
        self._last_short_term_obs = obs["text"].get("short_term_context", "")
        text = long_term_context

        image = obs.get("image", None)

        if self._current_state is None:
            # First observation establishes the initial state
            self._current_state = text
            self._current_image = image
            return

        # Build a transition using the previously stored state/action/reward and the new observation as next_state
        transition = {
            "state": self._current_state,
            "action": self._pending_action,
            "reward": self._pending_reward,
            "next_state": text,
            "reasoning": self._pending_reasoning,
            "state_image": self._current_image,
            "next_state_image": image,
        }

        self._transitions.append(transition)

        # Update the current state to the latest observation for future transitions
        self._current_state = text
        self._current_image = image
        self._pending_action = None
        self._pending_reward = None
        self._pending_reasoning = None

    def update_action(self, action: str):
        """Add an action to the prompt history, including reasoning if available."""
        self._pending_action = action
        self._pending_reasoning = self.previous_reasoning
        self.previous_reasoning = None

    def update_reasoning(self, reasoning: str):
        """Set the reasoning text to be included with subsequent actions."""
        self.previous_reasoning = reasoning

    def update_reward(self, reward: float):
        """Store the reward obtained after executing the pending action."""
        self._pending_reward = reward

    def reset(self):
        """Clear the event history."""
        self._transitions.clear()
        self._current_state = None
        self._current_image = None
        self._pending_action = None
        self._pending_reward = None
        self._pending_reasoning = None
        self.summary_text = None

    def get_prompt(self, icl_episodes=False) -> List[Message]:
        """Generate a list of Message objects representing the prompt.

        Returns:
            List[Message]: Messages constructed from the event history.
        """
        messages = []

        if self.system_prompt and not icl_episodes:
            messages.append(Message(role="user", content=self.system_prompt))

        if self.summary and self.summary_text and not icl_episodes:
            messages.append(
                Message(
                    role="user",
                    content=f"Summary past trajectory:\n{self.summary_text}",
                )
            )

        # Process stored transitions to create messages
        include_transitions = not (self.summary and self.summary_text and not icl_episodes)

        if include_transitions:
            transitions = list(self._transitions)[-ROLLING_SUMMARY_TIMESTEPS:]
            start_idx = max(len(self._transitions) - len(transitions) + 1, 1)

            for idx, transition in enumerate(transitions, start=start_idx):
                lines = [f"Step {idx}:"]

                if transition.get("state"):
                    lines.append("State:")
                    lines.append(transition["state"])

                if transition.get("action"):
                    lines.append(f"Action: {transition['action']}")

                if transition.get("reward") is not None:
                    lines.append(f"Reward: {transition['reward']}")

                if transition.get("next_state"):
                    lines.append("Next state:")
                    lines.append(transition["next_state"])

                content = "\n".join(lines)
                message = Message(role="user", content=content)
                messages.append(message)

                # Optionally include reasoning if available and within history budget
                if transition.get("reasoning") and idx > len(self._transitions) - self.max_cot_history:
                    reasoning_content = "Reasoning:\n" + transition["reasoning"]
                    messages.append(Message(role="assistant", content=reasoning_content))

        # Append the current state for the agent to act upon
        if self._current_state is not None:
            parts = ["Current state:"]
            if self._last_short_term_obs:
                parts.append(self._last_short_term_obs)
            parts.append(self._current_state)
            current_state_content = "\n".join(parts)
            messages.append(Message(role="user", content=current_state_content, attachment=self._current_image))

        return messages

    def refresh_summary(self, client):
        """Regenerate the rolling summary using the provided LLM client."""

        if not self.summary:
            return None

        history_text = self._build_history_text()
        if not history_text.strip():
            self.summary_text = None
            return None

        summary_prompt = [
            Message(role="system", content=self.summary_system_prompt),
            Message(role="user", content=self.summary_user_template.format(history=history_text)),
        ]

        response = client.generate(summary_prompt)
        summary = response.completion.strip()
        self.summary_text = summary
        return summary


    def _build_history_text(self) -> str:
        """Linearize history events into plain text for summarization."""

        chunks = []
        transitions = list(self._transitions)[-ROLLING_SUMMARY_TIMESTEPS:]
        start_idx = max(len(self._transitions) - len(transitions) + 1, 1)

        for idx, transition in enumerate(transitions, start=start_idx):
            parts = [f"Step {idx}"]
            if transition.get("state"):
                parts.append(f"State: {transition['state']}")
            if transition.get("action"):
                parts.append(f"Action: {transition['action']}")
            if transition.get("reward") is not None:
                parts.append(f"Reward: {transition['reward']}")
            if transition.get("next_state"):
                parts.append(f"Next: {transition['next_state']}")
            if transition.get("reasoning"):
                parts.append(f"Reason: {transition['reasoning']}")
            chunks.append(" | ".join(parts))

        if self._current_state is not None:
            chunks.append(f"Current: {self._current_state}")

        return "\n".join(chunks)
