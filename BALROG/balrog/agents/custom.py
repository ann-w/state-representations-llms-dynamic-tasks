import logging
import re

from balrog.agents.base import BaseAgent
from balrog.ascii_map import observation_to_ascii_map, get_map_legend


REASONING_INSTRUCTIONS = """
Task:
Analyze the current observation and the past trajectory to choose the single best action that moves toward the mission objective.
Use only one of the available actions:
- turn left
- turn right
- go forward
- pick up
- drop
- toggle

Output Requirements (no additional text):
<|ACTION|>YOUR_CHOSEN_ACTION<|END|>
<|REASON|>ONE SENTENCE (≤40 TOKENS) JUSTIFYING THE ACTION<|END|>

`YOUR_CHOSEN_ACTION` must match the list above exactly.
Keep the reason factual and tied to the current goal.
""".strip()


VOT_PROMPT = """
# Visualization-of-Thought Protocol
You will:
- Draw a compact top-down ASCII map of the current situation before choosing an action.
- Update that map as you reason about candidate moves, annotating notable changes (e.g., planned path, hazards, inventory).
- Decide the single best next action from the provided list.

Guidelines for the ASCII map:
- The ASCII map must represent only what is explicitly visible in the observation.
- Keep it <=6x6 unless the observation explicitly demands more detail.
- Use consistent symbols (e.g., `A` for agent, `G` for goal, `#` for wall, `.` for empty) and include a legend when helpful.
- If multiple hypothetical moves are considered, show intermediate marks such as `?` or arrow glyphs to illustrate your updates.
- If the environment is non-grid-based (e.g., Hanoi), draw the most compact symbolic state instead of a grid.

# Required Output Sections
1. **Map (Top-Down View):** Latest ASCII grid after incorporating reasoning updates.
2. **Map Update Notes:** Bullet list (<=3 bullets) summarizing how the map changed during reasoning.
3. **Reasoning:** Brief chain-of-thought tying observation/manual to the chosen action.
4. **Action:** `Action: [number] ([action name]). Reason: [concise justification].`

# One-Shot Example

Example Input
```
Game Description:
You are in a 3x3 room. Goal is to reach the exit `G`. Walls are `#`. Keys must be picked up before locked doors.

Current observation:
Agent at center (row2,col2). Exit north (row1,col2). Key east (row2,col3). North tile is locked door needing key.

Past trajectory:
Step1: Spawned at center.
Step2: Moved south to scout, then back north.

Question:
What should the agent do next?
```

Example Response
```
Map (Top-Down View):
Row1:  .  G  #
Row2:  .  A  K
Row3:  .  .  .

Map Update Notes:
- Marked `K` east of agent as immediate pickup target.
- Highlighted blocked tile north as `#` to reflect locked door.
- Planned route A->K->G indicated with arrows in notes instead of cluttering grid.

Reasoning:
Need key before approaching locked exit. Closest safe move is east to collect key, enabling unlock next turn.

Action: 2 (Move East). Reason: Collect key required before reaching locked exit.
```
""".strip()


# VoT Oracle: Uses programmatic (ground-truth) ASCII map instead of LLM-generated
VOT_ORACLE_PROMPT = """
# ASCII Map (Top-Down View)
{map_content}

{legend}

The map shows your current view. ^ = you facing UP (forward).
To reach objects: turn to face them, go forward, then interact.""".strip()


ALLOWED_ACTIONS = (
    "turn left",
    "turn right",
    "go forward",
    "pick up",
    "drop",
    "toggle",
)


class CustomAgent(BaseAgent):
    """Reasoning agent that grounds each action in a succinct justification.
    
    Supports three visualization modes via config:
    - vot=False: No visualization, just reasoning instructions
    - vot=True: LLM generates its own ASCII map (original VoT)
    - vot_oracle=True: Programmatic ground-truth ASCII map injected
    """

    def __init__(self, client_factory, prompt_builder, agent_config=None):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.logger = logging.getLogger(__name__)
        self.agent_config = agent_config
        self.vot_enabled = bool(self._read_vot_flag(agent_config))
        self.vot_oracle_enabled = bool(self._read_config_flag(agent_config, "vot_oracle"))
        
        # VoT Oracle takes precedence over regular VoT if both are enabled
        if self.vot_oracle_enabled:
            self.vot_enabled = False  # Disable regular VoT
            self._vot_instruction_block = None  # Will be built dynamically with map
        elif self.vot_enabled:
            self._vot_instruction_block = f"{VOT_PROMPT}\n\n{REASONING_INSTRUCTIONS}"
        else:
            self._vot_instruction_block = REASONING_INSTRUCTIONS

    def act(self, obs, prev_action=None):
        """Generate the next action with reasoning based on current and past observations."""
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)
        self.prompt_builder.refresh_summary(self.client)

        messages = self.prompt_builder.get_prompt()
        
        if messages and messages[-1].role == "user":
            if self.vot_oracle_enabled:
                # Build VoT Oracle prompt with programmatic ASCII map
                instruction_block = self._build_vot_oracle_instruction(obs)
            else:
                instruction_block = self._vot_instruction_block
            messages[-1].content += "\n\n" + instruction_block

        response = self.client.generate(messages)

        action, reason = self._extract_action_and_reason(response.completion)

        response = response._replace(completion=action, reasoning=reason)
        return response

    def _extract_action_and_reason(self, response_text):
        """Extract the chosen action and reason from the model output."""

        action_match = re.search(
            r"<\|ACTION\|>(.*?)(?:<\|END\|>|\|>|</\s*ACTION>|$)",
            response_text,
            re.IGNORECASE | re.DOTALL,
        )
        reason_match = re.search(
            r"<\|REASON\|>(.*?)(?:<\|END\|>|\|>|</\s*REASON>|$)",
            response_text,
            re.IGNORECASE | re.DOTALL,
        )

        raw_action = action_match.group(1).strip() if action_match else response_text
        action = self._sanitize_action(raw_action, response_text)
        action_found = action_match is not None

        reason = ""
        if reason_match:
            reason = reason_match.group(1).strip()
        elif action_match:
            # If the action exists but reason tag missing, try to capture any trailing text after action block
            tail = response_text[action_match.end():].strip()
            reason = tail.splitlines()[0].strip() if tail else ""

        normalized_text = response_text.replace('""', '"')

        if not action_found:
            vot_action = self._extract_vot_action(normalized_text)
            if vot_action:
                raw_action = vot_action
                action = self._sanitize_action(raw_action, normalized_text)
                action_found = True

        if not reason:
            # Attempt to extract reason from common JSON-like structures or labeled text
            json_reason_match = re.search(r'"reason"\s*:\s*"([^"\n]+)"', normalized_text, re.IGNORECASE)
            if json_reason_match:
                reason = json_reason_match.group(1).strip()

        if not reason:
            labeled_reason_match = re.search(r'Reason\s*[:\-]\s*(.+)', normalized_text, re.IGNORECASE)
            if labeled_reason_match:
                reason = labeled_reason_match.group(1).splitlines()[0].strip()

        if not action_found:
            json_action_match = re.search(r'"action"\s*:\s*"([^"\n]+)"', normalized_text, re.IGNORECASE)
            if json_action_match:
                raw_action = json_action_match.group(1).strip()
                action = self._sanitize_action(raw_action, normalized_text)
            action_found = True

        if not reason:
            self.logger.debug("Model response missing <|REASON|> block: %s", response_text)

        reason = reason.replace("\n", " ").strip()

        MAX_REASON_TOKENS = 40
        reason_tokens = reason.split()
        if len(reason_tokens) > MAX_REASON_TOKENS:
            reason = " ".join(reason_tokens[:MAX_REASON_TOKENS])

        return action, reason

    def _extract_vot_action(self, text: str) -> str:
        """Extract action declaration from Visualization-of-Thought formatted text."""

        vot_action_match = re.search(
            r"Action\s*:\s*(?:\d+\s*)?(?:\(([^)]+)\)|-?\s*([a-zA-Z\s]+))",
            text,
            re.IGNORECASE,
        )
        if vot_action_match:
            for group in vot_action_match.groups():
                if group:
                    candidate = group.strip()
                    candidate = re.split(r"Reason(?:ing)?\s*:\s*", candidate)[0]
                    candidate = candidate.strip(" .:-")
                    if candidate:
                        return candidate
        return ""

    @staticmethod
    def _read_vot_flag(agent_config):
        if agent_config is None:
            return False
        if isinstance(agent_config, dict):
            return agent_config.get("vot", False)
        get_method = getattr(agent_config, "get", None)
        if callable(get_method):
            return get_method("vot", False)
        return getattr(agent_config, "vot", False)

    @staticmethod
    def _read_config_flag(agent_config, flag_name: str):
        """Generic method to read a boolean flag from agent config."""
        if agent_config is None:
            return False
        if isinstance(agent_config, dict):
            return agent_config.get(flag_name, False)
        get_method = getattr(agent_config, "get", None)
        if callable(get_method):
            return get_method(flag_name, False)
        return getattr(agent_config, flag_name, False)

    def _build_vot_oracle_instruction(self, obs) -> str:
        """Build VoT Oracle instruction block with programmatic ASCII map."""
        # Extract text observation from obs dict
        # obs structure: obs["text"]["long_term_context"] or obs["text"]["short_term_context"]
        if isinstance(obs, dict):
            text_dict = obs.get("text", {})
            if isinstance(text_dict, dict):
                # Prefer short_term_context as it has the current observation
                text_obs = text_dict.get("short_term_context", "") or text_dict.get("long_term_context", "")
            else:
                text_obs = str(text_dict)
        else:
            text_obs = str(obs)
        
        # Generate ASCII map from observation
        ascii_map = observation_to_ascii_map(text_obs)
        legend = get_map_legend()
        
        # Build the full instruction block
        vot_oracle_block = VOT_ORACLE_PROMPT.format(
            map_content=ascii_map,
            legend=legend
        )
        
        return f"{vot_oracle_block}\n\n{REASONING_INSTRUCTIONS}"

    def _sanitize_action(self, candidate: str, full_response: str) -> str:
        """Normalize the model action to one of the allowed discrete actions."""

        def normalize(text: str) -> str:
            text = text.strip()
            text = text.split("\n", 1)[0]
            text = re.sub(r"[^a-zA-Z\s]", "", text)
            return text.lower().strip()

        normalized_candidate = normalize(candidate)
        for allowed in ALLOWED_ACTIONS:
            if allowed in normalized_candidate:
                return allowed

        normalized_full = normalize(full_response)
        for allowed in ALLOWED_ACTIONS:
            if allowed in normalized_full:
                return allowed

        # Fall back to a safe default if nothing matches
        return "go forward"
