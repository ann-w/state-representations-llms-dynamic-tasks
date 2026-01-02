"""
Programmatic ASCII map generator for BabyAI observations.

This module converts text-based BabyAI observations into ASCII maps
for the VoT Oracle ablation study. Unlike LLM-generated maps, these
are ground-truth representations derived directly from the observation.
"""

import re
from typing import Tuple, List, Optional


# Direction deltas when agent faces UP (forward = negative row)
DIRECTIONS = {
    'forward': (-1, 0),
    'ahead': (-1, 0),      # alias for forward
    'away': (-1, 0),       # treat as forward (ambiguous but common)
    'from': (-1, 0),       # treat as forward (in front)
    'right': (0, 1),
    'left': (0, -1),
    'behind': (1, 0),
    'backward': (1, 0),
    'back': (1, 0),
}

# Color prefixes for objects
COLOR_MAP = {
    'red': 'r',
    'green': 'g',
    'blue': 'b',
    'yellow': 'y',
    'purple': 'p',
    'grey': 'x',
}


def parse_observation_line(line: str) -> Tuple[str, int, int]:
    """
    Parse a single observation line from BabyAI.
    
    Returns: (object_name, delta_row, delta_col) relative to agent facing UP.
    
    Examples:
        "a wall 6 steps forward" → ("wall", -6, 0)
        "a green key 2 steps right and 5 steps forward" → ("green key", -5, 2)
        "a blue door 1 step left" → ("blue door", 0, -1)
    """
    d_row, d_col = 0, 0
    
    # Pattern: "N step(s) DIRECTION"
    pattern = r'(\d+)\s+steps?\s+(forward|ahead|away|from|right|left|behind|backward|back)'
    matches = re.findall(pattern, line.lower())
    
    for steps_str, direction in matches:
        steps = int(steps_str)
        dr, dc = DIRECTIONS[direction]
        d_row += dr * steps
        d_col += dc * steps
    
    # Extract object name (everything between "a " and the first number)
    obj_match = re.match(r'^a\s+(.+?)\s+\d+', line.lower())
    obj = obj_match.group(1).strip() if obj_match else "unknown"
    
    return obj, d_row, d_col


def get_symbol(obj: str) -> str:
    """
    Map object name to ASCII symbol.
    
    Returns a 1-2 character symbol:
        - Color prefix (r, g, b, y, p, x) if applicable
        - Object type (K=key, O=ball, B=box, D=door, L=locked, /=open, #=wall)
    """
    obj = obj.lower()
    
    # Determine color prefix
    color_prefix = ''
    for color, prefix in COLOR_MAP.items():
        if color in obj:
            color_prefix = prefix
            break
    
    # Determine object symbol
    if 'wall' in obj:
        return '#'
    elif 'key' in obj:
        return f'{color_prefix}K' if color_prefix else 'K'
    elif 'ball' in obj:
        return f'{color_prefix}o' if color_prefix else 'o'
    elif 'box' in obj:
        return f'{color_prefix}B' if color_prefix else 'B'
    elif 'door' in obj:
        if 'locked' in obj:
            return f'{color_prefix}L' if color_prefix else 'L'
        elif 'open' in obj:
            return f'{color_prefix}/' if color_prefix else '/'
        return f'{color_prefix}D' if color_prefix else 'D'
    else:
        return '?'


# Maximum distance to show on map (entities beyond this are clamped)
# This keeps the map compact and readable
MAX_MAP_DISTANCE = 4


def observation_to_ascii_map(observation: str, padding: int = 0, max_distance: int = MAX_MAP_DISTANCE) -> str:
    """
    Convert BabyAI text observation to a compact ASCII map.
    
    The grid is sized to fit visible entities, capped at max_distance
    to keep the map small and readable for language models.
    
    Args:
        observation: Multi-line text observation from BabyAI environment
        padding: Extra cells to add around the bounding box (default 0)
        max_distance: Maximum distance from agent to show (default 4)
    
    Returns:
        ASCII map string with space-separated symbols
    
    Example:
        Input:
            "a wall 6 steps forward
             a green key 2 steps right and 5 steps forward"
        
        Output (capped at 4):
            # . .
            . . gK
            . . .
            . . .
            ^ . .
    """
    # Parse all entities from observation
    entities: List[Tuple[str, int, int]] = []  # (symbol, d_row, d_col)
    
    for line in observation.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        obj, d_row, d_col = parse_observation_line(line)
        symbol = get_symbol(obj)
        
        # Clamp distances to max_distance to keep map compact
        d_row = max(-max_distance, min(max_distance, d_row))
        d_col = max(-max_distance, min(max_distance, d_col))
        
        entities.append((symbol, d_row, d_col))
    
    if not entities:
        # No entities visible, just show agent
        return "^"
    
    # Calculate bounding box (relative to agent at 0,0)
    min_row = min(0, min(e[1] for e in entities)) - padding
    max_row = max(0, max(e[1] for e in entities)) + padding
    min_col = min(0, min(e[2] for e in entities)) - padding
    max_col = max(0, max(e[2] for e in entities)) + padding
    
    # Grid dimensions
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    # Agent position in grid coordinates
    agent_grid_row = 0 - min_row
    agent_grid_col = 0 - min_col
    
    # Initialize grid with empty cells
    grid = [['.' for _ in range(width)] for _ in range(height)]
    
    # Place agent (facing up/forward)
    grid[agent_grid_row][agent_grid_col] = '^'
    
    # Place all entities
    for symbol, d_row, d_col in entities:
        grid_row = d_row - min_row
        grid_col = d_col - min_col
        grid[grid_row][grid_col] = symbol
    
    # Convert grid to string (space-separated for readability)
    return '\n'.join(' '.join(row) for row in grid)


def get_map_legend() -> str:
    """Return a compact legend explaining the ASCII map symbols."""
    return """Symbols: ^=You(facing up) #=Wall K=Key D=Door L=Locked /=Open o=Ball B=Box .=Empty
Colors: r=red g=green b=blue y=yellow p=purple x=grey"""


# VoT Oracle prompt that uses programmatic map
VOT_ORACLE_PROMPT = """
# Map Representation
Below is a precise top-down ASCII map generated from your current observation.
Use this map to understand your spatial position relative to objects.

{map_content}

{legend}

# Task
Based on the map above and your mission, choose the single best action.

Available actions:
- turn left
- turn right  
- go forward
- pick up
- drop
- toggle

Output Requirements (no additional text):
<|ACTION|>YOUR_CHOSEN_ACTION<|END|>
<|REASON|>ONE SENTENCE (≤40 TOKENS) JUSTIFYING THE ACTION<|END|>
""".strip()


def build_vot_oracle_prompt(observation: str) -> str:
    """
    Build the complete VoT Oracle prompt with programmatic ASCII map.
    
    Args:
        observation: Text observation from BabyAI environment
    
    Returns:
        Complete prompt string with map and instructions
    """
    ascii_map = observation_to_ascii_map(observation)
    legend = get_map_legend()
    
    return VOT_ORACLE_PROMPT.format(
        map_content=ascii_map,
        legend=legend
    )


if __name__ == "__main__":
    # Test the module
    test_observations = [
        "a wall 6 steps forward\na green key 2 steps right and 5 steps forward",
        "a wall 1 step forward\na wall 3 steps left",
        "a wall 5 steps forward\na wall 3 steps left\na locked blue door 2 steps left and 5 steps forward\na blue key 1 step left and 3 steps forward",
    ]
    
    for i, obs in enumerate(test_observations):
        print(f"=== Test {i+1} ===")
        print("Observation:")
        print(obs)
        print("\nASCII Map:")
        print(observation_to_ascii_map(obs))
        print("\n")
