import sys
from pathlib import Path

import pytest
from unittest.mock import Mock, patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts.memory import Memory
from scripts.utils.environment_utils import extract_summary_from_answer


class TestRollingSummary:

    def setup_method(self):
        """Setup for each test method"""
        self.memory = Memory(capacity=1000)

    def test_memory_rolling_summary_initialization(self):
        """Test that memory initializes rolling summary correctly"""
        assert self.memory.rolling_summary == "Start of game"
        assert self.memory.max_summary_tokens == 100

    def test_memory_reset_rolling_summary(self):
        """Test memory rolling summary reset"""
        self.memory.update_rolling_summary("Some progress made")
        assert self.memory.rolling_summary == "Some progress made"

        self.memory.reset_rolling_summary()
        assert self.memory.rolling_summary == "Start of game"

    def test_memory_update_rolling_summary(self):
        """Test memory rolling summary update with token limit"""
        # Test normal update
        new_summary = "Moved disk from A to B"
        self.memory.update_rolling_summary(new_summary)
        assert self.memory.rolling_summary == "Moved disk from A to B"

        # Test token limit (~5 chars per token)
        long_summary = "a" * 600  # 600 characters, should be truncated to 500
        self.memory.update_rolling_summary(long_summary)
        assert len(self.memory.rolling_summary) == self.memory.max_summary_tokens * 5

    def test_extract_summary_from_answer(self):
        """Test summary extraction from different answer formats"""

        # Test successful extraction
        answer1 = "Action: 1. Summary: Moved disk A to B. Reason: Good move"
        result1 = extract_summary_from_answer(answer1)
        assert result1 == "Moved disk A to B."

        # Test with different spacing
        answer2 = "Action: 2. Summary:    Started the game   . Reason: Beginning"
        result2 = extract_summary_from_answer(answer2)
        assert result2 == "Started the game."

        # Test fallback when no summary found
        answer3 = "Action: 1. Reason: Just a reason"
        result3 = extract_summary_from_answer(answer3)
        assert result3 == "Game progressing"

    def test_rolling_summary_vs_trajectory_selection(self):
        """Test that rolling summary properly replaces trajectory"""

        # Add some transitions to memory
        self.memory.store_transition(("state1", "action1", 1.0, "state2", False))
        self.memory.store_transition(("state2", "action2", 1.0, "state3", False))

        # Test with rolling summary enabled
        rolling_summary = self.memory.get_rolling_summary()
        trajectory_with_summary = []

        assert rolling_summary == "Start of game"
        assert len(trajectory_with_summary) == 0

        # Test with rolling summary disabled
        trajectory_without_summary = self.memory.get_transitions(n=5)

        assert len(trajectory_without_summary) == 2
        assert trajectory_without_summary[0][0] == "state1"

    def test_rolling_summary_end_to_end(self):
        """Test complete rolling summary workflow"""

        # Start with fresh memory
        self.memory.reset_rolling_summary()
        assert self.memory.rolling_summary == "Start of game"

        # Simulate step 1
        answer1 = "Action: 1. Summary: Moved small disk to middle. Reason: Good start"
        summary1 = extract_summary_from_answer(answer1)
        self.memory.update_rolling_summary(summary1)
        assert self.memory.rolling_summary == "Moved small disk to middle."

        # Simulate step 2
        answer2 = "Action: 2. Summary: Moved large disk to right, small to right. Reason: Progress"
        summary2 = extract_summary_from_answer(answer2)
        self.memory.update_rolling_summary(summary2)
        assert (
            self.memory.rolling_summary
            == "Moved large disk to right, small to right."
        )

        # Simulate step 3
        answer3 = (
            "Action: 3. Summary: Completed tower on right peg. Reason: Win condition"
        )
        summary3 = extract_summary_from_answer(answer3)
        self.memory.update_rolling_summary(summary3)
        assert self.memory.rolling_summary == "Completed tower on right peg."

    def test_rolling_summary_token_limit_enforcement(self):
        """Test that rolling summary enforces token limits"""

        # Create a summary that's exactly at the limit (25 tokens * 4 chars = 100)
        summary_at_limit = "a" * (self.memory.max_summary_tokens * 5)
        self.memory.update_rolling_summary(summary_at_limit)
        assert len(self.memory.rolling_summary) == self.memory.max_summary_tokens * 5

        # Create a summary that exceeds the limit
        summary_over_limit = "a" * (self.memory.max_summary_tokens * 6)
        self.memory.update_rolling_summary(summary_over_limit)
        assert len(self.memory.rolling_summary) == self.memory.max_summary_tokens * 5


if __name__ == "__main__":
    pytest.main([__file__])
