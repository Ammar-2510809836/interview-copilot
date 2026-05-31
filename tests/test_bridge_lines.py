import unittest

import core.bridge_lines as bridge_lines
from core.bridge_lines import pick_bridge_line, BANNED_SUBSTRINGS, BRIDGE_POOLS


class TestBridgeLines(unittest.TestCase):
    def setUp(self):
        # pick_bridge_line keeps anti-repeat state in a module global; reset it
        # so tests don't depend on execution order.
        bridge_lines._last_line = None

    def test_behavioral_question_uses_behavioral_pool(self):
        line = pick_bridge_line("Tell me about a time you handled conflict on your team.")
        self.assertIn(line, BRIDGE_POOLS["behavioral"])

    def test_technical_question_uses_technical_pool(self):
        line = pick_bridge_line("How would you design a scalable Kubernetes deployment on AWS?")
        self.assertIn(line, BRIDGE_POOLS["technical"])

    def test_coding_question_uses_coding_pool(self):
        line = pick_bridge_line("Write a function to reverse a linked list.")
        self.assertIn(line, BRIDGE_POOLS["coding"])

    def test_followup_flag_uses_followup_pool(self):
        line = pick_bridge_line("And why is that?", is_followup=True)
        self.assertIn(line, BRIDGE_POOLS["followup"])

    def test_empty_text_returns_generic_line_without_crashing(self):
        line = pick_bridge_line("")
        self.assertIn(line, BRIDGE_POOLS["generic"])

    def test_no_pool_line_contains_evaluative_phrasing(self):
        for pool in BRIDGE_POOLS.values():
            for line in pool:
                lowered = line.lower()
                for banned in BANNED_SUBSTRINGS:
                    self.assertNotIn(banned, lowered, f"{line!r} contains banned phrase {banned!r}")

    def test_consecutive_calls_avoid_immediate_repeat(self):
        # With anti-repeat, 20 calls on the same input should not return the same
        # line twice in a row (pools have >= 2 lines each).
        prev = None
        for _ in range(20):
            line = pick_bridge_line("How would you scale this API?")
            self.assertNotEqual(line, prev)
            prev = line


if __name__ == "__main__":
    unittest.main()
