import unittest

from core.turn_detection import (
    decide_turn_action,
    looks_incomplete,
    recommended_wait_timeout,
    COMPLETE_PAUSE_SECONDS,
)


class TestTurnDetection(unittest.TestCase):
    def test_complete_question_is_ready_after_patient_pause(self):
        decision = decide_turn_action(
            "Tell me about a time you handled an angry customer.",
            pause=3.0,
        )

        self.assertTrue(decision.should_answer)
        self.assertFalse(decision.force)
        self.assertEqual(decision.reason, "complete_pause")

    def test_complete_question_keeps_listening_during_short_pause(self):
        # A small breath pause mid-question (e.g. TTS interviewer like Zara)
        # must NOT trigger an answer.
        decision = decide_turn_action(
            "Tell me about a time you handled an angry customer.",
            pause=1.0,
        )

        self.assertFalse(decision.should_answer)
        self.assertEqual(decision.reason, "waiting")

    def test_dangling_fragment_waits_during_long_pause(self):
        decision = decide_turn_action(
            "Can you explain how you would handle a customer with",
            pause=4.0,
        )

        self.assertFalse(decision.should_answer)
        self.assertEqual(decision.reason, "incomplete")

    def test_dangling_fragment_forces_only_after_long_silence(self):
        decision = decide_turn_action(
            "Can you explain how you would handle a customer with",
            pause=6.5,
        )

        self.assertTrue(decision.should_answer)
        self.assertTrue(decision.force)
        self.assertEqual(decision.reason, "max_wait")

    def test_ambiguous_statement_waits_during_short_pause(self):
        decision = decide_turn_action(
            "I worked at Amazon for about three years",
            pause=2.0,
        )

        self.assertFalse(decision.should_answer)
        self.assertEqual(decision.reason, "waiting")

    def test_ambiguous_statement_answers_after_ambiguous_pause(self):
        decision = decide_turn_action(
            "I worked at Amazon for about three years",
            pause=4.0,
        )

        self.assertTrue(decision.should_answer)
        self.assertFalse(decision.force)
        self.assertEqual(decision.reason, "ambiguous_pause")

    def test_short_followup_ready_when_previous_question_exists(self):
        decision = decide_turn_action(
            "Why?",
            pause=0.5,
            previous_question="How would you handle an escalation?",
        )

        self.assertTrue(decision.should_answer)
        self.assertEqual(decision.reason, "short_followup")

    def test_pending_complete_text_polls_below_threshold(self):
        # The wait timeout is a POLL interval, not the pause threshold. While
        # interviewer text is pending it must be shorter than COMPLETE_PAUSE so
        # the loop re-checks often and fires near 2.5s, not at ~5s.
        timeout = recommended_wait_timeout(
            "Tell me about a time you handled an angry customer."
        )
        self.assertLess(timeout, COMPLETE_PAUSE_SECONDS)

    def test_pending_ambiguous_text_polls_below_threshold(self):
        timeout = recommended_wait_timeout("I worked at Amazon for three years")
        self.assertLess(timeout, COMPLETE_PAUSE_SECONDS)

    def test_short_followup_polls_fast(self):
        timeout = recommended_wait_timeout(
            "why?", previous_question="How would you handle an escalation?"
        )
        self.assertLessEqual(timeout, 0.5)

    def test_looks_incomplete_detects_trailing_comma_and_dangling_word(self):
        self.assertTrue(looks_incomplete("I would start by checking the account,"))
        self.assertTrue(looks_incomplete("I would start by checking with"))
        self.assertFalse(looks_incomplete("How would you handle a delayed response?"))


if __name__ == "__main__":
    unittest.main()
