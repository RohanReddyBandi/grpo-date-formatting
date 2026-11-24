# test_grpo.py
"""Unit tests for the GRPO helper functions in grpo_train.py.

Run:  pytest test_grpo.py -v
Note: importing grpo_train pulls in torch / transformers / pandas / matplotlib,
so those must be installed in the test environment.
"""
import pytest
import torch

from grpo_train import compute_reward, normalize_advantages

ANSWER = "01-15-2024"  # MM-DD-YYYY


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------
def test_exact_match_gives_one():
    assert compute_reward("01-15-2024", ANSWER) == pytest.approx(1.0)


def test_exact_match_ignores_surrounding_whitespace():
    assert compute_reward("  01-15-2024\n", ANSWER) == pytest.approx(1.0)


def test_partial_score_when_some_components_wrong():
    # format(0.2) + month(0.2) + year(0.2) = 0.6 ; day is wrong
    assert compute_reward("03-22-2024", "03-21-2024") == pytest.approx(0.6)


def test_partial_score_only_format_bonus():
    # correct MM-DD-YYYY shape but every component wrong -> 0.2
    assert compute_reward("12-31-1999", "01-01-2020") == pytest.approx(0.2)


def test_wrong_format_gives_zero():
    assert compute_reward("15/01/2024", ANSWER) == pytest.approx(0.0)   # slashes
    assert compute_reward("2024-01-15", ANSWER) == pytest.approx(0.0)   # wrong order
    assert compute_reward("not a date", ANSWER) == pytest.approx(0.0)


def test_extra_text_on_first_line_gives_zero():
    # fullmatch fails when the line has anything but the bare date
    assert compute_reward("the date is 01-15-2024", ANSWER) == pytest.approx(0.0)
    assert compute_reward("01-15-2024 today", ANSWER) == pytest.approx(0.0)


def test_empty_output_gives_zero():
    assert compute_reward("", ANSWER) == pytest.approx(0.0)
    assert compute_reward("   \n\t", ANSWER) == pytest.approx(0.0)


def test_only_first_line_is_graded():
    # correct date on a LATER line -> first line is wrong -> 0
    assert compute_reward("I think\n01-15-2024", ANSWER) == pytest.approx(0.0)
    # correct date on the FIRST line, junk after -> still graded -> 1.0
    assert compute_reward("01-15-2024\nsorry, wrong", ANSWER) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# normalize_advantages
# ---------------------------------------------------------------------------
def test_advantages_are_centered_per_group():
    g = 4
    rewards = torch.tensor([0.0, 0.2, 0.6, 1.0,   # group 0
                            0.0, 0.0, 0.0, 0.8])  # group 1
    adv = normalize_advantages(rewards, g)

    assert adv.shape == (rewards.shape[0],)
    group0, group1 = adv[:g], adv[g:]

    # each group is mean-centered on its own
    assert group0.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert group1.mean().item() == pytest.approx(0.0, abs=1e-6)

    # higher reward -> higher advantage, independently per group
    assert (group0[3] > group0[0]).item()
    assert (group1[3] > group1[0]).item()


def test_equal_rewards_give_zero_advantages_and_no_nans():
    g = 4
    rewards = torch.tensor([0.2, 0.2, 0.2, 0.2,
                            1.0, 1.0, 1.0, 1.0])
    adv = normalize_advantages(rewards, g)

    assert torch.all(torch.isfinite(adv))
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-8)
