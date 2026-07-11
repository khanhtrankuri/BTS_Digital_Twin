from argparse import ArgumentParser

from arguments import OptimizationParams
from utils.training_schedules import get_lr_multipliers, get_stage_loss_weights


def _options():
    parser = ArgumentParser()
    group = OptimizationParams(parser)
    return group.extract(parser.parse_args(["--loss_schedule_enabled"]))


def test_stage_loss_weights_are_valid_and_transition_cleanly():
    options = _options()
    for iteration, expected_mse in ((1, 0.05), (12001, 0.35), (30001, 0.60)):
        weights = get_stage_loss_weights(iteration, options)
        assert weights["mse"] == expected_mse
        assert all(value >= 0 for value in weights.values())


def test_stage_c_reduces_geometric_lrs():
    options = _options()
    assert get_lr_multipliers(1, options)["xyz"] == 1.0
    assert get_lr_multipliers(30001, options)["xyz"] == 0.1
