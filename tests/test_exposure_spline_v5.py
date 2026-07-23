import torch

from utils.exposure_utils import TemporalExposureSpline


def test_spline_identity_with_irregular_and_missing_train_times():
    field = TemporalExposureSpline([0.0, None, 4.0, 10.0], num_knots=6, degree=3)
    gain, bias = field.gains_biases()
    torch.testing.assert_close(gain, torch.ones_like(gain), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(bias, torch.zeros_like(bias))
    assert field.infer_time(5.0) is not None


def test_linear_spline_interpolates_continuously_and_has_gradients():
    field = TemporalExposureSpline([0.0, 1.0], num_knots=2, degree=1,
                                   per_view_residual=False)
    with torch.no_grad():
        field.raw_bias_knots[0].fill_(-0.5)
        field.raw_bias_knots[1].fill_(0.5)
    _, middle = field.evaluate(torch.tensor(0.5))
    torch.testing.assert_close(middle, torch.zeros_like(middle), atol=1e-6, rtol=1e-6)
    image = torch.ones((3, 2, 2))
    output = field(image, 0)
    output.mean().backward()
    assert field.raw_gain_knots.grad is not None and field.raw_bias_knots.grad is not None


def test_spline_regularization_and_state_round_trip():
    first = TemporalExposureSpline([0.0, 1.0, 3.0], num_knots=5, degree=3)
    with torch.no_grad():
        first.raw_gain_knots.normal_()
        first.raw_bias_residual.normal_()
    assert torch.isfinite(first.regularization_loss(0.01, 0.01))
    second = TemporalExposureSpline([0.0, 1.0, 3.0], num_knots=5, degree=3)
    second.load_state_dict(first.state_dict())
    left = first.gains_biases()
    right = second.gains_biases()
    torch.testing.assert_close(left[0], right[0])
    torch.testing.assert_close(left[1], right[1])


def test_missing_test_time_requests_pose_fallback():
    field = TemporalExposureSpline([None, None, None], num_knots=4, degree=3)
    assert field.infer_time(None) is None
    assert field.infer_time(1.0) is None

