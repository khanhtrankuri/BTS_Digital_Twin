import torch

from stage2_refiner.geometry import preprocess_geometry
from stage2_refiner.losses import Stage2Loss, get_stage2_loss_weights, ssim_metric
from stage2_refiner.evaluator import evaluate_loader, summarize_by_scene
from stage2_refiner.model import GeometryGuidedNAFNet


CFG = {"LOSS": {"L1_WEIGHT": .2, "MSE_WEIGHT": .55, "DSSIM_WEIGHT": .2, "EDGE_WEIGHT": .05,
                 "RESIDUAL_WEIGHT": .001, "LOW_FREQUENCY_WEIGHT": .02},
       "LOSS_SCHEDULE": {"A": {"MSE_WEIGHT": .35}, "B": {"MSE_WEIGHT": .55}, "C": {"MSE_WEIGHT": .65}}}


def test_geometry_is_finite_normalized_and_masked():
    depth = torch.tensor([[[float("nan"), 1.0], [10.0, 3.0]]])
    normal = torch.rand(3, 2, 2); alpha = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])
    depth, normal, alpha, valid = preprocess_geometry(depth, normal, alpha)
    assert torch.isfinite(depth).all() and torch.isfinite(normal).all()
    assert depth.min() >= -1 and depth.max() <= 1
    assert torch.allclose(torch.linalg.vector_norm(normal[:, valid[0]], dim=0), torch.ones(valid.sum()), atol=1e-5)
    assert alpha.min() >= 0 and alpha.max() <= 1


def test_loss_zero_like_identical_and_schedule():
    image = torch.full((1, 3, 16, 16), .4); depth = torch.zeros(1, 1, 16, 16)
    normal = torch.zeros(1, 3, 16, 16); alpha = torch.ones(1, 1, 16, 16)
    terms = Stage2Loss(CFG)(image, torch.zeros_like(image), image, image, depth, normal, alpha)
    assert terms["mse"].item() == 0
    assert terms["dssim"].item() < 1e-5
    assert torch.isfinite(terms["total"])
    assert get_stage2_loss_weights(0, 100, CFG)["mse"] == .35
    assert get_stage2_loss_weights(50, 100, CFG)["mse"] == .55
    assert get_stage2_loss_weights(90, 100, CFG)["mse"] == .65


def test_edge_and_residual_losses_respond():
    target = torch.zeros(1, 3, 16, 16); target[..., 8:] = 1
    blurred = torch.nn.functional.avg_pool2d(target, 5, stride=1, padding=2)
    depth = torch.zeros(1, 1, 16, 16); normal = torch.zeros(1, 3, 16, 16); alpha = torch.ones(1, 1, 16, 16)
    criterion = Stage2Loss(CFG)
    exact = criterion(target, torch.zeros_like(target), target, target, depth, normal, alpha)
    wrong = criterion(blurred, torch.ones_like(target) * .1, target, target, depth, normal, alpha)
    assert wrong["edge"] > exact["edge"]
    assert wrong["residual"] > exact["residual"]


def test_evaluator_reports_before_after_regions_and_scenes():
    model = GeometryGuidedNAFNet(width=8, encoder_blocks=(1,), decoder_blocks=(1,), middle_blocks=1)
    rgb = torch.rand(1, 3, 12, 13)
    batch = {"rgb": rgb, "target": rgb.clone(), "depth": torch.ones(1, 1, 12, 13),
             "normal": torch.zeros(1, 3, 12, 13), "alpha": torch.ones(1, 1, 12, 13),
             "scene": ["S"], "image_name": ["x.png"]}
    summary, rows = evaluate_loader(model, [batch], torch.device("cpu"))
    assert summary["delta_psnr"] == 0 and "region_high_alpha_delta_psnr" in rows[0]
    assert summarize_by_scene(rows)["S"]["delta_ssim"] == 0
