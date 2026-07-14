"""Opt-in BTS-GeoGS-v2 loss and learning-rate schedules."""

def get_stage_loss_weights(iteration, cfg):
    if not getattr(cfg, "loss_schedule_enabled", False):
        return {"l1": 1.0 - cfg.lambda_dssim, "mse": 0.0, "dssim": cfg.lambda_dssim,
                "geometry": 1.0, "edge": 1.0, "exposure": 1.0}
    if iteration <= cfg.loss_stage_a_end:
        prefix = "loss_stage_a"
    elif iteration <= cfg.loss_stage_b_end:
        prefix = "loss_stage_b"
    else:
        prefix = "loss_stage_c"
    return {key: float(getattr(cfg, f"{prefix}_{key}")) for key in
            ("l1", "mse", "dssim", "geometry", "edge", "exposure")}


def get_lr_multipliers(iteration, cfg):
    if not getattr(cfg, "loss_schedule_enabled", False):
        return {key: 1.0 for key in ("xyz", "scaling", "rotation", "features", "opacity", "exposure")}
    if iteration <= cfg.loss_stage_a_end:
        prefix = "lr_stage_a"
    elif iteration <= cfg.loss_stage_b_end:
        prefix = "lr_stage_b"
    else:
        prefix = "lr_stage_c"
    return {key: float(getattr(cfg, f"{prefix}_{key}")) for key in
            ("xyz", "scaling", "rotation", "features", "opacity", "exposure")}
