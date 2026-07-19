"""Opt-in BTS-GeoGS-v2 loss and learning-rate schedules."""

from __future__ import annotations


def _ordered_pairs(values, first_key, second_key):
    pairs = []
    for value in values:
        if isinstance(value, dict):
            pair = (value[first_key], value[second_key])
        else:
            pair = value
        pairs.append((int(pair[0]), float(pair[1])))
    if not pairs or pairs[0][0] != 0:
        raise ValueError("A progressive schedule must start at iteration 0")
    if any(current[0] >= following[0] for current, following in zip(pairs, pairs[1:])):
        raise ValueError("Schedule iterations must be strictly increasing")
    return pairs


def get_sh_degree(iteration, cfg, max_degree):
    """Return configured active SH degree, or ``None`` for legacy behavior."""

    if not getattr(cfg, "sh_schedule_enabled", False):
        return None
    milestones = _ordered_pairs(getattr(cfg, "sh_schedule_milestones"), "START_ITER", "DEGREE")
    degree = int(milestones[0][1])
    for start, value in milestones:
        if iteration < start:
            break
        degree = int(value)
    if degree < 0 or degree > int(max_degree):
        raise ValueError(f"SH schedule selected degree {degree}, max is {max_degree}")
    return degree


def get_resolution_stage(iteration, cfg):
    """Return ``(stage_index, image_scale)`` for cached camera resolution."""

    if not getattr(cfg, "resolution_schedule_enabled", False):
        return 0, 1.0
    stages = _ordered_pairs(getattr(cfg, "resolution_schedule_stages"), "START_ITER", "SCALE")
    stage_index, scale = 0, stages[0][1]
    for index, (start, value) in enumerate(stages):
        if iteration < start:
            break
        stage_index, scale = index, value
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"Resolution scale must be in (0,1], got {scale}")
    return stage_index, float(scale)


def resolution_cache_scales(cfg):
    """Return ``Scene`` resolution keys needed by the configured schedule."""

    if not getattr(cfg, "resolution_schedule_enabled", False):
        return [1.0]
    stages = _ordered_pairs(getattr(cfg, "resolution_schedule_stages"), "START_ITER", "SCALE")
    return sorted({1.0 / float(scale) for _, scale in stages})

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
