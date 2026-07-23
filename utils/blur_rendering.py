"""Render-average camera subposes to model motion/defocus-like training blur."""

from __future__ import annotations

import torch

from scene.pose_refinement import RefinedCamera


def render_with_blur_formation(camera, gaussian_model, blur_model, renderer, cfg):
    """Render a sharp view once or average a blurry view over bounded subposes."""

    index = int(getattr(camera, "view_index", getattr(camera, "uid", -1)))
    if blur_model is None or not blur_model.is_blurred(index):
        return renderer(camera, gaussian_model)
    num_samples = int(getattr(cfg, "blur_num_subposes", 3))
    packages = []
    for correction in blur_model.subposes(index, num_samples):
        corrected = RefinedCamera(camera, camera.world_view_transform @ correction.transpose(-1, -2))
        packages.append(renderer(corrected, gaussian_model))
    middle = packages[len(packages) // 2]
    output = dict(middle)
    average_keys = (
        "render", "canonical_render", "corrected_render", "depth", "normal",
        "alpha", "uncertainty", "depth_variance", "foreground_render", "background_render")
    for key in average_keys:
        values = [package.get(key) for package in packages]
        if all(value is not None for value in values):
            output[key] = torch.stack(values, dim=0).mean(dim=0)
    output["radii"] = torch.stack([package["radii"] for package in packages], dim=0).max(dim=0).values
    output["visibility_filter"] = torch.stack(
        [package["visibility_filter"] for package in packages], dim=0).any(dim=0)
    output["blur_num_subposes"] = num_samples
    return output

