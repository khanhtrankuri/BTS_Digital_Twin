#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation


def _rasterize_auxiliary_feature(rasterizer, means3D, means2D, opacity, scales,
                                 rotations, cov3D_precomp, feature):
    """Alpha-composite an arbitrary per-Gaussian 3-vector.

    The bundled rasterizer already accepts ``colors_precomp``.  Reusing that
    supported interface avoids a CUDA ABI change while remaining fully
    differentiable with respect to Gaussian positions, scale and opacity.
    """
    feature = feature.contiguous()
    image, _, _ = rasterizer(
        means3D=means3D, means2D=means2D, shs=None,
        colors_precomp=feature, opacities=opacity, scales=scales,
        rotations=rotations, cov3D_precomp=cov3D_precomp)
    return image


def _camera_space_depth(points, camera):
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    return (homogeneous @ camera.world_view_transform)[:, 2:3]


def _camera_space_normals(pc, camera):
    rotations = build_rotation(pc.get_rotation)
    min_axis = pc.get_scaling.argmin(dim=-1)
    normal = rotations[torch.arange(rotations.shape[0], device=rotations.device), :, min_axis]
    # Camera transforms in this repository are stored for row-vector use.
    normal = normal @ camera.world_view_transform[:3, :3]
    return torch.nn.functional.normalize(normal, dim=-1, eps=1e-6)

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, render_geometry=False, apply_exposure=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp or apply_exposure:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        # Legacy inverse depth remains available by its explicit name.
        "invdepth": depth_image,
        "depth": depth_image,
        "normal": None,
        "alpha": None,
        }

    if render_geometry:
        # The existing CUDA extension composites arbitrary 3-channel features.
        # Render z, normal and a one-vector against a zero background to obtain
        # alpha-composited depth, normal and alpha without changing the kernel.
        zero_bg_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height), image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx, tanfovy=tanfovy, bg=torch.zeros_like(bg_color),
            scale_modifier=scaling_modifier, viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform, sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center, prefiltered=False, debug=pipe.debug,
            antialiasing=pipe.antialiasing)
        aux_rasterizer = GaussianRasterizer(raster_settings=zero_bg_settings)
        alpha_rgb = _rasterize_auxiliary_feature(aux_rasterizer, means3D, means2D, opacity, scales,
                                                  rotations, cov3D_precomp, torch.ones_like(means3D))
        alpha = torch.clamp(alpha_rgb[:1], 0.0, 1.0)
        z = torch.clamp(_camera_space_depth(means3D, viewpoint_camera), min=1e-6).repeat(1, 3)
        depth_sum = _rasterize_auxiliary_feature(aux_rasterizer, means3D, means2D, opacity, scales,
                                                  rotations, cov3D_precomp, z)[:1]
        normal_sum = _rasterize_auxiliary_feature(aux_rasterizer, means3D, means2D, opacity, scales,
                                                   rotations, cov3D_precomp,
                                                   _camera_space_normals(pc, viewpoint_camera))
        valid = alpha > 1e-6
        depth = torch.where(valid, depth_sum / alpha.clamp_min(1e-6), torch.zeros_like(depth_sum))
        normal = torch.nn.functional.normalize(normal_sum, dim=0, eps=1e-6)
        normal = torch.where(valid.expand_as(normal), normal, torch.zeros_like(normal))
        out.update({"depth": depth, "normal": normal, "alpha": alpha})
    
    return out
