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
import math

import torch
import torch.nn.functional as F
from depth_diff_gaussian_rasterization_min_features import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from util.utils import (
    get_rotation_matrix_6d,
)


def render(
    viewpoint_camera,
    pc: GaussianModel,
    opt,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    render_visible=False,
    render_seg=False,
    exclude_sky=False,
    exclude_fg=False,
    render_current=False,
    fg_only=False,
    remove_list=[],
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz_all,
            dtype=pc.get_xyz_all.dtype,
            requires_grad=True,
            device="cuda",
        )
        + 0
    )
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
        debug=opt.debug,
        include_feature=render_seg,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz_all
    means2D = screenspace_points

    opacity = pc.get_opacity_all

    segs = pc.get_seg_all

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if opt.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance_all(scaling_modifier)
    else:
        scales = pc.get_scaling_all
        rotations = pc.get_rotation_all

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if opt.convert_SHs_python:
            shs_view = pc.get_features_all.transpose(1, 2).view(-1, 3)
            colors_precomp = pc.color_activation(shs_view)
        else:
            shs = pc.get_features_all
    else:
        colors_precomp = override_color

    if render_visible:
        visibility_filter_all = (
            pc.visibility_filter_all & ~pc.delete_mask_all
        )  # Seen in screen
    else:
        visibility_filter_all = ~pc.delete_mask_all

    if exclude_sky:
        visibility_filter_all = visibility_filter_all & ~pc.is_sky_filter

    if fg_only:
        visibility_filter_all = visibility_filter_all & pc.is_fg_filter

    if exclude_fg:
        # print(pc.is_fg_filter)
        visibility_filter_all = visibility_filter_all & ~pc.is_fg_filter
    if render_current:
        current_mask = torch.zeros_like(means3D[:, 0]).bool()
        current_mask[: pc.get_xyz.shape[0]] = True
        visibility_filter_all = visibility_filter_all & current_mask

    if len(remove_list):
        label = segs.argmax(-1)
        remove_mask = torch.zeros_like(means3D[:, 0]).bool()
        for k in remove_list:
            remove_mask[label == k] = 1
        visibility_filter_all = visibility_filter_all & (~remove_mask)

    means3D = means3D[visibility_filter_all]
    means2D = means2D[visibility_filter_all]
    shs = None if shs is None else shs[visibility_filter_all]
    colors_precomp = (
        None if colors_precomp is None else colors_precomp[visibility_filter_all]
    )
    opacity = opacity[visibility_filter_all]
    scales = scales[visibility_filter_all]
    rotations = rotations[visibility_filter_all]
    cov3D_precomp = (
        None if cov3D_precomp is None else cov3D_precomp[visibility_filter_all]
    )
    segs = segs[visibility_filter_all]

    # # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if render_seg:
        rendered_image, rendered_seg, radii, depth, median_depth, final_opacity = (
            rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                language_feature_precomp=segs,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None if cov3D_precomp is None else cov3D_precomp,
            )
        )

    else:
        rendered_image, _, radii, depth, median_depth, final_opacity = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        rendered_seg = None

    return {
        "render": rendered_image,
        "render_seg": rendered_seg,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "final_opacity": final_opacity,
        "depth": depth,
        "median_depth": median_depth,
    }


def wxyz_to_xyzw(q):
    # (..., 4) -> (..., 4)
    return torch.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], dim=-1)


def xyzw_to_wxyz(q):
    # (..., 4) -> (..., 4)
    return torch.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], dim=-1)


def render_single_obj(
    viewpoint_camera,
    pc: GaussianModel,
    opt,
    bg_color: torch.Tensor,
    center,
    scale,
    angles,
    scaling_modifier=1.0,
    override_color=None,
    render_visible=False,
    render_seg=False,
    exclude_sky=False,
    exclude_fg=False,
    render_current=False,
    render_opacity=False,
    remove_list=[],
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz_all,
            dtype=pc.get_xyz_all.dtype,
            requires_grad=True,
            device="cuda",
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # import pdb;pdb.set_trace()
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
        debug=opt.debug,
        include_feature=render_seg or render_opacity,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means3D = pc.get_xyz_all
    means2D = screenspace_points
    # opacity = pc.get_opacity_with_3D_filter
    opacity = pc.get_opacity_all

    segs = pc.get_seg_all

    # opacity = pc.get_opacity_with_3D_filter_all

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if opt.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance_all(scaling_modifier)
    else:
        scales = pc.get_scaling_all
        rotations = pc.get_rotation_all

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if opt.convert_SHs_python:
            shs_view = pc.get_features_all.transpose(1, 2).view(-1, 3)
            colors_precomp = pc.color_activation(shs_view)
        else:
            shs = pc.get_features_all
    else:
        colors_precomp = override_color

    if render_visible:
        visibility_filter_all = (
            pc.visibility_filter_all & ~pc.delete_mask_all
        )  # Seen in screen
    else:
        visibility_filter_all = ~pc.delete_mask_all

    if exclude_sky:
        visibility_filter_all = visibility_filter_all & ~pc.is_sky_filter

    if exclude_fg:
        # print(pc.is_fg_filter)
        visibility_filter_all = visibility_filter_all & ~pc.is_fg_filter
    if render_current:
        current_mask = torch.zeros_like(means3D[:, 0]).bool()
        current_mask[: pc.get_xyz.shape[0]] = True
        visibility_filter_all = visibility_filter_all & current_mask

    if len(remove_list):
        label = segs.argmax(-1)
        remove_mask = torch.zeros_like(means3D[:, 0]).bool()
        for k in remove_list:
            remove_mask[label == k] = 1
        visibility_filter_all = visibility_filter_all & (~remove_mask)

    means3D = means3D[visibility_filter_all]
    means2D = means2D[visibility_filter_all]
    shs = None if shs is None else shs[visibility_filter_all]
    colors_precomp = (
        None if colors_precomp is None else colors_precomp[visibility_filter_all]
    )
    opacity = opacity[visibility_filter_all]
    scales = scales[visibility_filter_all]
    rotations = rotations[visibility_filter_all]
    cov3D_precomp = (
        None if cov3D_precomp is None else cov3D_precomp[visibility_filter_all]
    )
    segs = segs[visibility_filter_all]

    scale = torch.exp(scale)
    means3D = (means3D - means3D.mean(0, keepdim=True)).detach()
    means3D_scaled = means3D * scale

    means3D = means3D_scaled @ get_rotation_matrix_6d(angles).T
    means3D = means3D + center

    scales = scales * scale

    if render_seg or render_opacity:
        rendered_image, rendered_seg, radii, depth, median_depth, final_opacity = (
            rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                language_feature_precomp=torch.ones_like(segs)
                if render_opacity
                else segs,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None if cov3D_precomp is None else cov3D_precomp,
            )
        )

    else:
        rendered_image, _, radii, depth, median_depth, final_opacity = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        rendered_seg = None

    return {
        "render": rendered_image,
        "render_seg": rendered_seg,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "final_opacity": final_opacity,
        "depth": depth,
        "median_depth": median_depth,
    }


def render_with_mask(
    viewpoint_camera,
    pc: GaussianModel,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    render_visible=False,
    render_seg=False,
    mask=None,
    center=None,
    scale=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz_all,
            dtype=pc.get_xyz_all.dtype,
            requires_grad=True,
            device="cuda",
        )
        + 0
    )
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
        debug=False,
        include_feature=render_seg,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz_all
    means2D = screenspace_points
    opacity = pc.get_opacity_all
    segs = pc.get_seg_all

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    scales = pc.get_scaling_all
    rotations = pc.get_rotation_all

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    shs = pc.get_features_all

    if render_visible:
        visibility_filter_all = (
            pc.visibility_filter_all & ~pc.delete_mask_all
        )  # Seen in screen
    else:
        visibility_filter_all = ~pc.delete_mask_all

    if mask is not None:
        visibility_filter_all = visibility_filter_all & mask

    if center is not None:
        means3D -= center.unsqueeze(0)
    if scale is not None:
        means3D /= scale
        scales /= scale
    means3D = means3D[visibility_filter_all]
    means2D = means2D[visibility_filter_all]
    shs = None if shs is None else shs[visibility_filter_all]
    colors_precomp = (
        None if colors_precomp is None else colors_precomp[visibility_filter_all]
    )
    opacity = opacity[visibility_filter_all]
    scales = scales[visibility_filter_all]
    rotations = rotations[visibility_filter_all]
    cov3D_precomp = (
        None if cov3D_precomp is None else cov3D_precomp[visibility_filter_all]
    )
    segs = segs[visibility_filter_all]

    # # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if render_seg:
        rendered_image, rendered_seg, radii, depth, median_depth, final_opacity = (
            rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                language_feature_precomp=segs,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None if cov3D_precomp is None else cov3D_precomp,
            )
        )

    else:
        rendered_image, _, radii, depth, median_depth, final_opacity = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        rendered_seg = None

    return {
        "render": rendered_image,
        "render_seg": rendered_seg,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "final_opacity": final_opacity,
        "depth": depth,
        "median_depth": median_depth,
    }


def render_precomp(
    viewpoint_camera,
    pc,
    means3D,
    opacity,
    shs,
    covariance,
    visibility_filter_all,
    opt,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = (
        torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
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
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=opt.debug,
        include_feature=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from

    cov3D_precomp = covariance

    shs_view = shs.transpose(1, 2).view(-1, 3)
    colors_precomp = pc.color_activation(shs_view)
    shs = None

    means3D = means3D[visibility_filter_all]
    means2D = means2D[visibility_filter_all]
    shs = None if shs is None else shs[visibility_filter_all]
    colors_precomp = (
        None if colors_precomp is None else colors_precomp[visibility_filter_all]
    )
    opacity = opacity[visibility_filter_all]

    cov3D_precomp = (
        None if cov3D_precomp is None else cov3D_precomp[visibility_filter_all]
    )

    rendered_image, _, radii, depth, median_depth, final_opacity = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=None,
        rotations=None,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "final_opacity": final_opacity,
        "depth": depth,
        "median_depth": median_depth,
    }
