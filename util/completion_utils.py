import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from Amodal3R_align.inference import inference
from gaussian_renderer import render, render_single_obj
from PIL import Image
from scene import GaussianModel
from scipy import ndimage
from tqdm import tqdm
from util.utils import (
    get_rotation_matrix_6d,
    matrix_to_quaternion_torch,
    quaternion_to_matrix_torch,
)


# Add DINOv2 similarity calculation function
def compute_dinov2_similarity(image1, image2, completion_pipeline, device="cuda"):
    """
    Calculate DINOv2 feature similarity between two images

    Args:
        image1 (torch.Tensor): First image, shape [3, H, W], value range [0, 1]
        image2 (torch.Tensor): Second image, shape [3, H, W], value range [0, 1]
        device (str): Device

    Returns:
        float: Cosine similarity value
    """

    # Load DINOv2 model
    dinov2_model = completion_pipeline.models["image_cond_model"]
    dinov2_model = dinov2_model.to(device).eval()

    # Preprocess images
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Ensure image is 4D tensor [B, C, H, W]
    if image1.dim() == 3:
        image1 = image1.unsqueeze(0)
    if image2.dim() == 3:
        image2 = image2.unsqueeze(0)

    # Normalize
    image1_norm = (image1 - mean) / std
    image2_norm = (image2 - mean) / std

    # Resize to DINOv2 expected size
    image1_resized = F.interpolate(
        image1_norm, size=(518, 518), mode="bilinear", align_corners=False
    )
    image2_resized = F.interpolate(
        image2_norm, size=(518, 518), mode="bilinear", align_corners=False
    )

    with torch.no_grad():
        # Extract features
        features1 = dinov2_model(image1_resized)  # [B, 1024]
        features2 = dinov2_model(image2_resized)  # [B, 1024]

        # Calculate cosine similarity
        features1_norm = F.normalize(features1, p=2, dim=1)
        features2_norm = F.normalize(features2, p=2, dim=1)

        similarity = torch.sum(features1_norm * features2_norm, dim=1)

    return similarity.item()


def wxyz_to_xyzw(q):
    # (..., 4) -> (..., 4)
    return torch.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], dim=-1)


def xyzw_to_wxyz(q):
    # (..., 4) -> (..., 4)
    return torch.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], dim=-1)


@torch.no_grad()
def get_completion_index(
    pos: torch.Tensor,
    seg_label: torch.Tensor,
    segments_info: dict,
    min_area: float = 100.0,
    percentile: float = 5.0,
    focal: float = 512.0,
):
    """
    Compute filtered instance indices based on 3D bounding boxes, sorted by area.

    Args:
        pos (torch.Tensor): Tensor of shape [N, 3] containing 3D point coordinates in camera space
        seg_label (torch.Tensor): Tensor of shape [N] containing segment labels
        segments_info (dict): Dictionary mapping segment labels to semantic labels
        min_area (float): Minimum area threshold in image plane
        percentile (float): Percentile value to use for computing bounding box (default: 5.0)
        focal (float): Camera focal length

    Returns:
        instance_indices (list): List of instance indices sorted by area (largest to smallest)
    """
    from util.utils import ade_fg_classes, ade_id2label

    # Convert to numpy for faster processing
    pos_np = pos.cpu().numpy()
    seg_label_np = seg_label.cpu().numpy()

    valid_indices = []
    valid_areas = []

    # Pre-compute valid semantic classes
    valid_semantic_ids = {
        int(seg_id)
        for seg_id, semantic_id in segments_info.items()
        if ade_id2label[str(semantic_id)] in ade_fg_classes
    }

    # Get unique segment labels
    unique_segments = np.unique(seg_label_np)

    # Process each segment
    for seg_id in unique_segments:
        if seg_id not in valid_semantic_ids:
            continue

        # Get points for this segment
        mask = seg_label_np == seg_id
        if not mask.any():
            continue

        points = pos_np[mask]

        # Compute bounding box using percentiles
        lower = np.percentile(points, percentile, axis=0)
        upper = np.percentile(points, 100 - percentile, axis=0)

        # Calculate area in camera space
        # Convert image plane area to camera space area
        # For a point at depth z, the scale factor is (z/focal)^2
        z = lower[2]  # Use the minimum z value for conservative estimation
        scale_factor = (z / focal) * (z / focal)
        camera_space_area = min_area * scale_factor

        # Calculate actual area in camera space
        area = (upper[0] - lower[0]) * (upper[1] - lower[1])

        # print(area, camera_space_area,ade_id2label[str(segments_info[int(seg_id)])])

        # Skip if area is too small
        if area < camera_space_area:
            continue

        valid_indices.append(int(seg_id))
        valid_areas.append(float(area))  # Store area instead of depth

    if not valid_indices:
        return []

    # Sort by area (largest to smallest)
    sorted_indices = np.argsort(valid_areas)[::-1]  # Reverse to get largest first
    instance_indices = [valid_indices[i] for i in sorted_indices]

    return instance_indices


@torch.no_grad()
def get_completion_index_2d(
    seg_label: torch.Tensor, segments_info: dict, min_area: float = 100.0
):
    """
    Compute filtered instance indices based on 3D bounding boxes, sorted by area.

    Args:
        pos (torch.Tensor): Tensor of shape [N, 3] containing 3D point coordinates in camera space
        seg_label (torch.Tensor): Tensor of shape [N] containing segment labels
        segments_info (dict): Dictionary mapping segment labels to semantic labels
        min_area (float): Minimum area threshold in image plane
        percentile (float): Percentile value to use for computing bounding box (default: 5.0)
        focal (float): Camera focal length

    Returns:
        instance_indices (list): List of instance indices sorted by area (largest to smallest)
    """
    from util.utils import ade_fg_classes, ade_id2label

    seg_label_np = seg_label  # .cpu().numpy()

    valid_indices = []
    valid_areas = []

    # Pre-compute valid semantic classes
    valid_semantic_ids = {
        int(seg_id)
        for seg_id, semantic_id in segments_info.items()
        if ade_id2label[str(semantic_id)] in ade_fg_classes
    }

    # Get unique segment labels
    unique_segments = np.unique(seg_label_np)

    # import pdb;pdb.set_trace()
    # Process each segment
    for seg_id in unique_segments:
        if seg_id not in valid_semantic_ids:
            continue

        # Get points for this segment
        mask = seg_label_np == seg_id
        if not mask.any():
            continue

        area = mask.sum()

        if area < min_area:
            continue

        coords = np.argwhere(mask == 1)

        min_x = coords[:, 1].min()  # Leftmost (minimum column)
        max_x = coords[:, 1].max()  # Rightmost (maximum column)
        # print(seg_id,min_x,max_x)
        if min_x < 15 or max_x > 512 - 15:
            continue

        valid_indices.append(int(seg_id))
        valid_areas.append(float(area))  # Store area instead of depth

    if not valid_indices:
        return []

    # Sort by area (largest to smallest)
    sorted_indices = np.argsort(valid_areas)[::-1]  # Reverse to get largest first
    instance_indices = [valid_indices[i] for i in sorted_indices]

    return instance_indices


def complete_and_align_objects(
    gaussians,
    instance_index_list,
    completion_pipeline,
    sim_image,
    sim_seg,
    tdgs_cam,
    gt_depth,
    opt,
    is_3d_fg=None,
    example="default",
    output_dir="./output",
    metric="lpips",
    similarity_threshold=0.4,  # Add similarity threshold parameter
):
    """
    Complete and align 3D objects using Amodal3R with rollback mechanism.

    Args:
        gaussians (GaussianModel): The Gaussian model containing all points
        instance_index_list (list): List of instance IDs to process
        completion_pipeline: The Amodal3R pipeline for 3D completion
        sim_image (np.ndarray): The rendered image in RGB format
        sim_seg (np.ndarray): The segmentation map
        tdgs_cam: The camera parameters
        gt_depth (torch.Tensor): The depth map
        opt: The Gaussian Splatting parameters
        is_3d_fg (torch.Tensor, optional): Tensor indicating which points are foreground
        example (str): Name of the example for saving files
        output_dir (str): Directory to save output files
        metric (str): The metric to use for loss calculation, either "lpips" or "dinov2"
        similarity_threshold (float): DINOv2 similarity threshold for rollback decision

    Returns:
        tuple: (merged_gaussians, is_3d_fg) - The updated Gaussian model and foreground indicator
    """
    if not len(instance_index_list):
        return gaussians, is_3d_fg

    R = torch.tensor(tdgs_cam.R, device=torch.device("cuda"), dtype=torch.float32)
    T = torch.tensor(tdgs_cam.T, device=torch.device("cuda"), dtype=torch.float32)

    # Create a copy of the original Gaussian model as background
    bg_gaussians = copy.deepcopy(gaussians)

    # Get all foreground object masks and create object-specific masks
    all_fg_masks = []
    object_masks = {}  # Store mask for each object for rollback
    for selected_obj in instance_index_list:
        mask = (gaussians.get_seg_all @ gaussians.codebook.T).argmax(-1)
        obj_mask = mask == selected_obj
        all_fg_masks.append(obj_mask)
        object_masks[selected_obj] = obj_mask

    # Remove all foreground objects to create background-only model
    combined_fg_mask = (
        torch.zeros_like(all_fg_masks[0])
        if all_fg_masks
        else torch.zeros(1, dtype=torch.bool).cuda()
    )
    skips = []

    # Initialize is_3d_fg if not provided
    if is_3d_fg is None:
        is_3d_fg = torch.zeros(
            gaussians.get_xyz_all.shape[0], dtype=torch.int32, device="cuda"
        )

    # Check which objects should be skipped (already 3D)
    for mask in all_fg_masks:
        if (is_3d_fg[mask] > 0).sum() / mask.sum() > 0.5:
            skips.append(True)
            continue
        combined_fg_mask = combined_fg_mask | mask
        skips.append(False)

    # Keep only background points and save pruned parameters for potential rollback
    bg_mask = ~combined_fg_mask
    pruned_params = bg_gaussians.prune_gaussian(bg_mask)
    is_3d_fg = is_3d_fg[bg_mask]

    # Start with background model
    merged_gaussians = bg_gaussians

    render_pkg = render(
        tdgs_cam,
        merged_gaussians,
        opt,
        torch.ones([3]).cuda().float() * 0.7,
    )
    rendered_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
    rendered_image = (rendered_image * 255).astype(np.uint8)
    Image.fromarray(rendered_image).save(
        f"{output_dir}/{example}/bg_{selected_obj}.png"
    )

    # Process each object
    for obj_idx, selected_obj in enumerate(instance_index_list):
        # Skip if already 3D
        if skips[obj_idx]:
            continue

        # Create training image for current object
        obj_mask_np = sim_seg == selected_obj
        image_training = sim_image.copy()
        image_training[obj_mask_np == 0, :] = 0
        tdgs_cam.original_image = (
            torch.from_numpy(image_training).permute(2, 0, 1).float() / 255.0
        )

        # Get object mask and check if it's foreground
        obj_mask = all_fg_masks[obj_idx]
        is_fg = (
            gaussians.is_fg_filter[obj_mask].sum() / obj_mask.sum()
            if obj_mask.sum() > 0
            else 0
        )
        is_fg = is_fg > 0.5

        # Process object points
        gaussian_obj = copy.deepcopy(gaussians)
        points_obj = gaussian_obj.get_xyz_all[obj_mask].detach().cpu().numpy()

        if points_obj.size > 0:
            # Calculate object center, filtering outliers
            camera_position = -R @ T
            distance = ((points_obj - camera_position.cpu().numpy()[None, :]) ** 2).sum(
                -1
            )
            mask_outlier = distance < (distance.mean() + distance.std())

            index = torch.nonzero(obj_mask == True)[:, 0]
            filtered_mask = obj_mask.clone()
            filtered_mask[index] = torch.from_numpy(mask_outlier).bool().cuda()

            center = (
                gaussian_obj.get_xyz_all[filtered_mask].detach().cpu().numpy().mean(0)
            )
            center = torch.from_numpy(center).float().cuda()
        else:
            raise NotImplementedError("No points found for object")

        # Clean up segmentation mask
        obj_mask_np = sim_seg == selected_obj
        obj_mask_np = ndimage.binary_opening(
            obj_mask_np, structure=np.ones((7, 7), dtype=bool)
        )
        y_indices, x_indices = np.where(obj_mask_np > 0)

        if len(y_indices) > 0 and len(x_indices) > 0:
            # Calculate object size in image space
            gt_length = (
                y_indices.max() - y_indices.min()
            )  # max(y_indices.max() - y_indices.min(), x_indices.max() - x_indices.min())
            obj_gt_depth = gt_depth.clone()
            obj_gt_depth[:, obj_mask_np == 0] = 0

            # Run Amodal3R completion
            os.makedirs(f"./amodal_output/{example}/{selected_obj}", exist_ok=True)
            _, pose = inference(
                sim_image,
                obj_mask_np,
                completion_pipeline,
                output_path=f"./amodal_output/{example}/{selected_obj}",
            )

            # Load completed model
            merged_gaussians = GaussianModel(
                sh_degree=0,
                codebook=gaussians.codebook,
                previous_gaussian=merged_gaussians,
            )
            merged_gaussians.load_ply(
                f"./amodal_output/{example}/{selected_obj}/gaussian.ply",
                overide_label=selected_obj,
                is_fg=is_fg,
                is_sky=False,
            )

            means3D = merged_gaussians.get_xyz
            rotation_matrix = get_rotation_matrix_6d(pose)
            curr_means3D = (means3D - means3D.mean(0, keepdim=True)).detach()
            curr_means3D = curr_means3D @ rotation_matrix.T
            curr_means3D = curr_means3D + center

            xyz_cam = curr_means3D @ R + T[None, :]
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)

            x = x / z * tdgs_cam.focal_x + tdgs_cam.image_width / 2.0
            y = y / z * tdgs_cam.focal_y + tdgs_cam.image_height / 2.0

            length = y.max() - y.min()  # max(x.max() - x.min(), y.max() - y.min())
            curr_init_scale = torch.log(torch.tensor(gt_length / length)).float().cuda()

            with torch.no_grad():
                render_pkg = render_single_obj(
                    tdgs_cam,
                    merged_gaussians,
                    opt,
                    torch.zeros([3]).cuda().float(),
                    center,
                    curr_init_scale,
                    pose,
                    render_current=True,
                )

            best_image = render_pkg["render"]
            best_scale = curr_init_scale
            best_pose = pose

            os.makedirs(f"{output_dir}/{example}", exist_ok=True)
            rendered_image = best_image.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)

            Image.fromarray(rendered_image).save(
                f"{output_dir}/{example}/best_match_{selected_obj}.png"
            )

            tmp_gaussian = copy.deepcopy(merged_gaussians)
            scale = torch.exp(best_scale)
            means3D = tmp_gaussian.get_xyz
            scales = tmp_gaussian.get_scaling
            means3D = (means3D - means3D.mean(0, keepdim=True)).detach()
            means3D_scaled = means3D * scale
            rotation_matrix = get_rotation_matrix_6d(best_pose)
            means3D = means3D_scaled @ rotation_matrix.T
            means3D = means3D + center
            scales = scales * scale

            with torch.no_grad():
                tmp_gaussian._xyz.data.copy_(means3D)
                tmp_gaussian._scaling.data.copy_(torch.log(scales))
            render_pkg = render(
                tdgs_cam,
                tmp_gaussian,
                opt,
                torch.ones([3]).cuda().float() * 0.7,
            )
            rendered_image = (
                render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
            )
            rendered_image = (rendered_image * 255).astype(np.uint8)
            Image.fromarray(rendered_image).save(
                f"{output_dir}/{example}/recompose_{selected_obj}_before_trained.png"
            )

            # Train alignment and get similarity
            similarity = train_gaussian_align(
                merged_gaussians,
                tdgs_cam,
                opt,
                center,
                best_scale,
                best_pose,
                obj_gt_depth,
                obj_mask_np,
                completion_pipeline,
            )

            print(f"Object {selected_obj} DINOv2 similarity: {similarity:.4f}")

            # Check if rollback is needed
            if similarity < similarity_threshold:
                print(
                    f"Similarity too low ({similarity:.4f} < {similarity_threshold}), rolling back object {selected_obj}"
                )

                # Save image with rejected label
                render_pkg = render(
                    tdgs_cam,
                    merged_gaussians,
                    opt,
                    torch.ones([3]).cuda().float() * 0.7,
                )
                rendered_image = (
                    render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
                )
                rendered_image = (rendered_image * 255).astype(np.uint8)

                # Also save rejected final rendered image
                Image.fromarray(rendered_image).save(
                    f"{output_dir}/{example}/recompose_rejected_{selected_obj}_{similarity}.png"
                )

                # Rollback: restore original Gaussian parameters for this object
                merged_gaussians.replace_gaussian_params(
                    pruned_params, target_seg_ids=[selected_obj]
                )

                # Update is_3d_fg to reflect rolled back object
                original_obj_count = (object_masks[selected_obj]).sum().item()
                rollback_is_3d_fg = torch.zeros(
                    original_obj_count, dtype=torch.int32, device="cuda"
                )
                is_3d_fg = torch.cat([rollback_is_3d_fg, is_3d_fg], dim=0)

                # Skip subsequent processing for this object
                merged_gaussians.compute_3D_filter([tdgs_cam])
                merged_gaussians.set_inscreen_points_to_visible(tdgs_cam)
                continue

            else:
                print(
                    f"Similarity meets requirement ({similarity:.4f} >= {similarity_threshold}), keeping object {selected_obj}"
                )

            # Update is_3d_fg
            curr_size = merged_gaussians.get_xyz.shape[0]
            new_is_3d_fg = (
                torch.ones(curr_size, dtype=torch.int32, device="cuda") * selected_obj
            )
            is_3d_fg = torch.cat([new_is_3d_fg, is_3d_fg], dim=0)

            # Update visibility
            merged_gaussians.compute_3D_filter([tdgs_cam])
            merged_gaussians.set_inscreen_points_to_visible(tdgs_cam)

    render_pkg = render(
        tdgs_cam,
        merged_gaussians,
        opt,
        torch.ones([3]).cuda().float() * 0.7,
    )
    rendered_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
    rendered_image = (rendered_image * 255).astype(np.uint8)
    Image.fromarray(rendered_image).save(
        f"{output_dir}/{example}/recompose_{selected_obj}.png"
    )

    return merged_gaussians, is_3d_fg


def update_params(gaussians, center, scale, angles, add_center_back=False):
    scale = torch.exp(scale)
    means3D = gaussians.get_xyz
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    ori_center = means3D.mean(0, keepdim=True)
    means3D = (means3D - means3D.mean(0, keepdim=True)).detach()
    means3D_scaled = means3D * scale
    if angles.shape[0] == 2:
        rotation_matrix = get_rotation_matrix_6d(angles)
    else:
        rotation_matrix = torch.eye(3).cuda()  # angles
        print(angles)
    means3D = means3D_scaled @ rotation_matrix.T
    if add_center_back:
        center = ori_center.squeeze()
    means3D = means3D + center
    scales = scales * scale

    # Use new quaternion update method
    # 1. Convert quaternion to rotation matrix
    rotation_local = quaternion_to_matrix_torch(
        rotations
    )  # rotations are in wxyz format

    # 2. Apply global transformation: R_new = R_global @ R_local
    rotation_new = torch.matmul(rotation_matrix.unsqueeze(0), rotation_local)

    # 3. Convert rotation matrix back to quaternion
    rotations_new = matrix_to_quaternion_torch(rotation_new)  # output in wxyz format

    with torch.no_grad():
        gaussians._xyz.data.copy_(means3D)
        gaussians._scaling.data.copy_(torch.log(scales))
        gaussians._rotation.data.copy_(rotations_new)


def train_gaussian_align(
    gaussians,
    viewpoint_cam,
    opt,
    center,
    scale,
    angles,
    gt_depth,
    obj_mask_np,
    completion_pipeline,
    initialize_scaling=True,
):
    """
    Train Gaussian model to align with depth map and image.

    Args:
        gaussians (GaussianModel): The Gaussian model to align
        viewpoint_cam: The camera parameters
        opt: The Gaussian Splatting parameters
        center (torch.Tensor): The object center
        scale (torch.Tensor): The scale parameter
        angles (torch.Tensor): The rotation angles
        gt_depth (torch.Tensor): The target depth map
        obj_mask_np (np.ndarray): Object mask for weighted loss calculation
        initialize_scaling (bool): Whether to initialize scaling
    """
    opt.iterations = 100
    iterable_gauss = range(1, opt.iterations + 1)
    background = torch.zeros([3]).cuda().float()
    gt_image = viewpoint_cam.original_image.cuda()

    # Convert obj_mask_np to torch tensor
    obj_mask = torch.from_numpy(obj_mask_np.astype(np.float32)).cuda()

    center = center.unsqueeze(0)
    center = torch.nn.Parameter(center).requires_grad_(True)
    scale = torch.nn.Parameter(scale).requires_grad_(True)
    angles = torch.nn.Parameter(angles).requires_grad_(True)

    optimizer = torch.optim.Adam([scale, angles, center], lr=0.001)

    def dice_loss(pred, target, smooth=1e-6):
        """Calculate Dice loss"""
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    for iteration in tqdm(iterable_gauss):
        render_pkg = render_single_obj(
            viewpoint_cam,
            gaussians,
            opt,
            background,
            center,
            scale,
            angles,
            render_current=True,
            render_opacity=True,
        )
        image = render_pkg["render"]

        # 1. Weighted depth loss - only calculate regions where obj_mask_np is 1
        depth_pred = render_pkg["depth"]
        weighted_depth_loss = F.mse_loss(
            depth_pred * obj_mask, gt_depth.detach() * obj_mask
        )

        # 2. Dice loss - render_pkg['final_opacity'] and obj_mask
        final_opacity = render_pkg["render_seg"][0]
        dice_loss_val = dice_loss(final_opacity, obj_mask)

        # Total loss
        loss = dice_loss_val + weighted_depth_loss

        loss.backward()

        if iteration < opt.iterations:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # if iteration == opt.iterations:
    update_params(gaussians, center, scale, angles, add_center_back=False)

    # Calculate DINOv2 similarity
    # Ensure image format is correct (convert rendered image to 3,H,W format)
    rendered_for_similarity = image.detach()  # image is in 3,H,W format
    gt_for_similarity = gt_image.detach()  # gt_image is in 3,H,W format

    similarity = compute_dinov2_similarity(
        rendered_for_similarity,
        gt_for_similarity,
        completion_pipeline,
        device=gt_image.device,
    )
    return similarity
