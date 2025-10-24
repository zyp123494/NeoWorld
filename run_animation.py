import json
import os
import random
import warnings
from argparse import ArgumentParser

import imageio
import numpy as np
import torch
import utils3d
from Amodal3R_align.Amodal3R.pipelines import Amodal3RImageTo3DPipeline
from arguments import GSParams
from gaussian_renderer import render, render_precomp
from models.models import KeyframeGen
from omegaconf import OmegaConf
from PIL import Image
from scene import GaussianModel
from scene.cameras import Camera
from util.animation_prompt_gen import AnimationPromptGen
from util.completion_utils import complete_and_align_objects
from util.utils import get_rotation_matrix

warnings.filterwarnings("ignore")

image_path = None
mask_path = None
segments_info = None
PROMPT = None
camera_idx = 0
latest_dir = None

camera_path = None
xyz_scale = 1000


def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"running with seed: {seed}.")


def get_render_cameras(num_frames, r=2, focal=512, H=512, W=512):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitchs = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitchs = pitchs.tolist()

    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]

    cameras = []
    for yaw, pitch in zip(yaws, pitchs, strict=False):
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = (
            torch.tensor(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                ]
            ).cuda()
            * r
        )
        w2c = utils3d.torch.extrinsics_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().cuda(),
            torch.tensor([0, 0, 1]).float().cuda(),
        )
        w2c = w2c.cpu().numpy()
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        focal_length = focal
        half_img_size_x = W / 2
        fovx = 2 * np.arctan(half_img_size_x / focal_length)
        half_img_size_y = H / 2
        fovy = 2 * np.arctan(half_img_size_y / focal_length)
        tdgs_cam = Camera(R=R, T=T, FoVx=fovx, FoVy=fovy)
        cameras.append(tdgs_cam)
    return cameras


def generate_circular_trajectory(center_point, radius, round=4, num_points=100):
    x0, y0, z0 = center_point
    y0 = y0 + radius
    theta = np.linspace(0, 2 * np.pi * round, num_points)
    x = x0 + radius * np.sin(theta)
    y = y0 - radius * np.cos(theta)
    z = np.full_like(theta, z0)
    return np.stack([x, y, z], axis=-1)


@torch.no_grad()
def compute_3d_bbox(
    pos: torch.Tensor,
    seg_label: torch.Tensor,
    segments_info: dict,
    threshold: float = 1e-6,
):
    """
    Compute axis-aligned 3D bounding boxes for each segment defined in segments_info.

    Args:
        pos (torch.Tensor): Tensor of shape [N, 3] containing 3D point coordinates.
        seg_label (torch.Tensor): Tensor of shape [N] containing segment labels for each point.
        segments_info (dict): Dictionary mapping segment labels (int) to semantic labels.
        threshold (float): Minimum volume threshold to keep a bounding box.

    Returns:
        boxes (torch.Tensor): Tensor of shape [M, 3, 2] containing bounding boxes,
                              where M is number of valid boxes.
                              Each box is [[x_min,x_max], [y_min,y_max], [z_min,z_max]].
        semantic_labels (list): List of length M of semantic labels for each box.
    """

    from util.utils import ade_id2label

    # Validate input shapes
    assert pos.ndim == 2 and pos.shape[1] == 3, "pos must be of shape [N,3]"
    assert seg_label.ndim == 1 and seg_label.shape[0] == pos.shape[0], (
        "seg_label must be of shape [N] to match pos"
    )
    seg_label = seg_label.cpu().numpy()

    boxes = []
    semantic_labels = []

    volumes = []

    # Iterate over each segment label defined in segments_info
    for seg_id, semantic_label in segments_info.items():
        # Find points belonging to this segment
        mask = seg_label == int(seg_id)

        # Skip if no points for this segment
        if not mask.any():
            continue

        points = pos[mask]
        # Move to CPU and NumPy for percentile computation (if needed)
        points_np = points.cpu().numpy()

        # Compute 5th and 95th percentiles for each dimension to remove outliers
        lower = np.percentile(points_np, 5, axis=0)
        upper = np.percentile(points_np, 95, axis=0)

        # Filter points to those within [lower, upper] in all dimensions
        inliers = np.all(points_np >= lower, axis=1) & np.all(
            points_np <= upper, axis=1
        )
        filtered = points_np[inliers]

        # Skip if no points remain after filtering
        if filtered.shape[0] == 0:
            continue

        # Compute min and max along each dimension
        mins = filtered.min(axis=0)
        maxs = filtered.max(axis=0)

        # Compute volume of bounding box
        dims = maxs - mins
        volume = float(dims[0] * dims[1] * dims[2])
        volumes.append(volume)

        # Discard boxes with negligible volume
        if volume < threshold:
            continue

        # Form the bounding box as [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        box = [
            [float(mins[0]), float(maxs[0])],
            [float(mins[1]), float(maxs[1])],
            [float(mins[2]), float(maxs[2])],
        ]
        boxes.append(box)
        semantic_labels.append((seg_id, ade_id2label[str(semantic_label)]))

    # Convert list of boxes to a torch.Tensor of shape [M, 3, 2]
    if len(boxes) == 0:
        boxes_tensor = torch.empty((0, 3, 2), dtype=pos.dtype)
    else:
        boxes_tensor = torch.tensor(boxes, dtype=pos.dtype)

    return boxes_tensor, semantic_labels


def run(config):
    seeding(config["seed"])
    example = config["example_name"]
    opt = GSParams()

    # Initialize completion pipeline
    completion_pipeline = Amodal3RImageTo3DPipeline.from_pretrained(
        "Sm0kyWu/Amodal3R",
        ss_path=config["ss_path"],
    )
    completion_pipeline.cuda()

    # Set global parameters based on config
    global image_path, mask_path, segments_info, PROMPT, camera_path, latest_dir
    name = config["name"]

    # Find the latest output directory with required files
    output_dir = f"output/{name}"
    if os.path.exists(output_dir):
        subdirs = [
            d
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d))
        ]
        valid_dirs = []

        for subdir in subdirs:
            subdir_path = os.path.join(output_dir, subdir)
            segments_file = os.path.join(subdir_path, "segments_info.json")
            w2c_file = f"./output/{example}/{subdir}/w2c_transforms.npy"

            if os.path.exists(segments_file) and os.path.exists(w2c_file):
                valid_dirs.append(subdir)

        if valid_dirs:
            latest_dir = sorted(valid_dirs)[-1]
            image_path = f"{output_dir}/{latest_dir}/gaussian_scene00.png"
            mask_path = f"{output_dir}/{latest_dir}/gaussian_scene00_seg.png"
            if not os.path.exists(image_path):
                image_path = f"{output_dir}/{latest_dir}/gaussian_scene_layer00.png"
                mask_path = f"{output_dir}/{latest_dir}/gaussian_scene_layer00_seg.png"
            segments_info = f"{output_dir}/{latest_dir}/segments_info.json"
            # Load camera parameters
            camera_path = f"./output/{example}/{latest_dir}/w2c_transforms.npy"
        else:
            raise ValueError(
                f"No subdirectories with both segments_info.json and w2c_transforms.npy found in {output_dir}"
            )
    else:
        raise ValueError(f"output directory {output_dir} does not exist")

    PROMPT = config.get("prompt", "Rotate the left boat vertically for two circle.")

    with open(segments_info) as f:
        segments_info = json.load(f)

    camera_idx = int(image_path[-6:-4])

    w2c = np.load(camera_path)
    w2c = w2c[camera_idx]
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]
    c2w = np.linalg.inv(w2c)
    camera_position = c2w[:3, 3]

    # Load image and mask
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))

    # Initialize KeyframeGen
    kf_gen = KeyframeGen(
        config=config,
        inpainter_pipeline=None,
        mask_generator=None,
        depth_model=None,
        rotation_path=config["rotation_path"][: config["num_scenes"]],
        inpainting_resolution=config["inpainting_resolution_gen"],
        makedir=False,
    ).to(config["device"])

    # Set camera parameters
    focal_length = kf_gen.init_focal_length
    half_img_size_x = kf_gen.inpainting_resolution / 2
    fovx = 2 * np.arctan(half_img_size_x / focal_length)
    half_img_size_y = kf_gen.inpainting_resolution / 2
    fovy = 2 * np.arctan(half_img_size_y / focal_length)

    # Create camera
    tdgs_cam = Camera(
        R=R,
        T=T,
        FoVx=fovx,
        FoVy=fovy,
        image=torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
    )

    R = torch.tensor(tdgs_cam.R, device=torch.device("cuda"), dtype=torch.float32)
    T = torch.tensor(tdgs_cam.T, device=torch.device("cuda"), dtype=torch.float32)

    # Load 3DGS model
    gaussians = GaussianModel(
        sh_degree=0, codebook=torch.load(f"examples/sky_images/{example}/codebook.pth")
    )
    gaussians.load_ply_with_filter(f"examples/sky_images/{example}/finished_3dgs.ply")
    gaussians.visibility_filter_all = torch.load(
        f"examples/sky_images/{example}/visibility_filter_all.pth"
    ).to("cuda")
    gaussians.is_sky_filter = torch.load(
        f"examples/sky_images/{example}/is_sky_filter.pth"
    ).to("cuda")
    gaussians.delete_mask_all = torch.load(
        f"examples/sky_images/{example}/delete_mask_all.pth"
    ).to("cuda")
    gaussians.is_fg_filter = torch.load(
        f"examples/sky_images/{example}/is_fg_filter.pth"
    )

    xyz_cam = gaussians.get_xyz_all @ R + T[None, :]
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]

    in_plane = z > 0
    z = torch.clamp(z, min=0.001)

    x = x / z * tdgs_cam.focal_x + tdgs_cam.image_width / 2.0
    y = y / z * tdgs_cam.focal_y + tdgs_cam.image_height / 2.0
    # filter out of image points
    visible_mask = (
        (x >= 0)
        & (x <= tdgs_cam.image_width)
        & (y >= 0)
        & (y <= tdgs_cam.image_height)
        & (in_plane)
    )
    seg_label = (gaussians.get_seg_all[visible_mask] @ gaussians.codebook.T).argmax(
        -1
    )  # 【N，C】

    # compute 3d bbox for each seg label, with format [[x_min,x_max,y_min,y_max,z_min,z_max]]
    bbox_3d, semantic_label = compute_3d_bbox(
        xyz_cam[visible_mask], seg_label, segments_info
    )
    # filter out bbox_3d that is too small

    results = AnimationPromptGen().animate(bbox_3d, semantic_label, PROMPT)

    results = results["objects"]
    print(results)

    # Store animation parameters for all objects
    all_translation = []
    all_rotation = []

    # Get instance list for completion
    instance_index_list = [int(result["instance_id"]) for result in results]

    # Use completion_utils to complete and align objects
    sim_seg = np.array(Image.open(mask_path))
    render_pkg = render(
        tdgs_cam, gaussians, opt, bg_color=0.7 * torch.ones([3]).cuda().float()
    )
    gt_depth = render_pkg["median_depth"]

    # Call complete_and_align_objects function
    merged_gaussians, is_3d_fg = complete_and_align_objects(
        gaussians=gaussians,
        instance_index_list=instance_index_list,
        completion_pipeline=completion_pipeline,
        sim_image=image,
        sim_seg=sim_seg,
        tdgs_cam=tdgs_cam,
        gt_depth=gt_depth,
        opt=opt,
        is_3d_fg=None,
        example=example,
        output_dir="./output",
        similarity_threshold=-10,  # disable rollback in such case
    )

    # Extract translation and rotation for animation
    for obj_idx, result in enumerate(results):
        selected_obj = int(result["instance_id"])

        # Convert translation from camera coordinate system to world coordinate system
        translation_world = []
        for trans in result["translation"]:
            trans_tensor = torch.tensor(trans).float().cuda()
            trans_world = (
                R @ trans_tensor
            )  # Use camera rotation matrix R to convert to world coordinate system
            translation_world.append(trans_world.cpu().numpy().tolist())

        # Rotation remains unchanged as it is local Euler angle rotation
        rotation = result["rotation"]

        # Record current object information
        all_translation.append(translation_world)
        all_rotation.append(rotation)

    # If no objects were processed (results is empty), return directly
    if not results:
        print("No objects were processed.")
        raise NotImplementedError

    # Save final merged Gaussian model
    print("Saving merged model...")

    merged_gaussians.save_ply_all_with_filter(
        f"examples/sky_images/{example}/finished_3dgs_completion.ply"
    )
    torch.save(
        merged_gaussians.visibility_filter_all,
        f"examples/sky_images/{example}/visibility_filter_all_completion.pth",
    )
    torch.save(
        merged_gaussians.codebook,
        f"examples/sky_images/{example}/codebook_completion.pth",
    )
    torch.save(
        merged_gaussians.is_fg_filter,
        f"examples/sky_images/{example}/is_fg_filter_completion.pth",
    )
    torch.save(
        merged_gaussians.is_sky_filter,
        f"examples/sky_images/{example}/is_sky_filter_completion.pth",
    )
    torch.save(
        merged_gaussians.delete_mask_all,
        f"examples/sky_images/{example}/delete_mask_all_completion.pth",
    )

    # Save is_3d_fg tensor
    torch.save(is_3d_fg, f"examples/sky_images/{example}/is_3d_fg.pth")

    return all_translation, all_rotation


def run_simulation(config, translation=None, rotation=None):
    seeding(config["seed"])
    example = config["example_name"]
    opt = GSParams()

    # Load camera parameters
    w2c = np.load(camera_path)
    w2c = w2c[camera_idx]
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]
    c2w = np.linalg.inv(w2c)
    camera_position = c2w[:3, 3]

    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))

    # Initialize KeyframeGen
    kf_gen = KeyframeGen(
        config=config,
        inpainter_pipeline=None,
        mask_generator=None,
        depth_model=None,
        rotation_path=[2, -2],
        makedir=False,
        inpainting_resolution=config["inpainting_resolution_gen"],
    ).to(config["device"])

    # Set camera parameters
    focal_length = kf_gen.init_focal_length
    half_img_size_x = kf_gen.inpainting_resolution / 2
    fovx = 2 * np.arctan(half_img_size_x / focal_length)
    half_img_size_y = kf_gen.inpainting_resolution / 2
    fovy = 2 * np.arctan(half_img_size_y / focal_length)

    # Create camera
    tdgs_cam = Camera(R=R, T=T, FoVx=fovx, FoVy=fovy, image=torch.zeros([3, 512, 512]))

    # Load 3DGS model
    gaussians = GaussianModel(
        sh_degree=0,
        codebook=torch.load(f"examples/sky_images/{example}/codebook_completion.pth"),
    )
    gaussians.load_ply_with_filter(
        f"examples/sky_images/{example}/finished_3dgs_completion.ply"
    )
    gaussians.visibility_filter_all = torch.load(
        f"examples/sky_images/{example}/visibility_filter_all_completion.pth"
    ).to("cuda")
    gaussians.is_sky_filter = torch.load(
        f"examples/sky_images/{example}/is_sky_filter_completion.pth"
    ).to("cuda")
    gaussians.delete_mask_all = torch.load(
        f"examples/sky_images/{example}/delete_mask_all_completion.pth"
    ).to("cuda")
    gaussians.is_fg_filter = torch.load(
        f"examples/sky_images/{example}/is_fg_filter_completion.pth"
    )

    # Load is_3d_fg tensor
    is_3d_fg = torch.load(f"examples/sky_images/{example}/is_3d_fg.pth").to("cuda")

    print("num objects:", torch.unique(is_3d_fg).shape[0] - 1)

    # Calculate centers of all foreground objects
    fg_centers = []
    means3D_fg_list = []
    covariance_fg_list = []

    # Extract point cloud for each object
    for obj_idx in torch.unique(is_3d_fg):
        if obj_idx == 0:
            continue
        obj_mask = is_3d_fg == obj_idx
        if obj_mask.sum() > 0:
            points = gaussians.get_xyz[obj_mask].clone()
            means3D_fg_list.append(points)
            covariance_fg_list.append(gaussians.get_covariance_all()[obj_mask].clone())
            # Calculate object center
            center = points.mean(dim=0)
            fg_centers.append(center.detach().cpu().numpy())

    # Get background point cloud
    bg_mask = is_3d_fg == 0
    means3D_bg = gaussians.get_xyz[bg_mask].clone()
    means3D_all = gaussians.get_xyz.clone()

    covariance_bg = gaussians.get_covariance_all()[bg_mask].clone()
    covariance_all = gaussians.get_covariance_all().clone()

    # 如果有前景物体，计算它们的整体中心
    if fg_centers:
        fg_center = np.mean(fg_centers, axis=0)
    else:
        # 如果没有前景物体，使用当前相机位置
        fg_center = T

    # 创建绕前景物体中心旋转的相机轨迹
    frames = 200

    # 使用当前相机位置作为起点
    # 计算半径 - 当前相机位置到前景物体中心在xz平面的距离
    r = np.sqrt((T[0] - fg_center[0]) ** 2 + (T[2] - fg_center[2]) ** 2)

    # 计算起始角度 - 当前相机位置相对于前景物体中心的角度
    start_angle = np.arctan2(T[2] - fg_center[2], T[0] - fg_center[0])

    novel_view_cam = []

    # 记录原始相机的y坐标
    original_y = T[1]

    for i in range(frames):
        # 计算角度，从起始角度开始旋转一周
        angle = start_angle + i * 2 * np.pi / frames

        # 计算新的相机位置（保持y坐标不变）
        new_x = fg_center[0] + r * np.cos(angle)
        new_z = fg_center[2] + r * np.sin(angle)
        new_pos = np.array([new_x, original_y, new_z])

        # 计算朝向前景中心的旋转矩阵
        # 使用look_at函数生成从相机位置朝向前景中心的旋转矩阵
        forward = fg_center - new_pos
        forward[1] = 0
        forward = forward / np.linalg.norm(forward)

        # 定义相机的上方向（通常是y轴正方向）
        up = np.array([0, 1, 0])

        # 计算相机坐标系的右向量
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        # 重新计算上向量以确保正交
        up = np.cross(right, forward)

        # 构建旋转矩阵
        new_R = np.stack(
            [right, -up, forward], axis=1
        )  # Note camera looks towards -forward direction

        # Create camera
        novel_view_cam.append(
            Camera(
                R=new_R,
                T=new_pos,
                FoVx=fovx,
                FoVy=fovy,
                image=torch.zeros([3, 512, 512]),
            )
        )

    # 物理模拟渲染
    print("Render Animation Videos")
    with torch.no_grad():
        imgs = []
        imgs_fixed = []

        # 获取所有帧数
        max_frames = 100  # animation_prompt_gen.py generates 100 frames of animation

        # 保存原始点云和协方差
        original_means3D = means3D_all.clone()
        original_covariance = covariance_all.clone()

        for i in range(max_frames):
            # 更新所有物体的点云
            for k, obj_idx in enumerate(torch.unique(is_3d_fg)):
                if obj_idx == 0:
                    continue

                # 获取当前帧的translation和rotation
                obj_trans = translation[k - 1][i]
                obj_rot = rotation[k - 1][i]

                # 将旋转角度转换为弧度
                obj_rot_rad = [angle * np.pi / 180 for angle in obj_rot]

                # 获取物体掩码
                obj_mask = is_3d_fg == obj_idx

                if obj_mask.sum() > 0:
                    # 获取物体原始点云
                    points = original_means3D[obj_mask].clone()
                    covs = original_covariance[obj_mask].clone()

                    # Calculate object center
                    center = points.mean(dim=0)

                    # 移动点云到原点
                    centered_points = points - center

                    # 应用旋转
                    rotation_matrix = get_rotation_matrix(
                        torch.tensor(obj_rot_rad).float().cuda()
                    )
                    rotated_points = centered_points @ rotation_matrix.T

                    # 移动回原位，并应用世界坐标系中的translation
                    translated_points = (
                        rotated_points + center + torch.tensor(obj_trans).float().cuda()
                    )

                    # 更新物体点云
                    means3D_all[obj_mask] = translated_points

                    # 保持协方差不变（简化处理）
                    covariance_all[obj_mask] = covs

            # 使用对应帧的相机渲染图像
            frame_idx = min(i, len(novel_view_cam) - 1)  # Prevent index out of range
            with torch.no_grad():
                visibility_filter_all = ~gaussians.delete_mask_all

                # 渲染动态视角
                render_pkg = render_precomp(
                    novel_view_cam[frame_idx],
                    gaussians,
                    means3D_all,
                    gaussians.get_opacity,
                    gaussians.get_features,
                    covariance_all,
                    visibility_filter_all,
                    opt,
                    0.7 * torch.ones([3]).cuda().float(),
                )
                image = render_pkg["render"]

                rendered_image = image.permute(1, 2, 0).detach().cpu().numpy()
                rendered_image = (rendered_image * 255).astype(np.uint8)
                imgs.append(rendered_image)

                # 渲染固定视角
                render_pkg = render_precomp(
                    tdgs_cam,
                    gaussians,
                    means3D_all,
                    gaussians.get_opacity,
                    gaussians.get_features,
                    covariance_all,
                    visibility_filter_all,
                    opt,
                    0.7 * torch.ones([3]).cuda().float(),
                )
                image = render_pkg["render"]

                rendered_image = image.permute(1, 2, 0).detach().cpu().numpy()
                rendered_image = (rendered_image * 255).astype(np.uint8)
                imgs_fixed.append(rendered_image)

        # 保存视频
        if imgs:
            imgs = np.array(imgs)
            imageio.mimwrite(
                f"./output/{example}/{latest_dir}/{PROMPT}.mp4",
                imgs,
                fps=24,
            )

            imageio.mimwrite(
                f"./output/{example}/{latest_dir}/{PROMPT}_fixed.gif",
                imgs_fixed,
                format="GIF",
                fps=24,
                loop=0,
            )

            imgs_fixed = np.array(imgs_fixed)
            imageio.mimwrite(
                f"./output/{example}/{latest_dir}/{PROMPT}_fixed.mp4",
                imgs_fixed,
                fps=24,
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name for example config (will be loaded from config/more_examples/{name}.yaml)",
    )
    parser.add_argument(
        "--prompt",
        default="Rotate the left boat vertically for two circle.",
        help="Prompt for animation",
    )

    args = parser.parse_args()
    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(f"config/more_examples/{args.name}.yaml")
    config = OmegaConf.merge(base_config, example_config)

    # Add name and prompt to config
    config.name = args.name
    config.prompt = args.prompt

    POSTMORTEM = config["debug"]
    if POSTMORTEM:
        try:
            print("Step 1: Running 3D Completion...")
            translation, rotation = run(config)
            print("Step 2: Running Animation Rendering...")
            run_simulation(config, translation, rotation)
        except Exception as e:
            print(e)
            import ipdb

            ipdb.post_mortem()
    else:
        print("Step 1: Running 3D Completion...")
        translation, rotation = run(config)
        print("Step 2: Running Animation Rendering...")
        run_simulation(config, translation, rotation)
