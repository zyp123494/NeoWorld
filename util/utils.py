import copy
import io
import logging
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from pytorch3d.renderer import PerspectiveCameras
from scene.cameras import Camera
from scipy.spatial import cKDTree
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import gaussian_blur

from .general_utils import save_video


def convert_pt3d_cam_to_3dgs_cam(pt3d_cam: PerspectiveCameras, xyz_scale=1):
    transform_matrix_pt3d = pt3d_cam.get_world_to_view_transform().get_matrix()[0]
    transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
    transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale
    transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()
    opengl_to_pt3d = torch.diag(
        torch.tensor([-1.0, 1, -1, 1], device=torch.device("cuda"))
    )
    transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d
    transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
    c2w = np.array(transform_matrix)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    focal_length = pt3d_cam.K[0, 0, 0].item()
    half_img_size_x = pt3d_cam.K[0, 0, 2].item()
    fovx = 2 * np.arctan(half_img_size_x / focal_length)
    half_img_size_y = pt3d_cam.K[0, 1, 2].item()
    fovy = 2 * np.arctan(half_img_size_y / focal_length)
    tdgs_cam = Camera(R=R, T=T, FoVx=fovx, FoVy=fovy)
    return tdgs_cam


def rotate_pytorch3d_camera(camera: PerspectiveCameras, angle_rad: float, axis="x"):
    """
    Rotate a PyTorch3D camera object around the specified axis by the given angle.
    It should keep its own location in the world frame.
    This means that the following equation should hold:
    x_world @ P_w2c^new = x_world @ P_w2c^old @ P^(-1),
    where P^(-1) denotes the inverse of the desired transform matrix.

    Parameters:
        camera (PyTorch3D Camera): The camera object to rotate.
        angle_rad (float): The angle in radians by which to rotate the camera.
        axis (str): The axis around which to rotate the camera. Can be 'x', 'y', or 'z'.

    Returns:
        PyTorch3D Camera: The rotated camera object.
    """
    if axis == "x":
        R = torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(angle_rad), -torch.sin(angle_rad)],
                [0, torch.sin(angle_rad), torch.cos(angle_rad)],
            ]
        ).float()
    elif axis == "y":
        R = torch.tensor(
            [
                [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
                [0, 1, 0],
                [-torch.sin(angle_rad), 0, torch.cos(angle_rad)],
            ]
        ).float()
    elif axis == "z":
        R = torch.tensor(
            [
                [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
                [torch.sin(angle_rad), torch.cos(angle_rad), 0],
                [0, 0, 1],
            ]
        ).float()
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    # Construct the pytorch3d-style P matrix from R and T. P=[[R', 0], [T, 1]]
    P = torch.eye(4)
    P[:3, :3] = R.transpose(0, 1)
    Pinv = torch.inverse(P).to(camera.device)

    P_old = camera.get_world_to_view_transform().get_matrix()
    P_new = P_old @ Pinv
    T_new = P_new[:, 3, :3]
    R_new = P_new[:, :3, :3]

    new_camera = camera.clone()
    new_camera.T = T_new
    new_camera.R = R_new

    return new_camera


def translate_pytorch3d_camera(camera: PerspectiveCameras, translation: torch.Tensor):
    """
    Translate a PyTorch3D camera object by the given translation vector.
    It should keep its own orientation in the world frame.
    This means that the following equation should hold:
    x_world @ P_w2c^new = x_world @ P_w2c^old @ P^(-1),
    where P^(-1) denotes the inverse of the desired transform matrix.

    Parameters:
        camera (PyTorch3D Camera): The camera object to translate.
        translation (torch.Tensor): The translation vector to apply to the camera.

    Returns:
        PyTorch3D Camera: The translated camera object.
    """
    # Construct the pytorch3d-style P matrix from R and T. P=[[R', 0], [T, 1]]
    P = torch.eye(4)
    P[3, :3] = translation
    Pinv = torch.inverse(P).to(camera.device)

    P_old = camera.get_world_to_view_transform().get_matrix()
    P_new = P_old @ Pinv
    T_new = P_new[:, 3, :3]
    R_new = P_new[:, :3, :3]

    new_camera = camera.clone()
    new_camera.T = T_new
    new_camera.R = R_new

    return new_camera


def find_biggest_connected_inpaint_region(mask):
    H, W = mask.shape
    visited = torch.zeros((H, W), dtype=torch.bool)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # up, right, down, left

    def bfs(i, j):
        queue = deque([(i, j)])
        region = []

        while queue:
            x, y = queue.popleft()
            if 0 <= x < H and 0 <= y < W and not visited[x, y] and mask[x, y] == 1:
                visited[x, y] = True
                region.append((x, y))
                for dx, dy in directions:
                    queue.append((x + dx, y + dy))

        return region

    max_region = []

    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1 and not visited[i, j]:
                current_region = bfs(i, j)
                if len(current_region) > len(max_region):
                    max_region = current_region

    mask_connected = torch.zeros((H, W)).to(mask.device)
    for x, y in max_region:
        mask_connected[x, y] = 1
    return mask_connected


def edge_pad(img, mask, mode=1):
    if mode == 0:
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res0 = 1 - nmask
        res1 = nmask
        p0 = np.stack(res0.nonzero(), axis=0).transpose()
        p1 = np.stack(res1.nonzero(), axis=0).transpose()
        min_dists, min_dist_idx = cKDTree(p1).query(p0, 1)
        loc = p1[min_dist_idx]
        for (a, b), (c, d) in zip(p0, loc, strict=False):
            img[a, b] = img[c, d]
    elif mode == 1:
        record = {}
        kernel = [[1] * 3 for _ in range(3)]
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res = scipy.signal.convolve2d(
            nmask, kernel, mode="same", boundary="fill", fillvalue=1
        )
        res[nmask < 1] = 0
        res[res == 9] = 0
        res[res > 0] = 1
        ylst, xlst = res.nonzero()
        queue = [(y, x) for y, x in zip(ylst, xlst, strict=False)]
        # bfs here
        cnt = res.astype(np.float32)
        acc = img.astype(np.float32)
        step = 1
        h = acc.shape[0]
        w = acc.shape[1]
        offset = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while queue:
            target = []
            for y, x in queue:
                val = acc[y][x]
                for yo, xo in offset:
                    yn = y + yo
                    xn = x + xo
                    if 0 <= yn < h and 0 <= xn < w and nmask[yn][xn] < 1:
                        if record.get((yn, xn), step) == step:
                            acc[yn][xn] = acc[yn][xn] * cnt[yn][xn] + val
                            cnt[yn][xn] += 1
                            acc[yn][xn] /= cnt[yn][xn]
                            if (yn, xn) not in record:
                                record[(yn, xn)] = step
                                target.append((yn, xn))
            step += 1
            queue = target
        img = acc.astype(np.uint8)
    else:
        nmask = mask.copy()
        ylst, xlst = nmask.nonzero()
        yt, xt = ylst.min(), xlst.min()
        yb, xb = ylst.max(), xlst.max()
        content = img[yt : yb + 1, xt : xb + 1]
        img = np.pad(
            content,
            ((yt, mask.shape[0] - yb - 1), (xt, mask.shape[1] - xb - 1), (0, 0)),
            mode="edge",
        )
    return img, mask


def gaussian_noise(img, mask):
    noise = np.random.randn(mask.shape[0], mask.shape[1], 3)
    noise = (noise + 1) / 2 * 255
    noise = noise.astype(np.uint8)
    nmask = mask.copy()
    nmask[mask > 0] = 1
    img = nmask[:, :, np.newaxis] * img + (1 - nmask[:, :, np.newaxis]) * noise
    return img, mask


def cv2_telea(img, mask, radius=5):
    ret = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    return ret, mask


def cv2_ns(img, mask, radius=5):
    ret = cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)
    return ret, mask


def mean_fill(img, mask):
    avg = img.mean(axis=0).mean(axis=0)
    img[mask < 1] = avg
    return img, mask


def estimate_scale_and_shift(x, y, init_method="identity", optimize_scale=True):
    assert len(x.shape) == 1 and len(y.shape) == 1, "Inputs should be 1D tensors"
    assert x.shape[0] == y.shape[0], "Input tensors should have the same length"

    n = x.shape[0]

    if init_method == "identity":
        shift_init = 0.0
        scale_init = 1.0
    elif init_method == "median":
        shift_init = (torch.median(y) - torch.median(x)).item()
        scale_init = (
            torch.sum(torch.abs(y - torch.median(y)))
            / n
            / (torch.sum(torch.abs(x - torch.median(x))) / n)
        ).item()
    else:
        raise ValueError("init_method should be either 'identity' or 'median'")
    shift = torch.tensor(shift_init).cuda().requires_grad_()
    scale = torch.tensor(scale_init).cuda().requires_grad_()

    # Set optimizer and scheduler
    optimizer = torch.optim.Adam([shift, scale], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, verbose=True
    )

    # Optimization loop
    for step in range(
        1000
    ):  # Set the range to the number of steps you find appropriate
        optimizer.zero_grad()
        if optimize_scale:
            loss = torch.abs((x.detach() + shift) * scale - y.detach()).mean()
        else:
            loss = torch.abs(x.detach() + shift - y.detach()).mean()
        loss.backward()
        if step == 0:
            print(f"Iteration {step + 1}: L1 Loss = {loss.item():.4f}")
        optimizer.step()
        scheduler.step(loss)

        # Early stopping condition if needed
        if (
            step > 20 and scheduler._last_lr[0] < 1e-6
        ):  # You might want to adjust these conditions
            print(f"Iteration {step + 1}: L1 Loss = {loss.item():.4f}")
            break

    if optimize_scale:
        return scale.item(), shift.item()
    else:
        return 1.0, shift.item()


def save_depth_map(depth_map, file_name, vmin=None, vmax=None, save_clean=False):
    depth_map = np.squeeze(depth_map)
    if depth_map.ndim != 2:
        raise ValueError("Depth map after squeezing must be 2D.")

    dpi = 100  # Adjust this value if necessary
    figsize = (
        depth_map.shape[1] / dpi,
        depth_map.shape[0] / dpi,
    )  # Width, Height in inches

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cax = ax.imshow(depth_map, cmap="viridis", vmin=vmin, vmax=vmax)

    if not save_clean:
        # Standard save with labels and color bar
        cbar = fig.colorbar(cax)
        ax.set_title("Depth Map")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
    else:
        # Clean save without labels, color bar, or axis
        plt.axis("off")
        ax.set_aspect("equal", adjustable="box")

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    img = img.convert("RGB")  # Convert to RGB
    img = img.resize(
        (depth_map.shape[1], depth_map.shape[0]), Image.Resampling.LANCZOS
    )  # Resize to original dimensions
    img.save(file_name, format="png")
    buf.close()
    plt.close()
    return img


"""
Apache-2.0 license
https://github.com/hafriedlander/stable-diffusion-grpcserver/blob/main/sdgrpcserver/services/generate.py
https://github.com/parlance-zz/g-diffuser-bot/tree/g-diffuser-bot-beta2
_handleImageAdjustment
"""

functbl = {
    "gaussian": gaussian_noise,
    "edge_pad": edge_pad,
    "cv2_ns": cv2_ns,
    "cv2_telea": cv2_telea,
}


def soft_stitching(source_img, target_img, mask, blur_size=11, sigma=2.5):
    # Apply Gaussian blur to the mask to create a soft transition area
    # The size of the kernel and the standard deviation can be adjusted
    # for more or less blending

    # blur_size  # Size of the Gaussian kernel, must be odd
    # sigma       # Standard deviation of the Gaussian kernel

    # Ensure the mask is float for blurring
    soft_mask = mask.float()

    # Adding padding to reduce edge effects during blurring
    padding = blur_size // 2
    soft_mask = F.pad(soft_mask, (padding, padding, padding, padding), mode="reflect")

    # Apply the Gaussian blur
    blurred_mask = gaussian_blur(
        soft_mask, kernel_size=(blur_size, blur_size), sigma=(sigma, sigma)
    )

    # Remove the padding
    blurred_mask = blurred_mask[:, :, padding:-padding, padding:-padding]

    # Ensure the mask is within 0 and 1 after blurring
    blurred_mask = torch.clamp(blurred_mask, 0, 1)

    # Blend the images based on the blurred mask
    stitched_img = source_img * blurred_mask + target_img * (1 - blurred_mask)

    return stitched_img


def prepare_scheduler(scheduler):
    # if hasattr(scheduler.config, "steps_offset"):
    #     new_config = dict(scheduler.config)
    #     new_config["steps_offset"] = 0
    #     scheduler._internal_dict = FrozenDict(new_config)
    if hasattr(scheduler, "is_scale_input_called"):
        scheduler.is_scale_input_called = True  # to surpress the warning
    return scheduler


def load_example_yaml(example_name, yaml_path):
    with open(yaml_path) as file:
        data = yaml.safe_load(file)
    yaml_data = None

    for d in data:
        if "name" in d and d["name"] == example_name:
            yaml_data = d
            break
    return yaml_data


def quaternion_to_matrix_torch(
    quaternion: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices (torch version)

    Args:
        quaternion (torch.Tensor): shape (..., 4), the quaternions to convert
        eps (float): small value to avoid division by zero

    Returns:
        torch.Tensor: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions
    """
    assert quaternion.shape[-1] == 4
    quaternion = quaternion / torch.clamp(
        torch.norm(quaternion, dim=-1, keepdim=True), min=eps
    )
    w, x, y, z = (
        quaternion[..., 0],
        quaternion[..., 1],
        quaternion[..., 2],
        quaternion[..., 3],
    )
    zeros = torch.zeros_like(w)
    I = torch.eye(3, dtype=quaternion.dtype, device=quaternion.device)
    xyz = quaternion[..., 1:]
    A = (
        xyz[..., :, None] * xyz[..., None, :]
        - I * (xyz**2).sum(dim=-1)[..., None, None]
    )
    B = torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1).reshape(
        *quaternion.shape[:-1], 3, 3
    )
    rot_mat = I + 2 * (A + w[..., None, None] * B)
    return rot_mat


def matrix_to_quaternion_torch(
    rot_mat: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z) (torch version)

    Args:
        rot_mat (torch.Tensor): shape (..., 3, 3), the rotation matrices to convert
        eps (float): small value to avoid division by zero

    Returns:
        torch.Tensor: shape (..., 4), the quaternions corresponding to the given rotation matrices
    """
    # Extract the diagonal and off-diagonal elements of the rotation matrix
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = [
        rot_mat[..., i, j] for i in range(3) for j in range(3)
    ]

    diag = torch.diagonal(rot_mat, dim1=-2, dim2=-1)
    M = torch.tensor(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        dtype=rot_mat.dtype,
        device=rot_mat.device,
    )
    wxyz = 0.5 * torch.clamp(1 + diag @ M.T, min=0.0) ** 0.5
    max_idx = torch.argmax(wxyz, dim=-1)
    xw = torch.sign(m21 - m12)
    yw = torch.sign(m02 - m20)
    zw = torch.sign(m10 - m01)
    yz = torch.sign(m21 + m12)
    xz = torch.sign(m02 + m20)
    xy = torch.sign(m01 + m10)
    ones = torch.ones_like(xw)
    sign = torch.where(
        max_idx[..., None] == 0,
        torch.stack([ones, xw, yw, zw], dim=-1),
        torch.where(
            max_idx[..., None] == 1,
            torch.stack([xw, ones, xy, xz], dim=-1),
            torch.where(
                max_idx[..., None] == 2,
                torch.stack([yw, xy, ones, yz], dim=-1),
                torch.stack([zw, xz, yz, ones], dim=-1),
            ),
        ),
    )
    quat = sign * wxyz
    quat = quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=eps)
    return quat


def merge_frames(
    all_rundir, fps=10, save_dir=None, is_forward=False, save_depth=False, save_gif=True
):
    """
    Merge frames from multiple run directories into a single directory with continuous naming.

    Parameters:
        all_rundir (list of pathlib.Path): Directories containing the run data.
        save_dir (pathlib.Path): Directory where all frames should be saved.
    """

    # Ensure save_dir/frames exists
    save_frames_dir = save_dir / "frames"
    save_frames_dir.mkdir(parents=True, exist_ok=True)

    if save_depth:
        save_depth_dir = save_dir / "depth"
        save_depth_dir.mkdir(parents=True, exist_ok=True)

    # Initialize a counter for the new filenames
    global_counter = 0

    # Iterate through all provided run directories
    if is_forward:
        all_rundir = all_rundir[::-1]
    for rundir in all_rundir:
        # Ensure the rundir and the frames subdir exist
        if not rundir.exists():
            print(f"Warning: {rundir} does not exist. Skipping...")
            continue

        frames_dir = rundir / "images" / "frames"
        if not frames_dir.exists():
            print(f"Warning: {frames_dir} does not exist. Skipping...")
            continue

        if save_depth:
            depth_dir = rundir / "images" / "depth"
            if not depth_dir.exists():
                print(f"Warning: {depth_dir} does not exist. Skipping...")
                continue

        # Get all .png files in the frames directory, assuming no nested dirs
        frame_files = sorted(frames_dir.glob("*.png"), key=lambda x: int(x.stem))
        if save_depth:
            depth_files = sorted(depth_dir.glob("*.png"), key=lambda x: int(x.stem))

        # Copy and rename each file
        for i, frame_file in enumerate(frame_files):
            # Form the new path and copy the file
            new_frame_path = save_frames_dir / f"{global_counter}.png"
            shutil.copy(str(frame_file), str(new_frame_path))

            if save_depth:
                # Form the new path and copy the file
                new_depth_path = save_depth_dir / f"{global_counter}.png"
                shutil.copy(str(depth_files[i]), str(new_depth_path))

            # Increment the global counter
            global_counter += 1

    last_keyframe_name = "kf1.png" if is_forward else "kf2.png"
    last_keyframe = all_rundir[-1] / "images" / last_keyframe_name
    new_frame_path = save_frames_dir / f"{global_counter}.png"
    shutil.copy(str(last_keyframe), str(new_frame_path))

    if save_depth:
        last_depth_name = "kf1_depth.png" if is_forward else "kf2_depth.png"
        last_depth = all_rundir[-1] / "images" / last_depth_name
        new_depth_path = save_depth_dir / f"{global_counter}.png"
        shutil.copy(str(last_depth), str(new_depth_path))

    frames = []
    for frame_file in sorted(save_frames_dir.glob("*.png"), key=lambda x: int(x.stem)):
        frame_image = Image.open(frame_file)
        frame = ToTensor()(frame_image).unsqueeze(0)
        frames.append(frame)

    if save_depth:
        depth = []
        for depth_file in sorted(
            save_depth_dir.glob("*.png"), key=lambda x: int(x.stem)
        ):
            depth_image = Image.open(depth_file)
            depth_frame = ToTensor()(depth_image).unsqueeze(0)
            depth.append(depth_frame)

    video = (255 * torch.cat(frames, dim=0)).to(torch.uint8).detach().cpu()
    video_reverse = (
        (255 * torch.cat(frames[::-1], dim=0)).to(torch.uint8).detach().cpu()
    )

    save_video(video, save_dir / "output.mp4", fps=fps, save_gif=save_gif)
    save_video(
        video_reverse, save_dir / "output_reverse.mp4", fps=fps, save_gif=save_gif
    )

    if save_depth:
        depth_video = (255 * torch.cat(depth, dim=0)).to(torch.uint8).detach().cpu()
        depth_video_reverse = (
            (255 * torch.cat(depth[::-1], dim=0)).to(torch.uint8).detach().cpu()
        )

        save_video(
            depth_video, save_dir / "output_depth.mp4", fps=fps, save_gif=save_gif
        )
        save_video(
            depth_video_reverse,
            save_dir / "output_depth_reverse.mp4",
            fps=fps,
            save_gif=save_gif,
        )


def merge_keyframes(all_keyframes, save_dir, save_folder="keyframes", fps=1):
    """
    Save a list of PIL images sequentially into a directory.

    Parameters:
        all_keyframes (list): A list of PIL Image objects.
        save_dir (Path): A pathlib Path object indicating where to save the images.
    """
    # Ensure that the save_dir exists
    save_path = save_dir / save_folder
    save_path.mkdir(parents=True, exist_ok=True)

    # Save each keyframe with a sequential filename
    for i, frame in enumerate(all_keyframes):
        frame.save(save_path / f"{i}.png")

    all_keyframes = [ToTensor()(frame).unsqueeze(0) for frame in all_keyframes]
    all_keyframes = torch.cat(all_keyframes, dim=0)
    video = (255 * all_keyframes).to(torch.uint8).detach().cpu()
    video_reverse = (255 * all_keyframes.flip(0)).to(torch.uint8).detach().cpu()

    save_video(video, save_dir / "keyframes.mp4", fps=fps)
    save_video(video_reverse, save_dir / "keyframes_reverse.mp4", fps=fps)


class SimpleLogger:
    def __init__(self, log_path):
        # Ensure log_path is a Path object, whether provided as str or Path
        if not isinstance(log_path, Path):
            log_path = Path(log_path)

        # Ensure the file ends with '.log'
        if not log_path.name.endswith(".txt"):
            raise ValueError("Log file must end with '.txt' extension")

        # Create the directory if it does not exist
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(str(log_path))
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def print(self, message, attach_time=False):
        if attach_time:
            current_time = datetime.now().strftime("[%H:%M:%S]")
            self.logger.info(current_time)
        self.logger.info(message)


def interp_poses(start_cam, end_cam, frames=100):
    cams = []
    cams.append(start_cam)

    for i in range(1, frames - 1):
        interp_w = i / (frames - 1)
        cur_R = (1 - interp_w) * start_cam.R + interp_w * end_cam.R
        cur_T = (1 - interp_w) * start_cam.T + interp_w * end_cam.T
        cur_cam = Camera(R=cur_R, T=cur_T, FoVx=start_cam.FoVx, FoVy=start_cam.FoVy)

        cams.append(cur_cam)
        del cur_cam

    cams.append(end_cam)

    return cams


def filter_mask(mask, mask2, min_area=200):
    """
    参数:
    mask [H,W]: 标签掩码，取值为1-N（torch.Tensor）
    mask2 [H,W]: 二值掩码，取值为0或1（torch.Tensor）
    threshold: 重叠阈值，默认为0.9 (90%)

    返回:
    filtered_mask [H,W]: 过滤后的掩码（torch.Tensor）
    """
    mask = mask.clone()
    mask[mask2.squeeze() == 0] = 0
    # Map to 0-K
    new_mask = mask.clone()
    i = 0
    for label in torch.unique(mask):
        if label != 0 and (mask == label).sum() < min_area:
            continue
        new_mask[mask == label] = i
        i += 1

    return new_mask


def filter_mask_by_overlap_torch(
    mask, mask2, threshold=0.4, min_area=200, mask_info=None
):
    """
    保留mask中90%以上位于mask2中的id，其余id置0（PyTorch实现）

    参数:
    mask [H,W]: 标签掩码，取值为1-N（torch.Tensor）
    mask2 [H,W]: 二值掩码，取值为0或1（torch.Tensor）
    threshold: 重叠阈值，默认为0.9 (90%)

    返回:
    filtered_mask [H,W]: 过滤后的掩码（torch.Tensor）
    """
    # Ensure input is torch.Tensor
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)
    if not isinstance(mask2, torch.Tensor):
        mask2 = torch.tensor(mask2)

    mask2 = mask2.clone().squeeze()
    # Copy original mask
    filtered_mask = mask.clone()

    # Get unique IDs (excluding background 0)
    unique_ids = torch.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]

    if len(unique_ids) == 0:
        return filtered_mask, mask_info

    # Convert mask2 to boolean mask
    mask2_bool = mask2.bool()

    # Calculate total number of pixels for each ID
    # Use one-hot encoding and sum to calculate pixel count for each ID
    mask_flat = mask.flatten()
    max_id = int(unique_ids.max().item())

    # Use torch.bincount to calculate total pixel count for each ID
    total_pixels = torch.bincount(mask_flat, minlength=max_id + 1)[
        1:
    ]  # Exclude background 0

    # Calculate pixel count for each ID in mask2
    overlap_mask = mask * mask2_bool.to(mask.dtype)
    overlap_pixels = torch.bincount(overlap_mask.flatten(), minlength=max_id + 1)[
        1:
    ]  # 排除背景0

    # 计算重叠比例
    overlap_ratio = overlap_pixels.float() / total_pixels.float()

    # 确定哪些ID需要保留（重叠比例 >= 阈值）
    ids_to_keep = overlap_ratio >= threshold

    # 创建一个映射数组，将不需要保留的ID映射为0
    id_mapping = torch.zeros(max_id + 1, dtype=mask.dtype, device=mask.device)
    label_keep = torch.arange(1, max_id + 1, device=mask.device, dtype=mask.dtype)[
        ids_to_keep
    ]
    id_mapping[label_keep] = label_keep

    # 应用映射到原始掩码
    filtered_mask = id_mapping[mask]

    filtered_mask[mask2 == 0] = 0

    # Map to 0-K
    updated_mask_info = {}

    new_mask = torch.zeros_like(filtered_mask)  # filtered_mask.clone()
    i = 0
    unique_labels = torch.unique(filtered_mask)

    for label in unique_labels:
        label_item = label.item()
        if label_item == 0:
            i = 0
            continue
        if (filtered_mask == label_item).sum() < min_area:
            continue
        i += 1
        new_mask[filtered_mask == label_item] = i
        updated_mask_info[i] = mask_info[label_item]

    return new_mask, updated_mask_info


PALETTE = [
    0,
    0,
    0,
    128,
    0,
    0,
    0,
    128,
    0,
    128,
    128,
    0,
    0,
    0,
    128,
    128,
    0,
    128,
    0,
    128,
    128,
    128,
    128,
    128,
    64,
    0,
    0,
    191,
    0,
    0,
]
_palette = (
    ((np.random.random((3 * (255 - len(PALETTE)))) * 0.7 + 0.3) * 255)
    .astype(np.uint8)
    .tolist()
)
PALETTE += _palette


def visualize_seg(img, with_id=True, segments_info=None):
    # 检查输入类型并进行适当转换
    if hasattr(img, "cpu") and callable(img.cpu):  # 检查是否为tensor
        img_array = img.cpu().numpy().astype(np.uint8)
    else:  # 假设已经是numpy数组
        img_array = img.astype(np.uint8)

    # 创建PIL图像并设置调色板
    img = Image.fromarray(img_array, mode="P")
    img.putpalette(PALETTE)
    if not with_id:
        return img

    # 转换为RGB模式以便绘制文本
    img_rgb = img.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    # 获取所有唯一的mask_id
    unique_ids = np.unique(img_array)
    unique_ids = unique_ids[unique_ids > 0]  # 排除背景0

    for mask_id in unique_ids:
        # 找到mask区域的中心点
        mask_region = img_array == mask_id
        y_indices, x_indices = np.where(mask_region)
        if len(y_indices) > 0 and len(x_indices) > 0:
            center_x = int(np.mean(x_indices))
            center_y = int(np.mean(y_indices))

            # 构建要显示的文本
            text = f"ID:{mask_id}"
            if segments_info is not None and mask_id in segments_info:
                text += f"\n{segments_info[mask_id]}"

            # 绘制文本
            draw.text((center_x, center_y), text, fill="white")

    return img_rgb


def generate_distant_vectors_(N, K, thresh):
    vectors = torch.randn((N, K))
    codes = F.normalize(vectors, dim=-1)
    cos_sim = codes @ codes.T
    cos_sim.fill_diagonal_(0)
    selected = []
    remaining = set(range(N))
    while remaining:
        i = remaining.pop()
        selected.append(i)
        # 找到所有与 i 相似度大于阈值的点，去掉它们

        to_remove = set(torch.where(cos_sim[i] > thresh)[0].tolist())
        # print(to_remove)
        remaining -= to_remove
    codes = codes[selected]
    return codes


def generate_distant_vectors(N, K, thresh, max_try=1000):
    codes = generate_distant_vectors_(10000, K, thresh)  # [N1,K]
    count = 0
    while codes.shape[0] < N and count < max_try:
        new_codes = generate_distant_vectors_(10000, K, thresh)  # [N2,K]
        cos_sim = (codes @ new_codes.T).max(0).values  # [N2]
        new_codes = new_codes[cos_sim < thresh]
        codes = torch.cat([codes, new_codes], dim=0)
        count += 1
    return codes[:N]


def merge_mask(mask, prev_mask, min_overlap_area=100, segments_info=None, debug=False):
    """
    mask [H,W]
    mask2 [H, W]
    """
    mask = mask.clone().squeeze()
    prev_mask = prev_mask.clone().squeeze().to(mask.device)
    # Get unique labels in each mask
    unique_labels = torch.unique(mask)
    prev_unique_labels = torch.unique(prev_mask)

    # Create a new mask to store the results
    merged_mask = mask.clone().squeeze()

    new_info = copy.deepcopy(segments_info)

    # Iterate over each label in the current mask
    for label in unique_labels:
        if label == 0:
            continue  # Skip background

        # Create a binary mask for the current label
        current_label_mask = mask == label

        # Calculate overlap with each label in the previous mask
        max_overlap = 0
        best_prev_label = None

        for prev_label in prev_unique_labels:
            if prev_label == 0:
                continue  # Skip background

            # Create a binary mask for the previous label
            prev_label_mask = prev_mask == prev_label

            # Calculate the overlap area
            overlap_area = (current_label_mask & prev_label_mask).sum().item()

            # Update max overlap if current overlap is larger
            if overlap_area > max_overlap:
                max_overlap = overlap_area
                best_prev_label = prev_label

        # If the maximum overlap area is greater than the minimum, update the label
        if max_overlap > min_overlap_area and best_prev_label is not None:
            merged_mask[current_label_mask] = best_prev_label.item()
            try:
                del new_info[label.item()]
            except:
                pass

    return merged_mask, new_info


def determine_is_fg(mask_i, id, prev_mask, min_overlap_area, segments_info):
    unique_label = np.unique(prev_mask)

    for label in unique_label:
        if label == 0 or (label not in segments_info.keys()):
            continue
        # if id != segments_info[label]:
        #     continue
        overlap = (mask_i * (prev_mask == label)).sum()
        if overlap > min_overlap_area and overlap / (prev_mask == label).sum() > 0.8:
            return True
    return False


def get_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    """
    输入：
        angle: (3,) 的Tensor, 分别是绕 x, y, z 轴旋转的角度（单位：弧度）
    输出：
        3x3 的旋转矩阵（torch.Tensor），可以反向传播
    """
    rx, ry, rz = angle[0], angle[1], angle[2]

    cos_rx = torch.cos(rx)
    sin_rx = torch.sin(rx)
    cos_ry = torch.cos(ry)
    sin_ry = torch.sin(ry)
    cos_rz = torch.cos(rz)
    sin_rz = torch.sin(rz)

    # 绕 x 轴的旋转矩阵
    Rx = torch.stack(
        [
            torch.stack(
                [torch.ones_like(rx), torch.zeros_like(rx), torch.zeros_like(rx)]
            ),
            torch.stack([torch.zeros_like(rx), cos_rx, -sin_rx]),
            torch.stack([torch.zeros_like(rx), sin_rx, cos_rx]),
        ]
    )

    # 绕 y 轴的旋转矩阵
    Ry = torch.stack(
        [
            torch.stack([cos_ry, torch.zeros_like(ry), sin_ry]),
            torch.stack(
                [torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry)]
            ),
            torch.stack([-sin_ry, torch.zeros_like(ry), cos_ry]),
        ]
    )

    # 绕 z 轴的旋转矩阵
    Rz = torch.stack(
        [
            torch.stack([cos_rz, -sin_rz, torch.zeros_like(rz)]),
            torch.stack([sin_rz, cos_rz, torch.zeros_like(rz)]),
            torch.stack(
                [torch.zeros_like(rz), torch.zeros_like(rz), torch.ones_like(rz)]
            ),
        ]
    )

    # 最终旋转矩阵：R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx
    return R


def get_rotation_matrix_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Apply Gram-Schmidt orthogonalization to a 2x3 matrix to get a 3x3 rotation matrix.

    Args:
        matrix: [2, 3] tensor representing first two rows of rotation matrix

    Returns:
        R: [3, 3] orthogonal rotation matrix
    """
    # 提取两行
    v1 = matrix[0]  # [3]
    v2 = matrix[1]  # [3]

    # Gram-Schmidt 正交化
    u1 = F.normalize(v1, dim=0)  # 单位化第一个向量

    # 去掉 v2 在 u1 方向的分量
    proj = torch.dot(v2, u1) * u1
    u2 = F.normalize(v2 - proj, dim=0)

    # 第三个向量 = 叉乘
    u3 = torch.cross(u1, u2)
    u3 = F.normalize(u3, dim=0)

    # 构造旋转矩阵 (3x3)
    R = torch.stack([u1, u2, u3], dim=0)
    R[:2, :] *= -1
    return R


ade_fg_classes = [
    "bed",
    "cabinet",
    "person",
    "chair",
    "car",
    "painting",
    "sofa",
    "shelf",
    "mirror",
    "armchair",
    "seat",
    "desk",
    "wardrobe",
    "lamp",
    "bathtub",
    "box",
    "signboard",
    "chest of drawers",
    "counter",
    "sink",
    "fireplace",
    "refrigerator",
    "case",
    "pool table",
    "pillow",
    "coffee table",
    "toilet",
    "book",
    "bench",
    "countertop",
    "stove",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "bus",
    "towel",
    "truck",
    "television receiver",
    "airplane",
    "apparel",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "van",
    "ship",
    "conveyer belt",
    "washer",
    "plaything",
    "stool",
    "barrel",
    "basket",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "tank",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "vase",
    "tray",
    "ashcan",
    "fan",
    "crt screen",
    "plate",
    "monitor",
    "shower",
    "glass",
    "clock",
    "flag",
]


ade_id2label = {
    "0": "wall",
    "1": "building",
    "2": "sky",
    "3": "floor",
    "4": "tree",
    "5": "ceiling",
    "6": "road",
    "7": "bed ",
    "8": "windowpane",
    "9": "grass",
    "10": "cabinet",
    "11": "sidewalk",
    "12": "person",
    "13": "earth",
    "14": "door",
    "15": "table",
    "16": "mountain",
    "17": "plant",
    "18": "curtain",
    "19": "chair",
    "20": "car",
    "21": "water",
    "22": "painting",
    "23": "sofa",
    "24": "shelf",
    "25": "house",
    "26": "sea",
    "27": "mirror",
    "28": "rug",
    "29": "field",
    "30": "armchair",
    "31": "seat",
    "32": "fence",
    "33": "desk",
    "34": "rock",
    "35": "wardrobe",
    "36": "lamp",
    "37": "bathtub",
    "38": "railing",
    "39": "cushion",
    "40": "base",
    "41": "box",
    "42": "column",
    "43": "signboard",
    "44": "chest of drawers",
    "45": "counter",
    "46": "sand",
    "47": "sink",
    "48": "skyscraper",
    "49": "fireplace",
    "50": "refrigerator",
    "51": "grandstand",
    "52": "path",
    "53": "stairs",
    "54": "runway",
    "55": "case",
    "56": "pool table",
    "57": "pillow",
    "58": "screen door",
    "59": "stairway",
    "60": "river",
    "61": "bridge",
    "62": "bookcase",
    "63": "blind",
    "64": "coffee table",
    "65": "toilet",
    "66": "flower",
    "67": "book",
    "68": "hill",
    "69": "bench",
    "70": "countertop",
    "71": "stove",
    "72": "palm",
    "73": "kitchen island",
    "74": "computer",
    "75": "swivel chair",
    "76": "boat",
    "77": "bar",
    "78": "arcade machine",
    "79": "hovel",
    "80": "bus",
    "81": "towel",
    "82": "light",
    "83": "truck",
    "84": "tower",
    "85": "chandelier",
    "86": "awning",
    "87": "streetlight",
    "88": "booth",
    "89": "television receiver",
    "90": "airplane",
    "91": "dirt track",
    "92": "apparel",
    "93": "pole",
    "94": "land",
    "95": "bannister",
    "96": "escalator",
    "97": "ottoman",
    "98": "bottle",
    "99": "buffet",
    "100": "poster",
    "101": "stage",
    "102": "van",
    "103": "ship",
    "104": "fountain",
    "105": "conveyer belt",
    "106": "canopy",
    "107": "washer",
    "108": "plaything",
    "109": "swimming pool",
    "110": "stool",
    "111": "barrel",
    "112": "basket",
    "113": "waterfall",
    "114": "tent",
    "115": "bag",
    "116": "minibike",
    "117": "cradle",
    "118": "oven",
    "119": "ball",
    "120": "food",
    "121": "step",
    "122": "tank",
    "123": "trade name",
    "124": "microwave",
    "125": "pot",
    "126": "animal",
    "127": "bicycle",
    "128": "lake",
    "129": "dishwasher",
    "130": "screen",
    "131": "blanket",
    "132": "sculpture",
    "133": "hood",
    "134": "sconce",
    "135": "vase",
    "136": "traffic light",
    "137": "tray",
    "138": "ashcan",
    "139": "fan",
    "140": "pier",
    "141": "crt screen",
    "142": "plate",
    "143": "monitor",
    "144": "bulletin board",
    "145": "shower",
    "146": "radiator",
    "147": "glass",
    "148": "clock",
    "149": "flag",
}
