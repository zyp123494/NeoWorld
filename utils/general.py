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
import random
import sys
from datetime import datetime

import numpy as np
import torch


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def quaternion2rotmat(q):
    # check if q is normalized
    assert torch.allclose(
        torch.norm(q, dim=-1), torch.ones(q.size(0), device=q.device)
    ), "quaternion is not normalized"

    R = torch.zeros((q.size(0), 3, 3), device=q.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotmat2quaternion(R):
    assert R.size(1) == 3 and R.size(2) == 3, "R must be of shape [B, 3, 3]"

    B = R.size(0)
    q = torch.zeros((B, 4), device=R.device, dtype=R.dtype)

    r11, r12, r13 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    r21, r22, r23 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    r31, r32, r33 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    # Calculate trace
    trace = r11 + r22 + r33

    # Case where the trace is positive
    s = torch.sqrt(trace + 1.0) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (r32 - r23) / s
    q[:, 2] = (r13 - r31) / s
    q[:, 3] = (r21 - r12) / s

    # Cases where the trace is negative
    t1 = (r11 > r22) & (r11 > r33)  # case for x dominant
    t2 = (r22 > r11) & (r22 > r33)  # case for y dominant
    t3 = (r33 > r11) & (r33 > r22)  # case for z dominant

    # Recalculate s for different cases
    s1 = torch.sqrt(1.0 + r11 - r22 - r33) * 2
    s2 = torch.sqrt(1.0 + r22 - r11 - r33) * 2
    s3 = torch.sqrt(1.0 + r33 - r11 - r22) * 2

    q[t1, 0] = (r32[t1] - r23[t1]) / s1[t1]
    q[t1, 1] = 0.25 * s1[t1]
    q[t1, 2] = (r12[t1] + r21[t1]) / s1[t1]
    q[t1, 3] = (r13[t1] + r31[t1]) / s1[t1]

    q[t2, 0] = (r13[t2] - r31[t2]) / s2[t2]
    q[t2, 1] = (r12[t2] + r21[t2]) / s2[t2]
    q[t2, 2] = 0.25 * s2[t2]
    q[t2, 3] = (r23[t2] + r32[t2]) / s2[t2]

    q[t3, 0] = (r21[t3] - r12[t3]) / s3[t3]
    q[t3, 1] = (r13[t3] + r31[t3]) / s3[t3]
    q[t3, 2] = (r23[t3] + r32[t3]) / s3[t3]
    q[t3, 3] = 0.25 * s3[t3]

    return q


def normal2rotation(n):
    n = torch.nn.functional.normalize(n, dim=1)  # Normalize the input normal vector
    proxy_x = torch.tensor([1, 0, 0], dtype=torch.float32, device=n.device).expand_as(n)
    proxy_y = torch.tensor([0, 1, 0], dtype=torch.float32, device=n.device).expand_as(n)

    # Determine whether n is more parallel to proxy_x or proxy_y
    dot_x = torch.abs(torch.sum(n * proxy_x, dim=1))
    dot_y = torch.abs(torch.sum(n * proxy_y, dim=1))

    # Allocate storage for x_dir and y_dir
    x_dir = torch.zeros_like(n)
    y_dir = torch.zeros_like(n)

    # Case 0: more parallel with proxy_x
    mask_case_0 = dot_x > dot_y
    x_dir[mask_case_0] = torch.cross(proxy_y[mask_case_0], n[mask_case_0])
    y_dir[mask_case_0] = torch.cross(n[mask_case_0], x_dir[mask_case_0])

    # Case 1: more parallel with proxy_y
    mask_case_1 = ~mask_case_0
    y_dir[mask_case_1] = torch.cross(n[mask_case_1], proxy_x[mask_case_1])
    x_dir[mask_case_1] = torch.cross(y_dir[mask_case_1], n[mask_case_1])

    # Normalize the direction vectors to ensure they are unit vectors
    x_dir = torch.nn.functional.normalize(x_dir, dim=1)
    y_dir = torch.nn.functional.normalize(y_dir, dim=1)

    # Stack the direction vectors to form the rotation matrix
    R = torch.stack([x_dir, y_dir, n], dim=-1)

    # Convert the rotation matrix to a quaternion using the corrected function
    q = rotmat2quaternion(R)
    return q


def rotation2normal(q):
    R = quaternion2rotmat(q)
    normal = R[:, :, 2]
    return normal


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_quaternion(R):
    """
    R: [N, 3, 3] rotation matrices
    Return: [N, 4] quaternions in [w, x, y, z] format
    """
    N = R.shape[0]
    q = torch.zeros((N, 4), device=R.device)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0
    cond1 = trace > 0
    t1 = torch.sqrt(trace[cond1] + 1.0) * 2
    q[cond1, 0] = 0.25 * t1
    q[cond1, 1] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / t1
    q[cond1, 2] = (R[cond1, 0, 2] - R[cond1, 2, 0]) / t1
    q[cond1, 3] = (R[cond1, 1, 0] - R[cond1, 0, 1]) / t1

    # Case 2: R[0,0] is largest
    cond2 = ~cond1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    t2 = torch.sqrt(1.0 + R[cond2, 0, 0] - R[cond2, 1, 1] - R[cond2, 2, 2]) * 2
    q[cond2, 0] = (R[cond2, 2, 1] - R[cond2, 1, 2]) / t2
    q[cond2, 1] = 0.25 * t2
    q[cond2, 2] = (R[cond2, 0, 1] + R[cond2, 1, 0]) / t2
    q[cond2, 3] = (R[cond2, 0, 2] + R[cond2, 2, 0]) / t2

    # Case 3: R[1,1] is largest
    cond3 = ~cond1 & ~cond2 & (R[:, 1, 1] > R[:, 2, 2])
    t3 = torch.sqrt(1.0 + R[cond3, 1, 1] - R[cond3, 0, 0] - R[cond3, 2, 2]) * 2
    q[cond3, 0] = (R[cond3, 0, 2] - R[cond3, 2, 0]) / t3
    q[cond3, 1] = (R[cond3, 0, 1] + R[cond3, 1, 0]) / t3
    q[cond3, 2] = 0.25 * t3
    q[cond3, 3] = (R[cond3, 1, 2] + R[cond3, 2, 1]) / t3

    # Case 4: R[2,2] is largest
    cond4 = ~cond1 & ~cond2 & ~cond3
    t4 = torch.sqrt(1.0 + R[cond4, 2, 2] - R[cond4, 0, 0] - R[cond4, 1, 1]) * 2
    q[cond4, 0] = (R[cond4, 1, 0] - R[cond4, 0, 1]) / t4
    q[cond4, 1] = (R[cond4, 0, 2] + R[cond4, 2, 0]) / t4
    q[cond4, 2] = (R[cond4, 1, 2] + R[cond4, 2, 1]) / t4
    q[cond4, 3] = 0.25 * t4

    # Normalize to unit quaternion
    q = q / q.norm(dim=1, keepdim=True)
    return q


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
