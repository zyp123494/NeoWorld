###
# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
# All rights reserved.
###
import numpy as np


class GSParams:
    def __init__(self):
        self.sh_degree = 3
        self.images = "images"
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.use_depth = False

        self.opacity_reset_interval = 3000
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 99999
        self.max_screen_size = 30
        self.scene_extent = None

        self.iterations = 100
        self.position_lr_init = 0.000
        self.position_lr_final = 0.0000
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 100
        self.feature_lr = 0.00
        self.opacity_lr = 0.08
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.seg_lr = 0.1
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densify_from_iter = 80
        self.prune_from_iter = 80
        self.densification_interval = 80

        self.convert_SHs_python = True
        self.compute_cov3D_python = False
        self.debug = False


class CameraParams:
    def __init__(self, H: int = 512, W: int = 512):
        self.H = H
        self.W = W
        self.focal = (5.8269e02, 5.8269e02)
        self.fov = (
            2 * np.arctan(self.W / (2 * self.focal[0])),
            2 * np.arctan(self.H / (2 * self.focal[1])),
        )
        self.K = np.array(
            [
                [self.focal[0], 0.0, self.W / 2],
                [0.0, self.focal[1], self.H / 2],
                [0.0, 0.0, 1.0],
            ]
        ).astype(np.float32)
