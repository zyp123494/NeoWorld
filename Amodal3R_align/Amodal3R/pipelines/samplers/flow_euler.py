from typing import *

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """

    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (
            self.sigma_min + (1 - self.sigma_min) * t
        ) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor(
            [1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32
        )
        model_output = model(x_t, t, cond, **kwargs)

        # Check if model returns pose along with main output
        if isinstance(model_output, tuple) and len(model_output) == 2:
            main_output, pose_output = model_output
            return main_output, pose_output
        else:
            return model_output

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        model_output = self._inference_model(model, x_t, t, cond, **kwargs)

        if isinstance(model_output, tuple) and len(model_output) == 2:
            pred_v, pose_v = model_output
            pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
            return pred_x_0, pred_eps, pred_v, pose_v
        else:
            pred_v = model_output
            pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
            return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        pose_t: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Sample x_{t-1} from the model using Euler method.

        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            pose_t: The pose tensor at time t (if pose sampling is enabled).
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
            - 'pred_pose_prev': pose_{t-1} (if pose sampling is enabled).
        """
        # Add pose to kwargs if provided
        if pose_t is not None:
            kwargs["pose"] = pose_t

        model_prediction = self._get_model_prediction(model, x_t, t, cond, **kwargs)

        if len(model_prediction) == 4:  # Has pose prediction
            pred_x_0, pred_eps, pred_v, pose_v = model_prediction
            pred_x_prev = x_t - (t - t_prev) * pred_v
            pred_pose_prev = (
                pose_t - (t - t_prev) * pose_v if pose_t is not None else None
            )
            return edict(
                {
                    "pred_x_prev": pred_x_prev,
                    "pred_x_0": pred_x_0,
                    "pred_pose_prev": pred_pose_prev,
                }
            )
        else:  # No pose prediction
            pred_x_0, pred_eps, pred_v = model_prediction
            pred_x_prev = x_t - (t - t_prev) * pred_v
            return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": [], "pose": None})

        # Initialize pose sampling if pose noise is provided
        pose_sample = None
        if "pose" in kwargs:
            pose_sample = kwargs["pose"]  # Start with pose noise

        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            # Pass current pose state to sample_once
            out = self.sample_once(
                model,
                sample,
                t,
                t_prev,
                cond,
                pose_t=pose_sample,
                **{k: v for k, v in kwargs.items() if k != "pose"},
            )

            # Update sample and pose
            sample = out.pred_x_prev
            if hasattr(out, "pred_pose_prev") and out.pred_pose_prev is not None:
                pose_sample = out.pred_pose_prev

            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        ret.samples = sample

        # Add final pose if it was being tracked
        if pose_sample is not None:
            ret.pose = pose_sample

        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(
            model,
            noise,
            cond,
            steps,
            rescale_t,
            verbose,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            **kwargs,
        )


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(
            model,
            noise,
            cond,
            steps,
            rescale_t,
            verbose,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            **kwargs,
        )
