from typing import *


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(
        self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs
    ):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)

            # Handle case where model returns tuple (main_output, pose_output)
            if isinstance(pred, tuple) and isinstance(neg_pred, tuple):
                if len(pred) == 2 and len(neg_pred) == 2:
                    pred_main, pred_pose = pred
                    neg_pred_main, neg_pred_pose = neg_pred

                    # Apply CFG to both main output and pose
                    guided_main = (
                        1 + cfg_strength
                    ) * pred_main - cfg_strength * neg_pred_main
                    guided_pose = (
                        1 + cfg_strength
                    ) * pred_pose - cfg_strength * neg_pred_pose

                    return guided_main, guided_pose

            # Handle case where only one is tuple (shouldn't happen but for safety)
            if isinstance(pred, tuple) and not isinstance(neg_pred, tuple):
                pred_main, pred_pose = pred
                guided_main = (1 + cfg_strength) * pred_main - cfg_strength * neg_pred
                return guided_main, pred_pose
            elif not isinstance(pred, tuple) and isinstance(neg_pred, tuple):
                neg_pred_main, _ = neg_pred
                guided_main = (1 + cfg_strength) * pred - cfg_strength * neg_pred_main
                return guided_main

            # Default case: both are single tensors
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
