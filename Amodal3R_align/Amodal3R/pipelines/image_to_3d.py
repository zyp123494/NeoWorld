from contextlib import contextmanager
from typing import *

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ..modules import sparse as sp
from . import samplers
from .base import Pipeline


class mask_patcher(nn.Module):
    def __init__(self):
        super(mask_patcher, self).__init__()

    def forward(self, mask, patch_size=14):
        mask = F.interpolate(mask.float(), size=(518, 518), mode='nearest')  # [B, 1, 518, 518]
        
        patches = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            
        patch_ratio = patches.mean(dim=(-1, -2))  # [B, 1, 37, 37]
        
        patch_ratio = patch_ratio.squeeze(1)  # [B, 37, 37]
        
        return patch_ratio


class Amodal3RImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Amodal3R image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str, ss_path = None) -> "Amodal3RImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(Amodal3RImageTo3DPipeline, Amodal3RImageTo3DPipeline).from_pretrained(path)
        new_pipeline = Amodal3RImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        new_pipeline.mask_patcher = mask_patcher()

        if ss_path is not None:
            checkpoint = torch.load(ss_path, map_location='cuda')
        
            # Extract the sparse structure model state dict from the checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Filter out the sparse structure model weights
                ss_model_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('net.ss_model.'):
                        # Remove the 'net.ss_model.' prefix
                        new_key = key[len('net.ss_model.'):]
                        ss_model_state_dict[new_key] = value
                
                if ss_model_state_dict:
                    print("Overriding sparse structure model with pose-aware weights...")
                    new_pipeline.models['sparse_structure_flow_model'].load_state_dict(ss_model_state_dict)
        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        # dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model = torch.hub.load('/root/.cache/torch/hub/facebookresearch_dinov2_main', name, pretrained=True,source='local')
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform
    
    def preprocess_image_w_mask(self, input, mask, kernel_size=3):
        image = input.astype(np.float32) / 255
        mask_ori = mask.astype(np.float32)
        mask = (mask_ori < 127).astype(np.uint8)
        if kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask_occlude = cv2.dilate(mask, kernel, iterations=1) 
        else:
            mask_occlude = mask
        mask_bg = (mask_ori>230).astype(np.uint8)
        mask = mask_occlude | mask_bg
        image = image * (1 - mask[:, :, None])
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.resize((518, 518), Image.Resampling.LANCZOS)
        mask_occ = np.zeros(mask.shape)
        mask_occ[mask_occlude==1] = 1
        return image, mask, mask_occ
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond_w_masks(self, image: Union[torch.Tensor, list[Image.Image]], mask: Union[torch.Tensor, list[Image.Image]], masks_occ: Union[torch.Tensor, list[Image.Image], None], mask_encode_type: Literal['dino', 'repeat'] = 'repeat', stage: Literal['ss', 'slat'] = 'ss') -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        mask = [torch.from_numpy(mask).unsqueeze(0).float() for mask in mask]
        mask = torch.stack(mask).to(self.device)
        masks_occ = [torch.from_numpy(masks_occ).unsqueeze(0).float() for masks_occ in masks_occ]
        masks_occ = torch.stack(masks_occ).to(self.device)
        mask = self.mask_patcher(1-mask)
        masked_feat = mask.view(mask.shape[0], -1).unsqueeze(-1).repeat(1, 1, 1024)
        masks_occ = self.mask_patcher(masks_occ)
        masks_occ = masks_occ.view(masks_occ.shape[0], -1).unsqueeze(-1).repeat(1, 1, 1024)
        cond = torch.cat([cond, masked_feat], dim=1)
        cond = torch.cat([cond, masks_occ], dim=1)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        sample_pose: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            sample_pose (bool): Whether to sample pose along with sparse structure.
            
        Returns:
            torch.Tensor or tuple: Coordinates of sparse structure, optionally with pose
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        
        if sample_pose:
            # For pose sampling, we need to provide pose noise to the model
            pose_noise = torch.randn(num_samples, 6).to(self.device)
            result = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond,
                pose=pose_noise,  # Pass pose noise to enable pose prediction
                **sampler_params,
                verbose=True
            )
            z_s = result.samples
            pose = result.pose if hasattr(result, 'pose') and result.pose is not None else torch.zeros(num_samples, 6).to(self.device)
        else:
            result = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True
            )
            z_s = result.samples
            pose = None
        
        decoder = self.models['sparse_structure_decoder']
        ss = decoder(z_s)
        coords = torch.argwhere(ss>0)[:, [0, 2, 3, 4]].int()
        
        if sample_pose and pose is not None:
            return coords, pose
        else:
            return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian'],
        preprocess_image: bool = True,
        sample_pose: bool = False,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
            sample_pose (bool): Whether to sample pose along with 3D generation.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        
        # Sample sparse structure and optionally pose
        if sample_pose:
            coords, pose = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, sample_pose=True)
        else:
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, sample_pose=False)
            pose = None
        
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        result = self.decode_slat(slat, formats)
        
        # Add pose to the result if sampled
        if sample_pose and pose is not None:
            result['pose'] = [pose.cpu().numpy()]
        
        return result

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        sampler._old_inference_model = sampler._inference_model

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, '_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        masks: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian'],
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
        erode_kernel_size: int = 3,
        sample_pose: bool = False,
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            sample_pose (bool): Whether to sample pose along with 3D generation.
            preprocess_image (bool): Whether to preprocess the image.
        """
        images, masks, masks_occ = zip(*[self.preprocess_image_w_mask(image, mask, erode_kernel_size) for image, mask in zip(images, masks, strict=False)], strict=False)
        images = list(images)
        masks = list(masks)
        masks_occ = list(masks_occ)
        cond_stage_1 = self.get_cond_w_masks(images, masks, masks_occ=masks_occ, mask_encode_type="patcher", stage = "ss")
        cond_stage_2 = self.get_cond_w_masks(images, masks, masks_occ=masks_occ, mask_encode_type="patcher", stage = "slat")
        cond_stage_1['neg_cond'] = cond_stage_1['neg_cond'][:1]
        cond_stage_2['neg_cond'] = cond_stage_2['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        
        # Sample sparse structure and optionally pose
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            if sample_pose:
                coords, pose = self.sample_sparse_structure(cond_stage_1, num_samples, sparse_structure_sampler_params, sample_pose=True)
            else:
                coords = self.sample_sparse_structure(cond_stage_1, num_samples, sparse_structure_sampler_params, sample_pose=False)
                pose = None
        
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond_stage_2, coords, slat_sampler_params)
        
        # Decode the structured latent
        result = self.decode_slat(slat, formats)
        
        # Add pose to the result if sampled
        if sample_pose and pose is not None:
            result['pose'] = [pose.cpu().numpy()]
        
        return result