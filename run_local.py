import gc
import json
import os
import random
import warnings
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from random import randint

import imageio
import numpy as np
import pytorch3d
import torch
import torch.nn.functional as F
from Amodal3R_align.Amodal3R.pipelines import Amodal3RImageTo3DPipeline
from arguments import GSParams
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from gaussian_renderer import render
from kornia.morphology import dilation
from marigold_lcm.marigold_pipeline import MarigoldNormalsPipeline, MarigoldPipeline
from models.models import KeyframeGen
from ObjectClear.objectclear.pipelines import ObjectClearPipeline
from omegaconf import OmegaConf
from PIL import Image
from scene import GaussianModel, Scene
from syncdiffusion.syncdiffusion_model import SyncDiffusion
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from util.completion_utils import complete_and_align_objects, get_completion_index_2d
from util.LLM import TextpromptGen
from util.segment_utils import create_mask_generator_repvit
from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from util.utils import (
    ade_id2label,
    convert_pt3d_cam_to_3dgs_cam,
    interp_poses,
    load_example_yaml,
    prepare_scheduler,
    soft_stitching,
    visualize_seg,
)
from utils.loss import l1_loss, ssim

warnings.filterwarnings("ignore")


xyz_scale = 1000
scene_name = None
view_matrix_fixed = np.array(
    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0.2, 0.5, 1]]
)
theta = np.radians(-3)
rotation_matrix_x = np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1],
    ]
)
view_matrix_fixed = np.dot(view_matrix_fixed, rotation_matrix_x)
view_matrix_fixed = view_matrix_fixed.flatten().tolist()

background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device="cuda")


kf_gen = None
gaussians = None
opt = None
scene_dict = None
style_prompt = None
pt_gen = None
exclude_sky = False


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"running with seed: {seed}.")


def run(config):
    global scene_name, kf_gen, gaussians, opt, background, scene_dict, style_prompt, pt_gen, exclude_sky

    ###### ------------------ Load modules ------------------ ######

    seeding(config["seed"])
    example = config["example_name"]

    segment_processor = OneFormerProcessor.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large"
    )

    segment_model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large"
    ).to("cuda")

    mask_generator = create_mask_generator_repvit()

    inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        config["stable_diffusion_checkpoint"],
        safety_checker=None,
        torch_dtype=torch.bfloat16,
    ).to(config["device"])
    inpainter_pipeline.scheduler = DDIMScheduler.from_config(
        inpainter_pipeline.scheduler.config
    )
    inpainter_pipeline.unet.set_attn_processor(AttnProcessor2_0())
    inpainter_pipeline.vae.set_attn_processor(AttnProcessor2_0())

    rotation_path = config["rotation_path"][: config["num_scenes"]]
    assert len(rotation_path) == config["num_scenes"]

    removal_pipeline = ObjectClearPipeline.from_pretrained_with_custom_modules(
        "jixin0101/ObjectClear",
        torch_dtype=torch.float16,
        apply_attention_guided_fusion=True,
        variant="fp16",
    )
    removal_pipeline.to(config["device"])

    depth_model = MarigoldPipeline.from_pretrained(
        "prs-eth/marigold-depth-v1-1", torch_dtype=torch.bfloat16
    ).to(config["device"])
    depth_model.scheduler = EulerDiscreteScheduler.from_config(
        depth_model.scheduler.config
    )
    depth_model.scheduler = prepare_scheduler(depth_model.scheduler)

    normal_estimator = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-v1-1", torch_dtype=torch.bfloat16
    ).to(config["device"])  # torch_dtype=torch.bfloat16

    completion_pipeline = Amodal3RImageTo3DPipeline.from_pretrained(
        "Sm0kyWu/Amodal3R",
        ss_path=config["ss_path"],
    )
    completion_pipeline.cuda()

    print(
        "###### ------------------ Keyframe (the major part of point clouds) generation ------------------ ######"
    )
    kf_gen = KeyframeGen(
        config=config,
        inpainter_pipeline=inpainter_pipeline,
        mask_generator=mask_generator,
        depth_model=depth_model,
        segment_model=segment_model,
        segment_processor=segment_processor,
        normal_estimator=normal_estimator,
        removal_pipeline=removal_pipeline,
        rotation_path=rotation_path,
        inpainting_resolution=config["inpainting_resolution_gen"],
    ).to(config["device"])

    yaml_data = load_example_yaml(config["example_name"], "examples/examples.yaml")
    (
        content_prompt,
        style_prompt,
        adaptive_negative_prompt,
        background_prompt,
    ) = (
        yaml_data["content_prompt"],
        yaml_data["style_prompt"],
        yaml_data["negative_prompt"],
        yaml_data.get("background", None),
    )

    # Initialize VLM for inpainting prompt generation
    pt_gen = TextpromptGen(kf_gen.run_dir)
    if adaptive_negative_prompt != "":
        adaptive_negative_prompt += ", "

    start_keyframe = (
        Image.open(yaml_data["image_filepath"]).convert("RGB").resize((512, 512))
    )
    kf_gen.image_latest = ToTensor()(start_keyframe).unsqueeze(0).to(config["device"])

    if config["gen_sky_image"] or (
        not os.path.exists(f"examples/sky_images/{example}/sky_0.png")
        and not os.path.exists(f"examples/sky_images/{example}/sky_1.png")
    ):
        syncdiffusion_model = SyncDiffusion(config["device"], sd_version="2.0-inpaint")
    else:
        syncdiffusion_model = None

    sky_mask = kf_gen.generate_sky_mask().float()

    kf_gen.generate_sky_pointcloud(
        syncdiffusion_model,
        image=kf_gen.image_latest,
        mask=sky_mask,
        gen_sky=config["gen_sky_image"],
        style=style_prompt,
    )

    kf_gen.recompose_image_latest_and_set_current_pc(scene_name=scene_name)

    content_list = content_prompt.split(",")
    scene_name = content_list[0]
    entities = content_list[1:]
    scene_dict = {
        "scene_name": scene_name,
        "entities": entities,
        "style": style_prompt,
        "background": background_prompt,
    }
    inpainting_prompt = content_prompt

    kf_gen.increment_kf_idx()
    ###### ------------------ Main loop ------------------ ######

    if not os.path.exists(f"examples/sky_images/{example}/finished_3dgs_sky_tanh.ply"):
        traindatas = kf_gen.convert_to_3dgs_traindata(
            xyz_scale=xyz_scale, remove_threshold=None, use_no_loss_mask=False
        )
        if config["gen_layer"]:
            traindata, traindata_sky, traindata_layer = traindatas
        else:
            traindata, traindata_sky = traindatas
        gaussians = GaussianModel(
            sh_degree=0, floater_dist2_threshold=9e9, codebook=kf_gen.codebook
        )
        opt = GSParams()
        opt.max_screen_size = (
            100  # Sky is supposed to be big; set a high max screen size
        )
        opt.scene_extent = 1.5  # Sky is supposed to be big; set a high scene extent
        opt.densify_from_iter = 200  # Need to do some densify
        opt.prune_from_iter = 200  # Don't prune for sky because sky 3DGS are supposed to be big; prevent it by setting a high prune iter
        opt.densify_grad_threshold = (
            1.0  # Do not need to densify; Set a high threshold to prevent densifying
        )
        opt.iterations = 399  # More iterations than 100 needed for sky
        scene = Scene(traindata_sky, gaussians, opt, is_sky=True)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config["runs_dir"]) / f"{dt_string}_gaussian_scene_sky"
        train_gaussian(gaussians, scene, opt, save_dir, initialize_scaling=False)
        gaussians.save_ply_with_filter(
            f"examples/sky_images/{example}/finished_3dgs_sky_tanh.ply"
        )
    else:
        gaussians = GaussianModel(sh_degree=0, codebook=kf_gen.codebook)
        gaussians.load_ply_with_filter(
            f"examples/sky_images/{example}/finished_3dgs_sky_tanh.ply"
        )  # pure sky

    gaussians.visibility_filter_all = torch.zeros(
        gaussians.get_xyz_all.shape[0], dtype=torch.bool, device="cuda"
    )
    gaussians.delete_mask_all = torch.zeros(
        gaussians.get_xyz_all.shape[0], dtype=torch.bool, device="cuda"
    )
    gaussians.is_sky_filter = torch.ones(
        gaussians.get_xyz_all.shape[0], dtype=torch.bool, device="cuda"
    )
    gaussians.is_fg_filter = torch.zeros(
        gaussians.get_xyz_all.shape[0], dtype=torch.bool, device="cuda"
    )

    if (
        config["load_gen"]
        and os.path.exists(f"examples/sky_images/{example}/finished_3dgs.ply")
        and os.path.exists(f"examples/sky_images/{example}/visibility_filter_all.pth")
        and os.path.exists(f"examples/sky_images/{example}/is_sky_filter.pth")
        and os.path.exists(f"examples/sky_images/{example}/delete_mask_all.pth")
    ):
        print("Loading existing 3DGS...")
        gaussians = GaussianModel(sh_degree=0, codebook=kf_gen.codebook)
        gaussians.load_ply_with_filter(
            f"examples/sky_images/{example}/finished_3dgs.ply"
        )
        gaussians.visibility_filter_all = torch.load(
            f"examples/sky_images/{example}/visibility_filter_all.pth"
        ).to("cuda")
        gaussians.is_sky_filter = torch.load(
            f"examples/sky_images/{example}/is_sky_filter.pth"
        ).to("cuda")
        gaussians.delete_mask_all = torch.load(
            f"examples/sky_images/{example}/delete_mask_all.pth"
        ).to("cuda")
    opt = GSParams()

    ### First scene 3DGS
    if config["gen_layer"]:
        traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(
            xyz_scale=xyz_scale
        )
        if not traindata_layer["pcd_points"].shape[-1] == 0:
            gaussians = GaussianModel(
                sh_degree=0, previous_gaussian=gaussians, codebook=kf_gen.codebook
            )
            scene = Scene(traindata_layer, gaussians, opt)
            # dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            save_dir = f"{kf_gen.run_dir}/gaussian_scene_layer{0:02d}"

            train_gaussian(
                gaussians, scene, opt, save_dir, train_seg=True
            )  # Base layer training
        # gaussians.update_fg_filter(is_fg = False)
    else:
        traindata = kf_gen.convert_to_3dgs_traindata_latest(
            xyz_scale=xyz_scale, use_no_loss_mask=False
        )

    i = 0
    if (kf_gen.mask_disocclusion > 0.5).any():
        gaussians = GaussianModel(
            sh_degree=0, previous_gaussian=gaussians, codebook=kf_gen.codebook
        )
        scene = Scene(traindata, gaussians, opt)

        save_dir = f"{kf_gen.run_dir}/gaussian_scene{i:02d}"

        train_gaussian(gaussians, scene, opt, save_dir, train_seg=True, is_fg=True)

    current_pt3d_cam = kf_gen.cameras[0]
    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam, xyz_scale=xyz_scale)

    R = torch.tensor(tdgs_cam.R, device=torch.device("cuda"), dtype=torch.float32)
    T = torch.tensor(tdgs_cam.T, device=torch.device("cuda"), dtype=torch.float32)

    xyz_cam = gaussians.get_xyz_all @ R + T[None, :]
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    z = torch.clamp(z, min=0.001)

    x = x / z * tdgs_cam.focal_x + tdgs_cam.image_width / 2.0
    y = y / z * tdgs_cam.focal_y + tdgs_cam.image_height / 2.0
    # filter out of image points
    visible_mask = (
        (x >= 0) & (x <= tdgs_cam.image_width) & (y >= 0) & (y <= tdgs_cam.image_height)
    )
    seg_label = (gaussians.get_seg_all[visible_mask] @ gaussians.codebook.T).argmax(
        -1
    )  # 【N】

    with torch.no_grad():
        render_pkg = render(
            tdgs_cam,
            gaussians,
            opt,
            bg_color=background,
            render_seg=True,
            render_current=False,
        )
    gt_depth = render_pkg["median_depth"]
    sim_image = render_pkg["render"].permute(1, 2, 0).cpu().numpy()
    sim_image = (sim_image * 255).astype(np.uint8)
    sim_seg = F.normalize(render_pkg["render_seg"], dim=0)
    sim_seg = (
        torch.einsum("nk,khw->nhw", gaussians.codebook, sim_seg).argmax(0).cpu().numpy()
    )  # [H,W]

    completion_index = get_completion_index_2d(
        sim_seg, kf_gen.segments_info, min_area=900
    )

    gaussians, _ = complete_and_align_objects(
        gaussians=gaussians,
        instance_index_list=completion_index[: config["num_completion"]],
        completion_pipeline=completion_pipeline,
        sim_image=sim_image,
        sim_seg=sim_seg,
        tdgs_cam=tdgs_cam,
        gt_depth=gt_depth,
        opt=opt,
        is_3d_fg=None,
        example=example,
        output_dir="./output",
    )

    w2c_transforms = []
    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(
        kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale
    )
    gaussians.set_inscreen_points_to_visible(tdgs_cam)

    trans_matrix = np.zeros([4, 4])
    trans_matrix[-1, -1] = 1
    trans_matrix[:3, :3] = np.transpose(tdgs_cam.R)
    trans_matrix[:3, 3] = tdgs_cam.T
    w2c_transforms.append(trans_matrix)

    config["use_gpt"] = True

    for i in range(config["num_scenes"]):
        inpainting_prompt = pt_gen.generate_prompt(
            style=style_prompt,
            entities=scene_dict["entities"],
            background=scene_dict["background"],
            scene_name=scene_dict["scene_name"],
        )
        scene_name = (
            scene_dict["scene_name"]
            if isinstance(scene_dict["scene_name"], str)
            else scene_dict["scene_name"][0]
        )

        if config["use_gpt"]:
            json_path = (
                Path(pt_gen.root_path)
                / f"scene_{str(pt_gen.scene_num + 1).zfill(2)}.json"
            )

            if json_path.exists():
                with open(json_path) as f:
                    scene_dict = json.load(f)
                pt_gen.scene_num += 1
                pt_gen.id += 1
            else:
                scene_dict = pt_gen.wonder_next_scene(
                    scene_name=scene_name,
                    entities=scene_dict["entities"],
                    style=style_prompt,
                    background=scene_dict["background"],
                    change_scene_name_by_user=False,
                )

        inpainting_prompt = pt_gen.generate_prompt(
            style=style_prompt,
            entities=scene_dict["entities"],
            background=scene_dict["background"],
            scene_name=scene_dict["scene_name"],
        )
        scene_name = (
            scene_dict["scene_name"]
            if isinstance(scene_dict["scene_name"], str)
            else scene_dict["scene_name"][0]
        )

        print("inpainting prompt:", inpainting_prompt)

        ###### ------------------ Keyframe (the major part of point clouds) generation ------------------ ######
        kf_gen.set_kf_param(
            inpainting_resolution=config["inpainting_resolution_gen"],
            inpainting_prompt=inpainting_prompt,
            adaptive_negative_prompt=adaptive_negative_prompt,
        )

        current_pt3d_cam = kf_gen.cameras[i + 1]

        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam, xyz_scale=xyz_scale)
        kf_gen.set_current_camera(current_pt3d_cam, archive_camera=True)

        trans_matrix = np.zeros([4, 4])
        trans_matrix[-1, -1] = 1
        trans_matrix[:3, :3] = np.transpose(tdgs_cam.R)
        trans_matrix[:3, 3] = tdgs_cam.T
        w2c_transforms.append(trans_matrix)

        if exclude_sky:
            with torch.no_grad():
                render_pkg = render(
                    tdgs_cam, gaussians, opt, background, render_seg=True
                )
                render_pkg_nosky = render(
                    tdgs_cam, gaussians, opt, background, exclude_sky=True
                )
                render_pkg_base = render(
                    tdgs_cam,
                    gaussians,
                    opt,
                    background,
                    exclude_fg=True,
                    render_seg=True,
                )
                render_pkg_fg = render(
                    tdgs_cam, gaussians, opt, background, fg_only=True, render_seg=True
                )

            side_sky_height = 128

            render_seg = F.normalize(render_pkg_fg["render_seg"], dim=0)
            render_seg = torch.einsum(
                "nk,khw->nhw", gaussians.codebook, render_seg
            ).argmax(0)
            kf_gen.prev_panoptic_mask_fg = render_seg  # [H,W]

            render_seg_base = F.normalize(render_pkg_base["render_seg"], dim=0)
            render_seg_base = torch.einsum(
                "nk,khw->nhw", gaussians.codebook, render_seg_base
            ).argmax(0)
            kf_gen.prev_panoptic_mask_bg = render_seg_base  # [H,W]

            kf_gen.prev_image_base = render_pkg_base["render"].unsqueeze(0)
            kf_gen.prev_is_fg_mask = (render_pkg_fg["final_opacity"] > 0.01).squeeze()

            inpaint_mask_0p5_nosky = render_pkg_nosky["final_opacity"] < 0.6
            inpaint_mask_0p0_nosky = (
                render_pkg_nosky["final_opacity"] < 0.01
            )  # Should not have holes in existing regions
            inpaint_mask_0p5 = render_pkg["final_opacity"] < 0.6
            inpaint_mask_0p0 = (
                render_pkg["final_opacity"] < 0.01
            )  # Should not have holes in existing regions

            mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config["device"])
            mask_using_full_render[:, :, :side_sky_height, :] = 1

            mask_using_nosky_render = 1 - mask_using_full_render

            outpaint_condition_image = (
                render_pkg_nosky["render"] * mask_using_nosky_render
                + render_pkg["render"] * mask_using_full_render
            )
            viz = outpaint_condition_image[0].permute(1, 2, 0).detach().cpu().numpy()
            viz = (viz * 255).astype(np.uint8)
            viz = viz[..., ::-1]

            fill_mask = (
                inpaint_mask_0p5_nosky * mask_using_nosky_render
                + inpaint_mask_0p5 * mask_using_full_render
            )
            outpaint_mask = (
                inpaint_mask_0p0_nosky * mask_using_nosky_render
                + inpaint_mask_0p0 * mask_using_full_render
            )
            outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())
            exclude_sky = False
        else:
            with torch.no_grad():
                render_pkg = render(
                    tdgs_cam, gaussians, opt, background, render_seg=True
                )
                render_pkg_nosky = render(
                    tdgs_cam, gaussians, opt, background, exclude_sky=True
                )
                render_pkg_base = render(
                    tdgs_cam,
                    gaussians,
                    opt,
                    background,
                    exclude_fg=True,
                    render_seg=True,
                )
                render_pkg_fg = render(
                    tdgs_cam, gaussians, opt, background, fg_only=True, render_seg=True
                )

            render_seg = F.normalize(render_pkg_fg["render_seg"], dim=0)
            render_seg = torch.einsum(
                "nk,khw->nhw", gaussians.codebook, render_seg
            ).argmax(0)
            kf_gen.prev_panoptic_mask_fg = render_seg  # [H,W]

            render_seg_base = F.normalize(render_pkg_base["render_seg"], dim=0)
            render_seg_base = torch.einsum(
                "nk,khw->nhw", gaussians.codebook, render_seg_base
            ).argmax(0)
            kf_gen.prev_panoptic_mask_bg = render_seg_base  # [H,W]

            kf_gen.prev_image_base = render_pkg_base["render"].unsqueeze(0)
            kf_gen.prev_is_fg_mask = (render_pkg_fg["final_opacity"] > 0.01).squeeze()

            side_sky_height = 128
            sky_cond_width = 40

            inpaint_mask_0p5_nosky = render_pkg_nosky["final_opacity"] < 0.6
            inpaint_mask_0p0_nosky = (
                render_pkg_nosky["final_opacity"] < 0.01
            )  # Should not have holes in existing regions
            inpaint_mask_0p5 = render_pkg["final_opacity"] < 0.6
            inpaint_mask_0p0 = (
                render_pkg["final_opacity"] < 0.01
            )  # Should not have holes in existing regions
            fg_mask_0p5_nosky = ~inpaint_mask_0p5_nosky.clone()
            foreground_cols = torch.sum(fg_mask_0p5_nosky == 1, dim=1) > 150  # [1, 512]
            foreground_cols_idx = torch.nonzero(foreground_cols, as_tuple=True)[1]

            mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config["device"])
            if foreground_cols_idx.numel() > 0:
                min_index = foreground_cols_idx.min().item()
                max_index = foreground_cols_idx.max().item()
                mask_using_full_render[:, :, :, min_index : max_index + 1] = 1
            mask_using_full_render[:, :, :sky_cond_width, :] = 1
            mask_using_full_render[:, :, :side_sky_height, :sky_cond_width] = 1
            mask_using_full_render[:, :, :side_sky_height, -sky_cond_width:] = 1

            mask_using_nosky_render = 1 - mask_using_full_render

            outpaint_condition_image = (
                render_pkg_nosky["render"] * mask_using_nosky_render
                + render_pkg["render"] * mask_using_full_render
            )
            viz = outpaint_condition_image[0].permute(1, 2, 0).detach().cpu().numpy()
            viz = (viz * 255).astype(np.uint8)
            viz = viz[..., ::-1]

            fill_mask = (
                inpaint_mask_0p5_nosky * mask_using_nosky_render
                + inpaint_mask_0p5 * mask_using_full_render
            )
            outpaint_mask = (
                inpaint_mask_0p0_nosky * mask_using_nosky_render
                + inpaint_mask_0p0 * mask_using_full_render
            )
            outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())

        kf_gen.current_inpainting_mask = outpaint_mask
        image = outpaint_condition_image[0].permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray((image * 255).astype(np.uint8))
        image.save(
            f"{kf_gen.run_dir}/images/layer/{kf_gen.kf_idx:02d}_outpaint_image.png"
        )

        image = outpaint_mask.squeeze().cpu().numpy()
        image = Image.fromarray((image * 255).astype(np.uint8))
        image.save(
            f"{kf_gen.run_dir}/images/layer/{kf_gen.kf_idx:02d}_outpaint_mask.png"
        )

        # Save the generated prompt
        prompt_save_path = (
            f"{kf_gen.run_dir}/prompts/generated_prompt_{kf_gen.kf_idx:02d}.txt"
        )
        os.makedirs(os.path.dirname(prompt_save_path), exist_ok=True)
        with open(prompt_save_path, "w", encoding="utf-8") as f:
            f.write(inpainting_prompt)

        inpaint_output = kf_gen.inpaint(
            outpaint_condition_image,
            inpaint_mask=outpaint_mask,
            fill_mask=fill_mask,
            inpainting_prompt=inpainting_prompt,
            mask_strategy=np.max,
            diffusion_steps=50,
        )

        sem_seg = kf_gen.update_sky_mask()
        recomposed = soft_stitching(
            render_pkg["render"], kf_gen.image_latest, kf_gen.sky_mask_latest
        )  # Replace generated sky with rendered sky

        depth_should_be = render_pkg["median_depth"][0:1].unsqueeze(0) / xyz_scale
        mask_to_align_depth = (depth_should_be < 0.006 * 0.8) & (
            depth_should_be > 0.001
        )  # If opacity < 0.5, then median_depth = -1

        ground_mask = kf_gen.generate_ground_mask(sem_map=sem_seg)[None, None]
        depth_should_be_ground = kf_gen.compute_ground_depth(camera_height=0.0003)
        ground_outputable_mask = (depth_should_be_ground > 0.001) & (
            depth_should_be_ground < 0.006 * 0.8
        )

        joint_mask = mask_to_align_depth | (ground_mask & ground_outputable_mask)
        depth_should_be_joint = torch.where(
            mask_to_align_depth, depth_should_be, depth_should_be_ground
        )

        with torch.no_grad():
            depth_guide_joint, _ = kf_gen.get_depth(
                kf_gen.image_latest,
                target_depth=depth_should_be_joint,
                mask_align=joint_mask,
                archive_output=True,
                diffusion_steps=30,
                guidance_steps=8,
            )

        kf_gen.refine_disp_with_segments(
            no_refine_mask=ground_mask.squeeze().cpu().numpy()
        )

        kf_gen.image_latest = recomposed
        if config["gen_layer"]:
            kf_gen.generate_layer(pred_semantic_map=sem_seg, scene_name=scene_name)

            depth_should_be = kf_gen.depth_latest_init
            mask_to_align_depth = ~(kf_gen.mask_disocclusion.bool()) & (
                depth_should_be < 0.006 * 0.8
            )
            mask_to_farther_depth = kf_gen.mask_disocclusion.bool() & (
                depth_should_be < 0.006 * 0.8
            )
            with torch.no_grad():
                kf_gen.depth, kf_gen.disparity = kf_gen.get_depth(
                    kf_gen.image_latest,
                    archive_output=True,
                    target_depth=depth_should_be,
                    mask_align=mask_to_align_depth,
                    mask_farther=mask_to_farther_depth,
                    diffusion_steps=30,
                    guidance_steps=8,
                )
            kf_gen.refine_disp_with_segments(
                no_refine_mask=ground_mask.squeeze().cpu().numpy(),
                existing_mask=~(kf_gen.mask_disocclusion)
                .bool()
                .squeeze()
                .cpu()
                .numpy(),
                existing_disp=kf_gen.disparity_latest_init.squeeze().cpu().numpy(),
            )
            wrong_depth_mask = kf_gen.depth_latest < kf_gen.depth_latest_init
            kf_gen.depth_latest[wrong_depth_mask] = (
                kf_gen.depth_latest_init[wrong_depth_mask] + 0.0001
            )
            kf_gen.depth_latest = (
                kf_gen.mask_disocclusion * kf_gen.depth_latest
                + (1 - kf_gen.mask_disocclusion) * kf_gen.depth_latest_init
            )
            kf_gen.update_sky_mask()
            valid_px_mask = outpaint_mask * (~kf_gen.sky_mask_latest)
            kf_gen.update_current_pc_by_kf(
                image=kf_gen.image_latest,
                depth=kf_gen.depth_latest,
                valid_mask=valid_px_mask,
            )  # Base only
            kf_gen.update_current_pc_by_kf(
                image=kf_gen.image_latest_init,
                depth=kf_gen.depth_latest_init,
                valid_mask=kf_gen.mask_disocclusion * outpaint_mask,
                gen_layer=True,
            )  # Object layer
        else:
            valid_px_mask = outpaint_mask * (~kf_gen.sky_mask_latest)
            kf_gen.update_current_pc_by_kf(
                image=kf_gen.image_latest,
                depth=kf_gen.depth_latest,
                valid_mask=valid_px_mask,
            )
        kf_gen.archive_latest()

        if config["gen_layer"]:
            traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(
                xyz_scale=xyz_scale
            )
            if not traindata_layer["pcd_points"].shape[-1] == 0:
                gaussians = GaussianModel(
                    sh_degree=0, previous_gaussian=gaussians, codebook=kf_gen.codebook
                )
                scene = Scene(traindata_layer, gaussians, opt)
                save_dir = f"{kf_gen.run_dir}/gaussian_scene_layer{i + 1:02d}"

                train_gaussian(
                    gaussians, scene, opt, save_dir, train_seg=True
                )  # Base layer training

        else:
            traindata = kf_gen.convert_to_3dgs_traindata_latest(
                xyz_scale=xyz_scale, use_no_loss_mask=False
            )

        if traindata["pcd_points"].shape[-1] == 0 or (
            not (kf_gen.mask_disocclusion > 0.5).any()
        ):
            gaussians.set_inscreen_points_to_visible(tdgs_cam)

            kf_gen.increment_kf_idx()
            continue

        mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config["device"])
        x = torch.sum(fg_mask_0p5_nosky == 1, dim=2) > 0  # [1, 512]
        x_idx = torch.nonzero(x, as_tuple=True)[1]
        if foreground_cols_idx.numel() > 0:
            min_index = foreground_cols_idx.min().item()
            max_index = foreground_cols_idx.max().item()
            mask_using_full_render[
                :, :, : x_idx.max().item(), min_index : max_index + 1
            ] = 1

        mask_using_nosky_render = 1 - mask_using_full_render

        gaussians = GaussianModel(
            sh_degree=0, previous_gaussian=gaussians, codebook=kf_gen.codebook
        )
        scene = Scene(traindata, gaussians, opt)
        # dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = f"{kf_gen.run_dir}/gaussian_scene{i + 1:02d}"
        train_gaussian(gaussians, scene, opt, save_dir, train_seg=True, is_fg=True)
        # gaussians.update_fg_filter(is_fg = True)

        gaussians.set_inscreen_points_to_visible(tdgs_cam)

        R = torch.tensor(tdgs_cam.R, device=torch.device("cuda"), dtype=torch.float32)
        T = torch.tensor(tdgs_cam.T, device=torch.device("cuda"), dtype=torch.float32)

        xyz_cam = gaussians.get_xyz_all @ R + T[None, :]
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        z = torch.clamp(z, min=0.001)

        x = x / z * tdgs_cam.focal_x + tdgs_cam.image_width / 2.0
        y = y / z * tdgs_cam.focal_y + tdgs_cam.image_height / 2.0
        # filter out of image points
        visible_mask = (
            (x >= 0)
            & (x <= tdgs_cam.image_width)
            & (y >= 0)
            & (y <= tdgs_cam.image_height)
        )

        # Use new completion and alignment function
        # Render depth map
        with torch.no_grad():
            render_pkg = render(
                tdgs_cam, gaussians, opt, bg_color=background, render_seg=True
            )
        gt_depth = render_pkg["median_depth"]
        sim_image = render_pkg["render"].permute(1, 2, 0).cpu().numpy()
        sim_image = (sim_image * 255).astype(np.uint8)
        sim_seg = F.normalize(render_pkg["render_seg"], dim=0)
        sim_seg = (
            torch.einsum("nk,khw->nhw", gaussians.codebook, sim_seg)
            .argmax(0)
            .cpu()
            .numpy()
        )  # [H,W]

        # completion_index = get_completion_index(xyz_cam[visible_mask], seg_label, kf_gen.segments_info, min_area = 500, focal = tdgs_cam.focal_x)
        completion_index = get_completion_index_2d(
            sim_seg, kf_gen.segments_info, min_area=900
        )

        gaussians, _ = complete_and_align_objects(
            gaussians=gaussians,
            instance_index_list=completion_index[: config["num_completion"]],
            completion_pipeline=completion_pipeline,
            sim_image=sim_image,
            sim_seg=sim_seg,
            tdgs_cam=tdgs_cam,
            gt_depth=gt_depth,
            opt=opt,
            is_3d_fg=None,
            example=example,
            output_dir="./output",
        )

        kf_gen.increment_kf_idx()
        empty_cache()

    segments_info = {
        x: ade_id2label[str(kf_gen.segments_info[x])] for x in kf_gen.segments_info
    }
    print("Render Videos")
    with torch.no_grad():
        mid_idx = config["num_scenes"] // 2
        all_cams = []
        interp_cams = []

        for i in range(mid_idx, -1, -1):
            interp_cams.append(
                convert_pt3d_cam_to_3dgs_cam(kf_gen.cameras[i], xyz_scale=xyz_scale)
            )
        for i in range(mid_idx + 1, len(kf_gen.cameras)):
            interp_cams.append(
                convert_pt3d_cam_to_3dgs_cam(kf_gen.cameras[i], xyz_scale=xyz_scale)
            )

        for i in range(len(interp_cams) - 1):
            all_cams.extend(interp_poses(interp_cams[i], interp_cams[i + 1], 50))

        imgs = []
        segs = []
        imgs_base = []
        segs_vis = []
        for k in tqdm(range(len(all_cams))):
            render_pkg = render(
                all_cams[k], gaussians, opt, background, render_seg=True
            )
            image = render_pkg["render"]
            render_seg = F.normalize(render_pkg["render_seg"], dim=0)
            render_seg = (
                torch.einsum("nk,khw->nhw", gaussians.codebook, render_seg)
                .argmax(0)
                .cpu()
                .numpy()
            )

            segs.append(
                np.array(
                    visualize_seg(
                        render_seg, segments_info=segments_info, with_id=False
                    ).convert("RGB")
                )
            )
            rendered_image = image.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)

            seg_vis = rendered_image.astype(np.float32) * 0.65 + 0.35 * segs[-1].astype(
                np.float32
            )
            segs_vis.append(seg_vis.astype(np.uint8))
            # Image.fromarray(rendered_image).save(f'./{k}.png')
            imgs.append(rendered_image)

            render_pkg_base = render(
                all_cams[k], gaussians, opt, background, exclude_fg=True
            )
            # print(gaussians.is_fg_filter)
            image_base = render_pkg_base["render"]
            rendered_image_base = image_base.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image_base = (rendered_image_base * 255).astype(np.uint8)
            imgs_base.append(rendered_image_base)
        imgs = np.array(imgs)
        segs = np.array(segs).astype(np.uint8)
        imgs_base = np.array(imgs_base)
        segs_vis = np.array(segs_vis)
        imageio.mimwrite(f"{kf_gen.run_dir}/scene.mp4", imgs, fps=12)
        imageio.mimwrite(f"{kf_gen.run_dir}/seg.mp4", segs, fps=12)
        imageio.mimwrite(f"{kf_gen.run_dir}/baselayer.mp4", imgs_base, fps=12)
        imageio.mimwrite(f"{kf_gen.run_dir}/seg_vis.mp4", segs_vis, fps=12)

    with torch.no_grad():
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(
            kf_gen.get_camera_by_js_view_matrix(
                view_matrix_fixed, xyz_scale=xyz_scale, big_view=True
            ),
            xyz_scale=xyz_scale,
        )
        tdgs_cam.image_width = 1536
        # tdgs_cam.image_height = 1024
        render_pkg = render(
            tdgs_cam, gaussians, opt, background, render_visible=True, render_seg=True
        )
    rendered_img = render_pkg["render"]
    rendered_image = rendered_img.permute(1, 2, 0).detach().cpu().numpy()
    rendered_image = (rendered_image * 255).astype(np.uint8)

    render_seg = F.normalize(render_pkg["render_seg"], dim=0)
    render_seg = (
        torch.einsum("nk,khw->nhw", gaussians.codebook, render_seg)
        .argmax(0)
        .cpu()
        .numpy()
    )
    # rendered_image = rendered_image[..., ::-1]
    ToPILImage()(rendered_img).save(kf_gen.run_dir / "rendered_img.png")
    visualize_seg(render_seg, with_id=False).save(kf_gen.run_dir / "rendered_seg.png")

    print("Saving...")
    gaussians.save_ply_all_with_filter(
        f"examples/sky_images/{example}/finished_3dgs.ply"
    )
    torch.save(
        gaussians.visibility_filter_all,
        f"examples/sky_images/{example}/visibility_filter_all.pth",
    )
    torch.save(gaussians.codebook, f"examples/sky_images/{example}/codebook.pth")
    torch.save(
        gaussians.is_fg_filter, f"examples/sky_images/{example}/is_fg_filter.pth"
    )
    torch.save(
        gaussians.is_sky_filter, f"examples/sky_images/{example}/is_sky_filter.pth"
    )
    torch.save(
        gaussians.delete_mask_all, f"examples/sky_images/{example}/delete_mask_all.pth"
    )
    # gaussians.yield_splat_data(f'examples/sky_images/{example}/{example}_finished_3dgs.splat')

    w2c_transforms = np.stack(w2c_transforms)
    np.save(f"{kf_gen.run_dir}/w2c_transforms.npy", w2c_transforms)

    # Save to file
    with open(f"{kf_gen.run_dir}/segments_info.json", "w", encoding="utf-8") as f:
        json.dump(kf_gen.segments_info, f, ensure_ascii=False, indent=4)


def train_gaussian(
    gaussians: GaussianModel,
    scene: Scene,
    opt: GSParams,
    save_dir: Path,
    initialize_scaling=True,
    train_seg=False,
    is_fg=False,
):
    iterable_gauss = range(1, opt.iterations + 1)
    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(
        cameras=trainCameras, initialize_scaling=initialize_scaling
    )

    for iteration in tqdm(iterable_gauss):
        # Pick a random Camera
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(
            viewpoint_cam, gaussians, opt, background, render_seg=train_seg
        )

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        if train_seg:
            gt_seg = viewpoint_cam.panoptic_mask.cuda().reshape([-1]).long()
            if is_fg:
                loss_mask = gt_seg != 0
                raw_loss = 1 - F.cosine_similarity(
                    render_pkg["render_seg"].permute(1, 2, 0).flatten(end_dim=-2),
                    gaussians.codebook[gt_seg.reshape([-1])],
                    dim=-1,
                )
                raw_loss += (
                    1
                    - torch.norm(
                        render_pkg["render_seg"].permute(1, 2, 0).flatten(end_dim=-2),
                        dim=-1,
                    )
                ) ** 2  # [N]
                seg_loss = raw_loss[loss_mask].sum() / (loss_mask.sum() + 1e-5)

            else:
                seg_loss = (
                    1
                    - F.cosine_similarity(
                        render_pkg["render_seg"].permute(1, 2, 0).flatten(end_dim=-2),
                        gaussians.codebook[gt_seg.reshape([-1])],
                        dim=-1,
                    ).mean()
                )
                seg_loss += (
                    (1 - torch.norm(render_pkg["render_seg"], dim=0)) ** 2
                ).mean()
            loss += 0.1 * seg_loss
        if iteration == opt.iterations:
            gaussians.update_fg_filter(is_fg=is_fg)
            print(f"Iteration {iteration}, Loss: {loss.item()}")
            with torch.no_grad():
                render_pkg = render(
                    scene.getTrainCameras().copy()[-1],
                    gaussians,
                    opt,
                    background,
                    render_seg=True,
                    exclude_fg=(not is_fg),
                )
                image = render_pkg["render"]

                seg = F.normalize(render_pkg["render_seg"], dim=0)
                seg = (
                    torch.einsum("nk,khw->nhw", gaussians.codebook, seg)
                    .argmax(0)
                    .cpu()
                    .numpy()
                )
                seg_img = visualize_seg(seg, with_id=False)
                seg_img.save(str(save_dir) + "_seg.png")

                seg_img = visualize_seg(seg)
                seg_img.save(str(save_dir) + "_seg_vis.png")

            rendered_image = image.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            Image.fromarray(rendered_image).save(str(save_dir) + ".png")
            rendered_image = rendered_image[..., ::-1]

        loss.backward()

        if iteration == opt.iterations:
            print(f"Final loss: {loss.item()}")

        # Use variables that related to the trainable GS
        n_trainable = gaussians.get_xyz.shape[0]

        viewspace_point_tensor_grad, visibility_filter, radii = (
            viewspace_point_tensor.grad.squeeze()[:n_trainable],
            visibility_filter[:n_trainable],
            radii[:n_trainable],
        )

        with torch.no_grad():
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor_grad, visibility_filter
                )

                if (
                    iteration >= opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    max_screen_size = (
                        opt.max_screen_size
                        if iteration >= opt.prune_from_iter
                        else None
                    )
                    camera_height = 0.0003 * xyz_scale
                    scene_extent = (
                        camera_height * 2
                        if opt.scene_extent is None
                        else opt.scene_extent
                    )
                    opacity_lowest = 0.05
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opacity_lowest,
                        scene_extent,
                        max_screen_size,
                    )
                    gaussians.compute_3D_filter(cameras=trainCameras)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration % 40 == 0 and iteration != 0:
                xyz = gaussians.get_xyz
                nearest_k_idx = pytorch3d.ops.knn_points(
                    xyz.unsqueeze(0),
                    xyz.unsqueeze(0),
                    K=32,
                ).idx.squeeze()
                normed_features = F.normalize(gaussians._seg_label, dim=-1, p=2)
                smoothed_feature = normed_features[nearest_k_idx, :].mean(dim=1)
                with torch.no_grad():
                    gaussians._seg_label.data.copy_(smoothed_feature)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="Config path",
    )
    parser.add_argument("--example_config")

    args = parser.parse_args()
    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_config, example_config)

    POSTMORTEM = config["debug"]
    if POSTMORTEM:
        try:
            run(config)
        except Exception as e:
            print(e)
            import ipdb

            ipdb.post_mortem()
    else:
        run(config)
