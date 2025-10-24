import copy
import gc
import json
import os
import random
import threading
import time
import warnings
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from random import randint

import cv2
import numpy as np
import pytorch3d
import torch
import torch.nn.functional as F
from Amodal3R_align.Amodal3R.pipelines import Amodal3RImageTo3DPipeline
from arguments import GSParams
from diffusers import (
    DDIMScheduler,
    EulerDiscreteScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO
from gaussian_renderer import render
from kornia.morphology import dilation
from marigold_lcm.marigold_pipeline import (
    MarigoldNormalsPipeline,
    MarigoldPipeline,
)
from models.models import KeyframeGen
from ObjectClear.objectclear.pipelines import ObjectClearPipeline
from omegaconf import OmegaConf
from PIL import Image
from scene import GaussianModel, Scene
from syncdiffusion.syncdiffusion_model import SyncDiffusion
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from transformers import (
    OneFormerForUniversalSegmentation,
    OneFormerProcessor,
)
from util.animation_prompt_gen import AnimationPromptGen
from util.completion_utils import complete_and_align_objects
from util.LLM import TextpromptGen
from util.segment_utils import create_mask_generator_repvit
from util.simulate_multi_obj import simulate_multi
from util.simulator_prompt_gen import SimulatorPromptGen
from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from util.utils import (
    ade_id2label,
    convert_pt3d_cam_to_3dgs_cam,
    get_rotation_matrix,
    load_example_yaml,
    prepare_scheduler,
    soft_stitching,
    visualize_seg,
)
from utils.loss import l1_loss, ssim

warnings.filterwarnings("ignore")

# Initialize Flask app and SocketIO
app = Flask(__name__)
CORS(app)  # Enable CORS on the Flask app
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for SocketIO

xyz_scale = 1000
client_id = None
scene_name = None
view_matrix = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_wonder = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_delete = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_sim = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_ani = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

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
latest_frame = None
latest_viz = None
latest_seg = None
keep_rendering = True
iter_number = None
kf_gen = None
gaussians = None
opt = None
scene_dict = None
style_prompt = None
pt_gen = None
change_scene_name_by_user = False
undo = False
save = False
delete = False
do_simulation = False
do_animation = False
exclude_sky = False
simulation_prompt = None
animation_prompt = None

# Event object used to control the synchronization
start_event = threading.Event()
gen_event = threading.Event()


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


def start_server(port):
    socketio.run(app, host="0.0.0.0", port=port)


@socketio.on("connect")
def handle_connect():
    print("Client connected:", request.sid)
    global client_id
    client_id = request.sid


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected:", request.sid)
    global client_id
    client_id = None


@socketio.on("start")
def handle_start(data):
    print("Client connected:", request.sid)
    print("Received start signal.")
    start_event.set()  # Signal the main program to proceed


@socketio.on("gen")
def handle_gen(data):
    print("Received gen signal. Camera matrix: ", data)
    global view_matrix, keep_rendering
    keep_rendering = False
    view_matrix = data


@socketio.on("sim")
def handle_sim(data):
    global view_matrix_sim, keep_rendering, simulation_prompt, do_simulation
    if simulation_prompt is None:
        socketio.emit(
            "server-state",
            "Error, simulation prompt must be specified!",
            room=client_id,
        )
        return
    else:
        view_matrix_sim = data
        keep_rendering = False
        do_simulation = True


@socketio.on("ani")
def handle_ani(data):
    global view_matrix_ani, keep_rendering, animation_prompt, do_animation
    if animation_prompt is None:
        socketio.emit(
            "server-state", "Error, animation prompt must be specified!", room=client_id
        )
        return
    else:
        view_matrix_ani = data
        keep_rendering = False
        do_animation = True


@socketio.on("render-pose")
def handle_render_pose(data):
    global view_matrix_wonder, keep_rendering
    view_matrix_wonder = data


@socketio.on("scene-prompt")
def handle_new_prompt(data):
    assert isinstance(data, str)
    print("Received new scene prompt: " + data)
    global scene_name, change_scene_name_by_user
    scene_name = data
    change_scene_name_by_user = True


@socketio.on("sim-prompt")
def handle_sim_prompt(data):
    assert isinstance(data, str)
    print("Received new simulation prompt: " + data)
    global simulation_prompt
    simulation_prompt = data


@socketio.on("ani-prompt")
def handle_ani_prompt(data):
    assert isinstance(data, str)
    print("Received new animation prompt: " + data)
    global animation_prompt
    animation_prompt = data


@socketio.on("undo")
def handle_undo():
    print("Received undo signal.")
    global undo
    undo = True


@socketio.on("save")
def handle_save():
    print("Received save signal.")
    global save
    save = True


@socketio.on("delete")
def handle_delete(data):
    print("Received delete signal.")
    global delete, view_matrix_delete
    delete = True
    view_matrix_delete = data


@socketio.on("fill_hole")
def handle_fill_hole():
    print("Received fill hole signal.")
    global exclude_sky
    exclude_sky = True


@torch.no_grad()
def compute_3d_bbox(pos: torch.Tensor, seg_label: torch.Tensor, segments_info: dict):
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

    # Validate input shapes
    assert pos.ndim == 2 and pos.shape[1] == 3, "pos must be of shape [N,3]"
    assert seg_label.ndim == 1 and seg_label.shape[0] == pos.shape[0], (
        "seg_label must be of shape [N] to match pos"
    )
    seg_label = seg_label.cpu().numpy()

    boxes = []
    semantic_labels = []

    # Iterate over each segment label defined in segments_info
    for seg_id, semantic_label in segments_info.items():
        # Find points belonging to this segment
        mask = seg_label == int(seg_id)

        # print(seg_id)
        # Skip if no points for this segment
        if not mask.any():
            continue

        points = pos[mask]
        # Move to CPU and NumPy for percentile computation (if needed)
        points_np = points.cpu().numpy()

        # Compute 10th and 90th percentiles for each dimension to remove outliers
        lower = np.percentile(points_np, 1, axis=0)
        upper = np.percentile(points_np, 99, axis=0)
        # print(lower, upper)
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


def render_current_scene():
    global \
        latest_frame, \
        client_id, \
        iter_number, \
        latest_viz, \
        kf_gen, \
        gaussians, \
        opt, \
        background, \
        view_matrix_wonder, \
        save, \
        latest_seg
    while True:
        time.sleep(0.05)
        try:
            # print(kf_gen.get_camera_by_js_view_matrix(view_matrix_wonder, xyz_scale=xyz_scale).R,kf_gen.get_camera_by_js_view_matrix(view_matrix_wonder, xyz_scale=xyz_scale).T)
            with torch.no_grad():
                tdgs_cam = convert_pt3d_cam_to_3dgs_cam(
                    kf_gen.get_camera_by_js_view_matrix(
                        view_matrix_wonder, xyz_scale=xyz_scale
                    ),
                    xyz_scale=xyz_scale,
                )
                render_pkg = render(
                    tdgs_cam,
                    gaussians,
                    opt,
                    background,
                    render_visible=True,
                    render_seg=True,
                )
            rendered_img = render_pkg["render"]
            rendered_image = rendered_img.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            rendered_image = rendered_image[..., ::-1]
            latest_frame = rendered_image

            render_seg = F.normalize(render_pkg["render_seg"], dim=0)
            render_seg = (
                torch.einsum("nk,khw->nhw", gaussians.codebook, render_seg)
                .argmax(0)
                .cpu()
                .numpy()
            )

            render_seg = np.array(
                visualize_seg(render_seg, with_id=False).convert("RGB")
            )[..., ::-1]
            latest_seg = render_seg

            with torch.no_grad():
                tdgs_cam = convert_pt3d_cam_to_3dgs_cam(
                    kf_gen.get_camera_by_js_view_matrix(
                        view_matrix_fixed, xyz_scale=xyz_scale, big_view=True
                    ),
                    xyz_scale=xyz_scale,
                )
                tdgs_cam.image_width = 1536
                render_pkg = render(
                    tdgs_cam, gaussians, opt, background, render_visible=True
                )
            rendered_img = render_pkg["render"]
            rendered_image = rendered_img.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            rendered_image = rendered_image[..., ::-1]
            latest_viz = rendered_image
            if save:
                ToPILImage()(rendered_img).save(kf_gen.run_dir / "rendered_img.png")
        except Exception:
            # print(f"Render error: {e}")
            pass

        if latest_frame is not None and client_id is not None:
            image_bytes = cv2.imencode(".jpg", latest_frame)[1].tobytes()
            socketio.emit("frame", image_bytes, room=client_id)
            socketio.emit("iter-number", f"Iter: {iter_number}", room=client_id)
        if latest_viz is not None and client_id is not None:
            image_bytes = cv2.imencode(".jpg", latest_viz)[1].tobytes()
            socketio.emit("viz", image_bytes, room=client_id)
        if latest_seg is not None and client_id is not None:
            image_bytes = cv2.imencode(".jpg", latest_seg)[1].tobytes()
            socketio.emit("seg", image_bytes, room=client_id)


def run(config):
    global view_matrix, scene_name, latest_frame, keep_rendering, kf_gen, latest_viz, gaussians, opt, background, scene_dict, style_prompt, pt_gen, change_scene_name_by_user, undo, save, delete, exclude_sky, view_matrix_delete, simulation_prompt, do_simulation, animation_prompt, do_animation, latest_seg

    ###### ------------------ Load modules ------------------ ######

    seeding(config["seed"])
    example = config["example_name"]

    config["use_gpt"] = True

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

    completion_pipeline = Amodal3RImageTo3DPipeline.from_pretrained(
        "Sm0kyWu/Amodal3R",
        ss_path=config["ss_path"],
    )
    completion_pipeline.cuda()

    depth_model = MarigoldPipeline.from_pretrained(
        "prs-eth/marigold-depth-v1-1", torch_dtype=torch.bfloat16
    ).to(config["device"])
    depth_model.scheduler = EulerDiscreteScheduler.from_config(
        depth_model.scheduler.config
    )
    depth_model.scheduler = prepare_scheduler(depth_model.scheduler)

    normal_estimator = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-v1-1", torch_dtype=torch.bfloat16
    ).to(config["device"])

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
        control_text,
    ) = (
        yaml_data["content_prompt"],
        yaml_data["style_prompt"],
        yaml_data["negative_prompt"],
        yaml_data.get("background", None),
        yaml_data.get("control_text", None),
    )
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

    pt_gen = TextpromptGen(config["runs_dir"], isinstance(control_text, list))

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
    socketio.emit("scene-prompt", scene_name, room=client_id)

    kf_gen.increment_kf_idx()
    ###### ------------------ Main loop ------------------ ######

    if not os.path.exists(f"examples/sky_images/{example}/finished_3dgs_sky_tanh.ply"):
        socketio.emit("server-state", "Generating sky 3DGS...", room=client_id)
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
        gaussians.is_fg_filter = torch.load(
            f"examples/sky_images/{example}/is_fg_filter.pth"
        ).to("cuda")
    opt = GSParams()

    ### First scene 3DGS
    if config["gen_layer"]:
        traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(
            xyz_scale=xyz_scale
        )
        gaussians = GaussianModel(
            sh_degree=0, previous_gaussian=gaussians, codebook=kf_gen.codebook
        )
        scene = Scene(traindata_layer, gaussians, opt)
        save_dir = f"{kf_gen.run_dir}/gaussian_scene_layer{0:02d}"

        train_gaussian(
            gaussians, scene, opt, save_dir, train_seg=True
        )  # Base layer training
        # gaussians.update_fg_filter(is_fg = False)

    else:
        traindata = kf_gen.convert_to_3dgs_traindata_latest(
            xyz_scale=xyz_scale, use_no_loss_mask=False
        )

    gaussians = GaussianModel(
        sh_degree=0, previous_gaussian=gaussians, codebook=kf_gen.codebook
    )
    scene = Scene(traindata, gaussians, opt)
    # dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
    i = 0
    save_dir = f"{kf_gen.run_dir}/gaussian_scene{i:02d}"

    train_gaussian(gaussians, scene, opt, save_dir, train_seg=True, is_fg=True)

    # import pdb
    # pdb.set_trace()

    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(
        kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale
    )
    gaussians.set_inscreen_points_to_visible(tdgs_cam)

    def llm_prompt_generation(event):
        global scene_dict, style_prompt, pt_gen, change_scene_name_by_user, scene_name
        while True:
            event.wait()
            print("-- start llm...")
            scene_dict = pt_gen.wonder_next_scene(
                scene_name=scene_name,
                entities=scene_dict["entities"],
                style=style_prompt,
                background=scene_dict["background"],
                change_scene_name_by_user=change_scene_name_by_user,
            )
            change_scene_name_by_user = False
            print("-- llm done.")
            event.clear()

    if config["use_gpt"]:
        llm_event = threading.Event()
        llm_thread = threading.Thread(target=llm_prompt_generation, args=(llm_event,))
        llm_thread.daemon = True
        llm_thread.start()

    gaussians_tmp = copy.deepcopy(gaussians)

    is_3d_fg = torch.zeros(
        gaussians.get_xyz_all.shape[0], dtype=torch.int32, device="cuda"
    )
    while True:
        # Update scene information
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
        i += 1

        socketio.emit("scene-prompt", scene_name, room=client_id)
        print("Waiting for scene gen signal...")
        socketio.emit(
            "server-state", "Waiting to generate new scenes...", room=client_id
        )

        while keep_rendering:
            time.sleep(0.05)
            if delete:
                print("Deleting...")
                current_pt3d_cam_delete = kf_gen.get_camera_by_js_view_matrix(
                    view_matrix_delete, xyz_scale=xyz_scale
                )
                tdgs_cam_delete = convert_pt3d_cam_to_3dgs_cam(
                    current_pt3d_cam_delete, xyz_scale=xyz_scale
                )
                gaussians.delete_points(tdgs_cam_delete)
                delete = False
            if save:
                print("Saving...")
                gaussians.save_ply_all_with_filter(
                    f"examples/sky_images/{example}/finished_3dgs.ply"
                )
                torch.save(
                    gaussians.visibility_filter_all,
                    f"examples/sky_images/{example}/visibility_filter_all.pth",
                )
                torch.save(
                    gaussians.codebook, f"examples/sky_images/{example}/codebook.pth"
                )
                torch.save(
                    gaussians.is_fg_filter,
                    f"examples/sky_images/{example}/is_fg_filter.pth",
                )
                torch.save(
                    gaussians.is_sky_filter,
                    f"examples/sky_images/{example}/is_sky_filter.pth",
                )
                torch.save(
                    gaussians.delete_mask_all,
                    f"examples/sky_images/{example}/delete_mask_all.pth",
                )
                # gaussians.yield_splat_data(f'examples/sky_images/{example}/{example}_finished_3dgs.splat')
                save = False

                # Save to file
                with open(
                    f"{kf_gen.run_dir}/segments_info.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(kf_gen.segments_info, f, ensure_ascii=False, indent=4)

        if undo:
            print("Undoing...")
            gaussians = copy.deepcopy(gaussians_tmp)
            undo = False
        else:
            print("Not undo...")
            gaussians_tmp = copy.deepcopy(gaussians)

        if do_simulation:
            socketio.emit("server-state", "Running 3D completion...", room=client_id)

            current_pt3d_cam = kf_gen.get_camera_by_js_view_matrix(
                view_matrix_sim, xyz_scale=xyz_scale
            )
            tdgs_cam = convert_pt3d_cam_to_3dgs_cam(
                current_pt3d_cam, xyz_scale=xyz_scale
            )

            R = torch.tensor(
                tdgs_cam.R, device=torch.device("cuda"), dtype=torch.float32
            )
            T = torch.tensor(
                tdgs_cam.T, device=torch.device("cuda"), dtype=torch.float32
            )

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
            seg_label = (
                gaussians.get_seg_all[visible_mask] @ gaussians.codebook.T
            ).argmax(-1)  # 【N，C】

            # compute 3d bbox for each seg label, with format [[x_min,x_max,y_min,y_max,z_min,z_max]]
            bbox_3d, semantic_label = compute_3d_bbox(
                xyz_cam[visible_mask], seg_label, kf_gen.segments_info
            )
            results = SimulatorPromptGen().simulate(
                bbox_3d, semantic_label, simulation_prompt
            )

            results = results["objects"]
            print(results)

            # Store material parameters and forces for all objects
            all_material_params = []
            all_forces = []
            all_index = []

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

            # Record object information
            for result in results:
                selected_obj = int(result["instance_id"])
                material_params = result["material_params"]
                force = torch.tensor(result["force"]).float().cuda()
                force = R @ force  # convert to world coordinate
                force = force.cpu().numpy().tolist()

                # Record current object information
                all_material_params.append(material_params)
                all_forces.append(force)
                all_index.append(selected_obj)

            # Use new completion and alignment function
            merged_gaussians, is_3d_fg = complete_and_align_objects(
                gaussians=gaussians,
                instance_index_list=all_index,
                completion_pipeline=completion_pipeline,
                sim_image=sim_image,
                sim_seg=sim_seg,
                tdgs_cam=tdgs_cam,
                gt_depth=gt_depth,
                opt=opt,
                is_3d_fg=is_3d_fg,
                example=example,
                output_dir="./output",
            )

            gaussians = merged_gaussians

            means3D_fg_list = []
            covariance_fg_list = []

            # Extract point cloud for each object
            for obj_idx in all_index:
                obj_mask = is_3d_fg == obj_idx
                if obj_mask.sum() > 0:
                    points = gaussians.get_xyz_all[obj_mask].clone()
                    means3D_fg_list.append(points)
                    covariance_fg_list.append(
                        gaussians.get_covariance_all()[obj_mask].clone()
                    )

            # Get background point cloud
            bg_mask = is_3d_fg == 0
            means3D_bg = gaussians.get_xyz_all[bg_mask].clone()
            means3D_all = gaussians.get_xyz_all.clone()

            covariance_bg = gaussians.get_covariance_all()[bg_mask].clone()
            covariance_all = gaussians.get_covariance_all()

            socketio.emit("server-state", "Running MPM simulation...", room=client_id)

            with torch.no_grad():
                # Run physics simulation
                if means3D_fg_list:
                    results = simulate_multi(
                        means3D_fg_list,
                        covariance_fg_list,
                        means3D_bg,
                        covariance_bg,
                        fg_material_configs=all_material_params,
                        force=all_forces,
                        use_bg=False,
                    )
                    max_frames = max([len(obj[0]) for obj in results]) if results else 0

                    for frame_idx in range(max_frames):
                        for obj_idx, (obj_means, obj_conv) in enumerate(results):
                            obj_mask = is_3d_fg == all_index[obj_idx]
                            if obj_mask.sum() > 0:
                                means3D_all[obj_mask] = obj_means[frame_idx].cuda()
                                covariance_all[obj_mask] = obj_conv[frame_idx].cuda()

                        gaussians.update_parameters(means3D_all)
                        time.sleep(0.05)

            keep_rendering = True
            do_simulation = False
            continue

        if do_animation:
            socketio.emit("server-state", "Running 3D completion...", room=client_id)

            current_pt3d_cam = kf_gen.get_camera_by_js_view_matrix(
                view_matrix_ani, xyz_scale=xyz_scale
            )
            tdgs_cam = convert_pt3d_cam_to_3dgs_cam(
                current_pt3d_cam, xyz_scale=xyz_scale
            )

            R = torch.tensor(
                tdgs_cam.R, device=torch.device("cuda"), dtype=torch.float32
            )
            T = torch.tensor(
                tdgs_cam.T, device=torch.device("cuda"), dtype=torch.float32
            )

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
            seg_label = (
                gaussians.get_seg_all[visible_mask] @ gaussians.codebook.T
            ).argmax(-1)  # 【N，C】

            # compute 3d bbox for each seg label, with format [[x_min,x_max,y_min,y_max,z_min,z_max]]
            bbox_3d, semantic_label = compute_3d_bbox(
                xyz_cam[visible_mask], seg_label, kf_gen.segments_info
            )
            results = AnimationPromptGen().animate(
                bbox_3d, semantic_label, animation_prompt
            )

            results = results["objects"]
            print(results)

            # Store animation parameters for all objects
            all_translation = []
            all_rotation = []
            all_index = []

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

            # Record object information
            for result in results:
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
                all_index.append(selected_obj)

            # Use new completion and alignment function
            merged_gaussians, is_3d_fg = complete_and_align_objects(
                gaussians=gaussians,
                instance_index_list=all_index,
                completion_pipeline=completion_pipeline,
                sim_image=sim_image,
                sim_seg=sim_seg,
                tdgs_cam=tdgs_cam,
                gt_depth=gt_depth,
                opt=opt,
                is_3d_fg=is_3d_fg,
                example=example,
                output_dir="./output",
                similarity_threshold=-10,  # disable rollback in such case
            )

            gaussians = merged_gaussians

            socketio.emit(
                "server-state", "Running Animation Rendering...", room=client_id
            )

            # Save original point cloud and covariance
            means3D_all = gaussians.get_xyz_all.clone()
            covariance_all = gaussians.get_covariance_all().clone()

            with torch.no_grad():
                # Get total frames
                max_frames = (
                    100  # animation_prompt_gen.py generates 100 frames of animation
                )

                for frame_idx in range(max_frames):
                    # Copy original point cloud as starting point
                    frame_means3D = means3D_all.clone()

                    # frame_covariance = covariance_all.clone()

                    # Update point clouds for all objects
                    for i, obj_idx in enumerate(all_index):
                        obj_mask = is_3d_fg == obj_idx
                        if obj_mask.sum() > 0:
                            # Get translation and rotation for current frame
                            obj_trans = all_translation[i][frame_idx]
                            obj_rot = all_rotation[i][frame_idx]

                            # Convert rotation angles to radians
                            obj_rot_rad = [angle * np.pi / 180 for angle in obj_rot]

                            # Get original point cloud of object
                            points = means3D_all[obj_mask].clone()
                            covs = covariance_all[obj_mask].clone()

                            # Calculate object center
                            center = points.mean(dim=0)

                            # Move point cloud to origin
                            centered_points = points - center

                            # Apply rotation
                            rotation_matrix = get_rotation_matrix(
                                torch.tensor(obj_rot_rad).float().cuda()
                            )
                            rotated_points = centered_points @ rotation_matrix.T

                            # Move back to original position and apply translation
                            translated_points = (
                                rotated_points
                                + center
                                + torch.tensor(obj_trans).float().cuda()
                            )

                            # Update object point cloud
                            frame_means3D[obj_mask] = translated_points
                            # Covariance matrix remains unchanged

                    # Update point cloud for current frame
                    gaussians.update_parameters(frame_means3D)
                    time.sleep(0.05)

            keep_rendering = True
            do_animation = False
            continue

        socketio.emit("server-state", "Generating new scene...", room=client_id)

        # LLM prompt generation
        if config["use_gpt"]:
            llm_event.set()

        if config["use_gpt"]:
            scene_dict = pt_gen.wonder_next_scene(
                scene_name=scene_name,
                entities=scene_dict["entities"],
                style=style_prompt,
                background=scene_dict["background"],
                change_scene_name_by_user=change_scene_name_by_user,
            )
            change_scene_name_by_user = False
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

        kf_gen.set_kf_param(
            inpainting_resolution=config["inpainting_resolution_gen"],
            inpainting_prompt=inpainting_prompt,
            adaptive_negative_prompt=adaptive_negative_prompt,
        )

        current_pt3d_cam = kf_gen.get_camera_by_js_view_matrix(
            view_matrix, xyz_scale=xyz_scale
        )
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam, xyz_scale=xyz_scale)
        kf_gen.set_current_camera(current_pt3d_cam, archive_camera=True)

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

            # render_seg = F.normalize(render_pkg['render_seg'], dim = 0)
            # render_seg  = torch.einsum('nk,khw->nhw', gaussians.codebook, render_seg).argmax(0)
            # kf_gen.prev_panoptic_mask_fg = render_seg  #[H,W]

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
            # latest_viz = viz
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

            # render_seg = F.normalize(render_pkg['render_seg'], dim = 0)
            # render_seg  = torch.einsum('nk,khw->nhw', gaussians.codebook, render_seg).argmax(0)
            # kf_gen.prev_panoptic_mask_fg = render_seg  #[H,W]

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
            foreground_cols = (
                torch.sum(fg_mask_0p5_nosky == 1, dim=1) > 150
            )  # [1, 512] Empty regions where sky_layer images are rendered, use full_renderer for these regions
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
            # latest_viz = viz
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
        # image = outpaint_condition_image[0].permute(1,2,0).cpu().numpy()
        # image = Image.fromarray((image*255).astype(np.uint8))
        # image.save(f'{kf_gen.run_dir}/images/layer/{kf_gen.kf_idx:02d}_outpaint_image.png')

        # image = outpaint_mask.squeeze().cpu().numpy()
        # image = Image.fromarray((image*255).astype(np.uint8))
        # image.save(f'{kf_gen.run_dir}/images/layer/{kf_gen.kf_idx:02d}_outpaint_mask.png')

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

        socketio.emit("server-state", "Getting depth estimation...", room=client_id)
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
            gaussians = GaussianModel(
                sh_degree=0, previous_gaussian=gaussians, codebook=kf_gen.codebook
            )
            scene = Scene(traindata_layer, gaussians, opt)
            save_dir = f"{kf_gen.run_dir}/gaussian_scene_layer{i + 1:02d}"

            train_gaussian(
                gaussians, scene, opt, save_dir, train_seg=True
            )  # Base layer training
            # gaussians.update_fg_filter(is_fg = False)
            is_3d_fg_cur = torch.zeros(
                gaussians.get_xyz.shape[0], dtype=torch.int32, device="cuda"
            )
            is_3d_fg = torch.cat([is_3d_fg_cur, is_3d_fg], dim=0)

        else:
            traindata = kf_gen.convert_to_3dgs_traindata_latest(
                xyz_scale=xyz_scale, use_no_loss_mask=False
            )

        if traindata["pcd_points"].shape[-1] == 0:
            gaussians.set_inscreen_points_to_visible(tdgs_cam)

            kf_gen.increment_kf_idx()
            keep_rendering = True
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
        image_tmp = (
            render_pkg_nosky["render"] * mask_using_nosky_render
            + render_pkg["render"] * mask_using_full_render
        )

        gaussians = GaussianModel(
            sh_degree=0, previous_gaussian=gaussians, codebook=kf_gen.codebook
        )
        scene = Scene(traindata, gaussians, opt)
        save_dir = f"{kf_gen.run_dir}/gaussian_scene{i + 1:02d}"
        train_gaussian(gaussians, scene, opt, save_dir, train_seg=True, is_fg=True)
        # gaussians.update_fg_filter(is_fg = True)
        is_3d_fg_cur = torch.zeros(
            gaussians.get_xyz.shape[0], dtype=torch.int32, device="cuda"
        )
        is_3d_fg = torch.cat([is_3d_fg_cur, is_3d_fg], dim=0)

        gaussians.set_inscreen_points_to_visible(tdgs_cam)

        kf_gen.increment_kf_idx()
        keep_rendering = True
        empty_cache()


def train_gaussian(
    gaussians: GaussianModel,
    scene: Scene,
    opt: GSParams,
    save_dir: Path,
    initialize_scaling=True,
    train_seg=False,
    is_fg=False,
):
    global latest_frame, iter_number, view_matrix, latest_viz, client_id, latest_seg

    socketio.emit(
        "server-state",
        f"Training Gaussian {'foreground' if is_fg else 'sky/background'} for {opt.iterations} iterations...",
        room=client_id,
    )

    # if render_seg:
    #     opt.iterations = 2000
    iterable_gauss = range(1, opt.iterations + 1)
    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(
        cameras=trainCameras, initialize_scaling=initialize_scaling
    )

    total = 0
    for iteration in tqdm(iterable_gauss):
        # Pick a random Camera
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # import pdb; pdb.set_trace()
        # Render
        # s = time.time()
        render_pkg = render(
            viewpoint_cam, gaussians, opt, background, render_seg=train_seg
        )
        # total+=time.time()-s
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
            # if iteration % 5 == 0 or iteration == 1:
            time.sleep(0.1)
            gaussians.update_fg_filter(is_fg=is_fg)
            print(f"Iteration {iteration}, Loss: {loss.item()}")
            socketio.emit(
                "server-state",
                f"Gaussian training complete. Final loss: {loss.item():.6f}",
                room=client_id,
            )

            with torch.no_grad():
                # tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_by_js_view_matrix(view_matrix, xyz_scale=xyz_scale), xyz_scale=xyz_scale)
                render_pkg = render(
                    scene.getTrainCameras().copy()[-1],
                    gaussians,
                    opt,
                    background,
                    render_seg=True,
                    exclude_fg=(not is_fg),
                )
                image = render_pkg["render"]
                # if train_seg:
                # import pdb
                # pdb.set_trace()
                # print("a"render_pkg['render_seg'][1:].max())
                seg = F.normalize(render_pkg["render_seg"], dim=0)
                seg = (
                    torch.einsum("nk,khw->nhw", gaussians.codebook, seg)
                    .argmax(0)
                    .cpu()
                    .numpy()
                )
                seg_img = visualize_seg(seg, with_id=False)
                seg_img.save(str(save_dir) + "_seg.png")

                # seg_img = visualize_seg(seg)
                # seg_img.save(str(save_dir)+'_seg_vis.png')

                # rendered_normal = render_pkg['render_normal']
                # rendered_normal_map = rendered_normal/2-0.5
            rendered_image = image.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            Image.fromarray(rendered_image).save(str(save_dir) + ".png")
            rendered_image = rendered_image[..., ::-1]

            latest_frame = rendered_image
            iter_number = iteration
            seg_img = np.array(seg_img)[..., ::-1]
            latest_seg = seg_img

        loss.backward()
        # if render_seg:
        #     import pdb
        #     pdb.set_trace()
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

                # if (iteration % opt.opacity_reset_interval == 0
                #     or (opt.white_background and iteration == opt.densify_from_iter)
                # ):
                #     gaussians.reset_opacity()

            # if iteration % 100 == 0 and iteration > opt.densify_until_iter:
            #     if iteration < opt.iterations - 100:
            #         # don't update in the end of training
            #         gaussians.compute_3D_filter(cameras=trainCameras)

            # Optimizer step
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

    # print("Forward cost:",total)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="Config path",
    )
    parser.add_argument("--example_config")

    parser.add_argument(
        "--port",
        default=7778,
        type=int,
        help="Port for the server",
    )

    args = parser.parse_args()
    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_config, example_config)

    # Start the server on a separate thread
    server_thread = threading.Thread(target=start_server, args=(args.port,))
    server_thread.start()

    # Start the rendering loop on the main thread
    render_thread = threading.Thread(target=render_current_scene)
    render_thread.start()

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
