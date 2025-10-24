import gradio as gr
import spaces

import os

import shutil
os.environ['SPCONV_ALGO'] = 'native'
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from Amodal3R.pipelines import Amodal3RImageTo3DPipeline
from Amodal3R.representations import Gaussian, MeshExtractResult
from Amodal3R.utils import render_utils, postprocessing_utils
from segment_anything import sam_model_registry, SamPredictor
from huggingface_hub import hf_hub_download
import cv2


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
      
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)

def change_message():
    return "Please wait for a few seconds after uploading the image."

def reset_image(predictor, img):
    img = np.array(img)
    predictor.set_image(img)
    original_img = img.copy()
    return predictor, original_img, "The models are ready.", [], [], [], original_img

def button_clickable(selected_points):
    if len(selected_points) > 0:
        return gr.Button.update(interactive=True)
    else:
        return gr.Button.update(interactive=False)

def run_sam(img, predictor, selected_points):
    if len(selected_points) == 0:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    input_points = [p for p in selected_points]
    input_labels = [1 for _ in range(len(selected_points))]
    masks, _, _ = predictor.predict(
        point_coords=np.array(input_points),
        point_labels=np.array(input_labels),
        multimask_output=False,
    )
    best_mask = masks[0].astype(np.uint8)
    # dilate
    if len(selected_points) > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        best_mask = cv2.dilate(best_mask, kernel, iterations=1)
        best_mask = cv2.erode(best_mask, kernel, iterations=1)
    return best_mask


@spaces.GPU
def image_to_3d(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    erode_kernel_size: int,
    req: gr.Request,
) -> Tuple[dict, str]:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    np.save('./image.npy', image)
    np.save('./mask.npy', mask)
    outputs = pipeline.run_multi_image(
        [image],
        [mask],
        seed=seed,
        formats=["gaussian", "mesh"],
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
        mode="stochastic",
        erode_kernel_size=erode_kernel_size,
    )
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120, bg_color=(1,1,1))['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return state, video_path


@spaces.GPU(duration=90)
def extract_glb(
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    req: gr.Request,
) -> tuple:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path


@spaces.GPU
def extract_gaussian(state: dict, req: gr.Request) -> tuple:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }
    
    
def unpack_state(state: dict) -> tuple:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    
    return gs, mesh

def get_sam_predictor():
    sam_checkpoint = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def draw_points_on_image(image, point):
    image_with_points = image.copy()
    x, y = point
    color = (255, 0, 0)
    cv2.circle(image_with_points, (int(x), int(y)), radius=10, color=color, thickness=-1)
    return image_with_points


def see_point(image, x, y):
    updated_image = draw_points_on_image(image, [x,y])
    return updated_image

def add_point(x, y, visible_points):
    if [x, y] not in visible_points:
        visible_points.append([x, y])
    return visible_points

def delete_point(visible_points):
    visible_points.pop()
    return visible_points


def clear_all_points(image):
    updated_image = image.copy()
    return updated_image

def see_visible_points(image, visible_points):
    updated_image = image.copy()
    for p in visible_points:
        cv2.circle(updated_image, (int(p[0]), int(p[1])), radius=10, color=(255, 0, 0), thickness=-1)
    return updated_image

def see_occlusion_points(image, occlusion_points):
    updated_image = image.copy()
    for p in occlusion_points:
        cv2.circle(updated_image, (int(p[0]), int(p[1])), radius=10, color=(0, 255, 0), thickness=-1)
    return updated_image

def update_all_points(points):
    text = f"Points: {points}"
    dropdown_choices = [f"({p[0]}, {p[1]})" for p in points]
    return text, gr.Dropdown(show_label=False, choices=dropdown_choices, value=None, interactive=True)

def delete_selected(image, visible_points, occlusion_points, occlusion_mask_list, selected_value, point_type):
    if point_type == "visibility":
        try:
            selected_index = [f"({p[0]}, {p[1]})" for p in visible_points].index(selected_value)
        except ValueError:
            selected_index = None
        if selected_index is not None and 0 <= selected_index < len(visible_points):
            visible_points.pop(selected_index)
    else:
        try:
            selected_index = [f"({p[0]}, {p[1]})" for p in occlusion_points].index(selected_value)
        except ValueError:
            selected_index = None
        if selected_index is not None and 0 <= selected_index < len(occlusion_points):
            occlusion_points.pop(selected_index)
            occlusion_mask_list.pop(selected_index)
    updated_image = image.copy()
    updated_image = see_visible_points(updated_image, visible_points)
    updated_image = see_occlusion_points(updated_image, occlusion_points)
    if point_type == "visibility":
        updated_text, dropdown = update_all_points(visible_points)
    else:
        updated_text, dropdown = update_all_points(occlusion_points)
    return updated_image, visible_points, occlusion_points, updated_text, dropdown

def add_current_mask(visibility_mask, visibilty_mask_list, point_type): 
    if point_type == "visibility":
        if len(visibilty_mask_list) > 0:
            if np.array_equal(visibility_mask, visibilty_mask_list[-1]):
                return visibilty_mask_list
        visibilty_mask_list.append(visibility_mask)
        return visibilty_mask_list
    else: # the occlusion mask will be automatically added, so do nothing here
        return visibilty_mask_list

def apply_mask_overlay(image, mask, color=(255, 0, 0)):
    img_arr = image
    overlay = img_arr.copy()
    gray_color = np.array([200, 200, 200], dtype=np.uint8)
    non_mask = mask == 0
    overlay[non_mask] = (0.5 * overlay[non_mask] + 0.5 * gray_color).astype(np.uint8)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay

def vis_mask(image, mask_list):
    updated_image = image.copy()
    combined_mask = np.zeros_like(updated_image[:, :, 0])
    for mask in mask_list:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    updated_image = apply_mask_overlay(updated_image, combined_mask)
    return updated_image

def segment_and_overlay(image, points, sam_predictor, mask_list, point_type):
    if point_type == "visibility":
        visible_mask = run_sam(image, sam_predictor, points)
        for mask in mask_list:
            visible_mask = cv2.bitwise_or(visible_mask, mask)
        overlaid = apply_mask_overlay(image, visible_mask * 255)
        return overlaid, visible_mask, mask_list
    else:
        combined_occlusion_mask = np.zeros_like(image[:, :, 0])
        mask_list = []
        if len(points) != 0:
            for point in points:
                mask = run_sam(image, sam_predictor, [point])
                mask_list.append(mask)
                combined_occlusion_mask = cv2.bitwise_or(combined_occlusion_mask, mask)
        overlaid = apply_mask_overlay(image, combined_occlusion_mask * 255, color=(0, 255, 0))
        return overlaid, combined_occlusion_mask, mask_list

def delete_mask(visibility_mask_list, occlusion_mask_list, occlusion_points_state, point_type):
    if point_type == "visibility":
        if len(visibility_mask_list) > 0:
            visibility_mask_list.pop()
    else:
        if len(occlusion_mask_list) > 0:
            occlusion_mask_list.pop()
            occlusion_points_state.pop()
    return visibility_mask_list, occlusion_mask_list, occlusion_points_state

def check_combined_mask(image, visibility_mask, visibility_mask_list, occlusion_mask_list, scale=0.68):
    if visibility_mask.sum() == 0:
        return np.zeros_like(image), np.zeros_like(image[:, :, 0])
    updated_image = image.copy()
    combined_mask = np.zeros_like(updated_image[:, :, 0])
    occluded_mask = np.zeros_like(updated_image[:, :, 0])
    binary_visibility_masks = [(m > 0).astype(np.uint8) for m in visibility_mask_list]
    combined_mask = np.zeros_like(binary_visibility_masks[0]) if binary_visibility_masks else (visibility_mask > 0).astype(np.uint8)
    for m in binary_visibility_masks:
        combined_mask = cv2.bitwise_or(combined_mask, m)

    if len(binary_visibility_masks) > 1:
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    binary_occlusion_masks = [(m > 0).astype(np.uint8) for m in occlusion_mask_list]
    occluded_mask = np.zeros_like(binary_occlusion_masks[0]) if binary_occlusion_masks else np.zeros_like(combined_mask)
    for m in binary_occlusion_masks:
        occluded_mask = cv2.bitwise_or(occluded_mask, m)

    kernel_small = np.ones((3, 3), np.uint8)
    if len(binary_occlusion_masks) > 0:
        dilated = cv2.dilate(combined_mask, kernel_small, iterations=1)
        boundary_mask = dilated - combined_mask
        occluded_mask = cv2.bitwise_or(occluded_mask, boundary_mask)
        occluded_mask = (occluded_mask > 0).astype(np.uint8)
        occluded_mask = cv2.dilate(occluded_mask, kernel_small, iterations=1)
        occluded_mask = (occluded_mask > 0).astype(np.uint8)
    else:
        occluded_mask = 1 - combined_mask

    combined_mask[occluded_mask == 1] = 0

    occluded_mask = (1-occluded_mask) * 255

    masked_img = updated_image * combined_mask[:, :, None]
    occluded_mask[combined_mask == 1] = 127

    x, y, w, h = cv2.boundingRect(combined_mask.astype(np.uint8))

    ori_h, ori_w = masked_img.shape[:2]
    target_size = 512
    scale_factor = target_size / max(w, h)
    final_scale = scale_factor * scale
    new_w = int(round(ori_w * final_scale))
    new_h = int(round(ori_h * final_scale))
    
    resized_occluded_mask = cv2.resize(occluded_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized_img = cv2.resize(masked_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    final_img = np.zeros((target_size, target_size, 3), dtype=updated_image.dtype)
    final_occluded_mask = np.ones((target_size, target_size), dtype=np.uint8) * 255

    new_x = int(round(x * final_scale))
    new_y = int(round(y * final_scale))
    new_w_box = int(round(w * final_scale))
    new_h_box = int(round(h * final_scale))

    new_cx = new_x + new_w_box // 2
    new_cy = new_y + new_h_box // 2

    final_cx, final_cy = target_size // 2, target_size // 2
    x_offset = final_cx - new_cx
    y_offset = final_cy - new_cy

    final_x_start = max(0, x_offset)
    final_y_start = max(0, y_offset)
    final_x_end = min(target_size, x_offset + new_w)
    final_y_end = min(target_size, y_offset + new_h)

    img_x_start = max(0, -x_offset)
    img_y_start = max(0, -y_offset)
    img_x_end = min(new_w, target_size - x_offset)
    img_y_end = min(new_h, target_size - y_offset)

    final_img[final_y_start:final_y_end, final_x_start:final_x_end] = resized_img[img_y_start:img_y_end, img_x_start:img_x_end]
    final_occluded_mask[final_y_start:final_y_end, final_x_start:final_x_end] = resized_occluded_mask[img_y_start:img_y_end, img_x_start:img_x_end]

    return final_img, final_occluded_mask


def get_point(img, point_type, visible_points_state, occlusion_points_state, evt: gr.SelectData):
    updated_img = np.array(img).copy()
    if point_type == "visibility":
        visible_points_state = add_point(evt.index[0], evt.index[1], visible_points_state)
    else:
        occlusion_points_state = add_point(evt.index[0], evt.index[1], occlusion_points_state)
    updated_img = see_visible_points(updated_img, visible_points_state)
    updated_img = see_occlusion_points(updated_img, occlusion_points_state)
    return updated_img, visible_points_state, occlusion_points_state


def change_point_type(point_type, visible_points_state, occlusion_points_state):
    if point_type == "visibility":
        text = f"Points: {visible_points_state}"
        dropdown_choices = [f"({p[0]}, {p[1]})" for p in visible_points_state]
    else:
        text = f"Points: {occlusion_points_state}"
        dropdown_choices = [f"({p[0]}, {p[1]})" for p in occlusion_points_state]
    return text, gr.Dropdown(show_label=False, choices=dropdown_choices, value=None, interactive=True)


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## 3D Amodal Reconstruction with [Amodal3R](https://sm0kywu.github.io/Amodal3R/)          
    """)

    predictor = gr.State(value=get_sam_predictor())
    visible_points_state = gr.State(value=[])
    occlusion_points_state = gr.State(value=[])
    occlusion_mask = gr.State(value=None)
    occlusion_mask_list = gr.State(value=[])
    original_image = gr.State(value=None)
    visibility_mask = gr.State(value=None)
    visibility_mask_list = gr.State(value=[])

    occluded_mask = gr.State(value=None)
    output_buf = gr.State()


    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### Step 1 - Generate Visibility and Occlusion Mask.
            * Please click "Load Example Image" when using the provided example images (bottom).
            * Please wait for a few seconds after uploading the image. Segment Anything is getting ready.
            * **Click to add the point prompts** to indicate the target object (multiple points supported) and occluders (one point for an occluder for better usability).
            * "Add mask", current mask will be saved if the input needs to be added sequentially.
            * The scale of target object can be adjusted for better reconstruction, we suggest 0.4 to 0.7 for most cases.
            """)
            with gr.Row():
                input_image = gr.Image(interactive=True, type='pil', label='Input Occlusion Image', show_label=True, sources="upload", height=300)
                input_with_prompt = gr.Image(type="numpy", label='Input with Prompt', interactive=False, height=300)
            with gr.Row():
                apply_example_btn = gr.Button("Load Example Image")
                message = gr.Markdown("Please wait a few seconds after uploading the image.", label="Message")
            with gr.Row():
                point_type = gr.Radio(["visibility", "occlusion"], label="Point Prompt Type", value="visibility")
            with gr.Row():
                with gr.Column():
                    points_text = gr.Textbox(show_label=False, interactive=False)
                with gr.Column():
                    points_dropdown = gr.Dropdown(show_label=False, choices=[], value=None, interactive=True)
                    delete_button = gr.Button("Delete Selected Point")
            with gr.Row():
                with gr.Column():
                    render_mask = gr.Image(label='Render Mask', interactive=False, height=300)
                    with gr.Row():
                        add_mask = gr.Button("Add Mask")
                        undo_mask = gr.Button("Undo Last Mask")
                with gr.Column():
                    vis_input = gr.Image(label='Visible Input', interactive=False, height=300)
                    with gr.Row():
                        zoom_scale = gr.Slider(0.3, 1.0, label="Target Object Scale", value=0.68, step=0.1)
                    with gr.Row():
                        check_visible_input = gr.Button("Generate Occluded Input")

        with gr.Column():
            gr.Markdown("""
            ### Step 2 - 3D Amodal Reconstruction. (Thanks to [TRELLIS](https://huggingface.co/spaces/JeffreyXiang/TRELLIS) for the 3D rendering component!)
            * Different random seeds can be tried in "Generation Settings", if you think the results are not ideal.
            * The boundary of the segmentation may not be accurate, so here we provide the option to erode the visible area (try 0, 3 or 5).
            * If the reconstructed 3D asset is satisfactory, interactive GLB file can be extracted (may look dull due to the absence of light source) and downloaded.
            """)
            with gr.Row():
                video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)
            with gr.Row():
                with gr.Accordion(label="Generation Settings", open=False):
                    with gr.Row():
                        with gr.Column():
                            seed = gr.Slider(0, MAX_SEED, label="Seed", value=1, step=1)
                            randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)
                        with gr.Column():
                            erode_kernel_size = gr.Slider(0, 5, label="Erode Kernel Size", value=3, step=1)
                    gr.Markdown("Stage 1: Sparse Structure Generation")
                    with gr.Row():
                        ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                        ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    gr.Markdown("Stage 2: Structured Latent Generation")
                    with gr.Row():
                        slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                        slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
            with gr.Row():
                generate_btn = gr.Button("Amodal 3D Reconstruction")
            with gr.Row():
                model_output = gr.Model3D(label="Extracted GLB", pan_speed=0.5, height=300, clear_color=(0.9,0.9,0.9,1))
            with gr.Row():
                with gr.Accordion(label="GLB Extraction Settings", open=False):
                    mesh_simplify = gr.Slider(0.9, 0.98, label="Simplify", value=0.95, step=0.01)
                    texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)
            with gr.Row():
                extract_glb_btn = gr.Button("Extract GLB")
                download_glb = gr.DownloadButton(label="Download GLB", interactive=False) 
    
    with gr.Row():
        examples = gr.Examples(
            examples=[
                f'assets/example_image/{image}'
                for image in os.listdir("assets/example_image")
            ],
            inputs=[input_image],
            fn=lambda x: x,
            outputs=[input_image],
            run_on_click=True,
            examples_per_page=12,
        )
    

    # # Handlers
    demo.load(start_session)
    demo.unload(end_session)

    input_image.upload(
        change_message,
        [],
        [message]
    ).then(
        reset_image,
        [predictor, input_image],
        [predictor, original_image, message, visible_points_state, occlusion_points_state, occlusion_mask_list, input_with_prompt],
    )
    
    apply_example_btn.click(
        change_message,
        [],
        [message]
    ).then(
        reset_image,
        inputs=[predictor, input_image],
        outputs=[predictor, original_image, message, visible_points_state, occlusion_points_state, occlusion_mask_list, input_with_prompt]
    )
    input_image.select(
        get_point,
        inputs=[input_image, point_type, visible_points_state, occlusion_points_state],
        outputs=[input_with_prompt, visible_points_state, occlusion_points_state]
    )

    point_type.change(
        change_point_type,
        inputs=[point_type, visible_points_state, occlusion_points_state],
        outputs=[points_text, points_dropdown]
    )

    visible_points_state.change(
        update_all_points,
        inputs=[visible_points_state],
        outputs=[points_text, points_dropdown]
    ).then(
        segment_and_overlay,
        inputs=[original_image, visible_points_state, predictor, visibility_mask_list, point_type],
        outputs=[render_mask, visibility_mask, visibility_mask_list]
    ).then(
        check_combined_mask,
        inputs=[original_image, visibility_mask, visibility_mask_list, occlusion_mask_list, zoom_scale],
        outputs=[vis_input, occluded_mask]
    )

    occlusion_points_state.change(
        update_all_points,
        inputs=[occlusion_points_state],
        outputs=[points_text, points_dropdown]
    ).then(
        segment_and_overlay,
        inputs=[original_image, occlusion_points_state, predictor, occlusion_mask_list, point_type],
        outputs=[render_mask, occlusion_mask, occlusion_mask_list]
    ).then(
        check_combined_mask,
        inputs=[original_image, visibility_mask, visibility_mask_list, occlusion_mask_list, zoom_scale],
        outputs=[vis_input, occluded_mask]
    )

    delete_button.click(
        delete_selected,
        inputs=[original_image, visible_points_state, occlusion_points_state, occlusion_mask_list, points_dropdown, point_type],
        outputs=[input_with_prompt, visible_points_state, occlusion_points_state, points_text, points_dropdown]
    )

    add_mask.click(
        add_current_mask,
        inputs=[visibility_mask, visibility_mask_list, point_type],
        outputs=[visibility_mask_list]
    )

    undo_mask.click(
        delete_mask,
        inputs=[visibility_mask_list, occlusion_mask_list, occlusion_points_state, point_type],
        outputs=[visibility_mask_list, occlusion_mask_list, occlusion_points_state]
    )

    check_visible_input.click(
        check_combined_mask,
        inputs=[original_image, visibility_mask, visibility_mask_list, occlusion_mask_list, zoom_scale],
        outputs=[vis_input, occluded_mask]
    )
    

    # 3D Amodal Reconstruction
    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        image_to_3d,
        inputs=[vis_input, occluded_mask, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, erode_kernel_size],
        outputs=[output_buf, video_output],
    )

    extract_glb_btn.click(
        extract_glb,
        inputs=[output_buf, mesh_simplify, texture_size],
        outputs=[model_output, download_glb],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_glb],
    )

    model_output.clear(
        lambda: gr.Button(interactive=False),
        outputs=[download_glb],
    )


    
if __name__ == "__main__":
    pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
    pipeline.cuda()
    try:
        pipeline.preprocess_image(Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)))
    except:
        pass
    demo.launch()