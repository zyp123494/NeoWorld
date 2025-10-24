import os

import cv2
import numpy as np
import torch
from PIL import Image

from .Amodal3R.pipelines import Amodal3RImageTo3DPipeline


def preprocess_image_mask(image, mask, scale=0.68):
    """
    预处理图像和mask，与app.py中的check_combined_mask函数逻辑相同
    """
    if mask.sum() == 0:
        return np.zeros_like(image), np.zeros_like(image[:, :, 0])

    updated_image = image.copy()
    combined_mask = (mask > 0).astype(np.uint8)
    occluded_mask = 1 - combined_mask

    # 处理图像
    masked_img = updated_image * combined_mask[:, :, None]
    occluded_mask = (1 - occluded_mask) * 255
    occluded_mask[combined_mask == 1] = 127

    # 计算边界框
    x, y, w, h = cv2.boundingRect(combined_mask.astype(np.uint8))

    # 计算缩放比例
    ori_h, ori_w = masked_img.shape[:2]
    target_size = 512
    scale_factor = target_size / max(w, h)
    final_scale = scale_factor * scale

    # 缩放图像和mask
    new_w = int(round(ori_w * final_scale))
    new_h = int(round(ori_h * final_scale))
    resized_img = cv2.resize(
        masked_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
    )
    resized_occluded_mask = cv2.resize(
        occluded_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )

    # 创建输出图像和mask
    final_img = np.zeros((target_size, target_size, 3), dtype=updated_image.dtype)
    final_occluded_mask = np.ones((target_size, target_size), dtype=np.uint8) * 255

    # 计算居中位置
    new_x = int(round(x * final_scale))
    new_y = int(round(y * final_scale))
    new_w_box = int(round(w * final_scale))
    new_h_box = int(round(h * final_scale))

    new_cx = new_x + new_w_box // 2
    new_cy = new_y + new_h_box // 2
    final_cx, final_cy = target_size // 2, target_size // 2
    x_offset = final_cx - new_cx
    y_offset = final_cy - new_cy

    # 计算放置位置
    final_x_start = max(0, x_offset)
    final_y_start = max(0, y_offset)
    final_x_end = min(target_size, x_offset + new_w)
    final_y_end = min(target_size, y_offset + new_h)

    img_x_start = max(0, -x_offset)
    img_y_start = max(0, -y_offset)
    img_x_end = min(new_w, target_size - x_offset)
    img_y_end = min(new_h, target_size - y_offset)

    # 放置图像和mask
    final_img[final_y_start:final_y_end, final_x_start:final_x_end] = resized_img[
        img_y_start:img_y_end, img_x_start:img_x_end
    ]
    final_occluded_mask[final_y_start:final_y_end, final_x_start:final_x_end] = (
        resized_occluded_mask[img_y_start:img_y_end, img_x_start:img_x_end]
    )

    return final_img, final_occluded_mask


def inference(image, mask, pipeline=None, output_path="./output"):
    # 加载模型
    if pipeline is None:
        pipeline = Amodal3RImageTo3DPipeline.from_pretrained(
            "Sm0kyWu/Amodal3R",
            ss_path="/data/vjuicefs_ai_camera_lgroup_ql/11181721/ICLR26/Amodal3R_sspose/train_sparse_strcuture_50/epoch=49.ckpt",
        )
        pipeline.cuda()

    # 读取图像和mask

    image, mask = preprocess_image_mask(image, mask)

    # 设置参数
    seed = 42
    ss_guidance_strength = 7.5
    ss_sampling_steps = 12
    slat_guidance_strength = 3.0
    slat_sampling_steps = 12
    erode_kernel_size = 3

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    outputs = pipeline.run_multi_image(
        [image],
        [mask],
        seed=seed,
        formats=["gaussian"],  # , "mesh"],
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
        sample_pose=True,
    )

    # 保存结果
    # video = render_utils.render_video(outputs['gaussian'][0], num_frames=120, bg_color=(1,1,1))['color']
    # video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    # video =video_geo# [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]

    # 保存结果
    gaussian = outputs["gaussian"][0]
    pose = torch.from_numpy(outputs["pose"][0]).reshape([2, 3]).float().cuda()

    # mesh = outputs['mesh'][0]

    # 保存高斯点云
    gaussian.save_ply(f"{output_path}/gaussian.ply", transform=None)

    # # 保存网格
    # glb = postprocessing_utils.to_obj(
    #     gaussian,
    #     mesh,
    #     simplify=0.95,
    #     fill_holes=True,
    #     fill_holes_max_size=0.04,
    #     texture_size=1024,
    #     verbose=False
    # )
    # vertices = mesh.vertices.cpu().numpy()
    # faces = mesh.faces.cpu().numpy()
    # vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # mesh = trimesh.Trimesh(vertices, faces)

    # mesh.export(f"{output_path}/mesh.obj")

    # # 保存视频
    # video_path = f"{output_path}/video.mp4"
    # imageio.mimsave(video_path, video, fps=15)

    return gaussian, pose


if __name__ == "__main__":
    image_path = "/data/vjuicefs_ai_camera_lgroup_ql/11181721/WonderWorld_seg_bg_seg_ppt_multi_seg/output/zelda/Gen-01-05_18-11-00/gaussian_scene00.png"
    mask_path = "/data/vjuicefs_ai_camera_lgroup_ql/11181721/WonderWorld_seg_bg_seg_ppt_multi_seg/output/zelda/Gen-01-05_18-11-00/gaussian_scene00_seg.png"
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))
    mask = mask == 2
    output_path = "./output"
    os.makedirs(output_path, exist_ok=True)
    outputs = inference(image, mask, output_path)
