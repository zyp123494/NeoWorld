import json
import os
import time

import numpy as np
import torch
from openai import OpenAI
from scipy.interpolate import interp1d


class AnimationPromptGen:
    def __init__(self):
        super(AnimationPromptGen, self).__init__()
        self.model = "google/gemini-2.5-pro-preview"
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )

    def interpolate_to_length(self, data, target_length=100):
        """
        将任意长度的数据线性插值到指定长度

        Args:
            data: List of lists, 数据序列，如translation或rotation
            target_length: 目标长度，默认为100

        Returns:
            插值后的数据，长度为target_length
        """
        # Check original data length
        original_length = len(data)
        if original_length == target_length:
            return data

        # Create original time points and target time points
        original_times = np.linspace(0, 1, original_length)
        target_times = np.linspace(0, 1, target_length)

        # Perform interpolation
        result = []
        # Convert data to numpy array for interpolation
        data_array = np.array(data)

        # Interpolate each dimension separately
        for dim in range(data_array.shape[1]):
            interpolator = interp1d(original_times, data_array[:, dim], kind="linear")
            result.append(interpolator(target_times))

        # Convert result back to original format
        result = np.array(result).T.tolist()
        return result

    def generate_prompt(self, bbox, instance_info, user_prompt):
        """
        Generate animation prompt

        Args:
            bbox: M×3×2 PyTorch tensor, representing bounding boxes (x_min, x_max, y_min, y_max, z_min, z_max)
            instance_info: List of length M, each element is (instance_id, category)
            user_prompt: User provided prompt

        Returns:
            Animation parameters in JSON format
        """
        # Calculate center points from bounding boxes
        centers = []
        sizes = []

        # Prompt engineering, make +y represent up
        bbox[:, 1, :] *= -1

        centers_ = (bbox[:, :, 0] + bbox[:, :, 1]) / 2
        min_value = centers_.min(0).values
        for i in range(bbox.shape[0]):
            # Calculate center as (min + max) / 2 for each dimension
            center_x = (bbox[i, 0, 0] + bbox[i, 0, 1]) / 2 - min_value[0]
            center_y = (bbox[i, 1, 0] + bbox[i, 1, 1]) / 2 - min_value[1]
            center_z = (bbox[i, 2, 0] + bbox[i, 2, 1]) / 2 - min_value[2]

            size_x = abs(bbox[i, 0, 1] - bbox[i, 0, 0])
            size_y = abs(bbox[i, 1, 1] - bbox[i, 1, 0])
            size_z = abs(bbox[i, 2, 1] - bbox[i, 2, 0])

            centers.append((center_x, center_y, center_z))
            sizes.append((size_x, size_y, size_z))

        # Build instance information string with more descriptive details
        instances_str = ""
        category_counts = {}

        # First, count occurrences of each category
        for _, category in instance_info:
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1

        # Then build detailed descriptions, especially for duplicate categories
        for i, (instance_id, category) in enumerate(instance_info):
            center = centers[i]
            size = sizes[i]

            # Build the full description
            count_info = ""
            if category_counts[category] > 1:
                count_info = (
                    f" (one of {category_counts[category]} {category}s in the scene)"
                )

            instances_str += (
                f"Instance ID={instance_id}, Category={category}{count_info}, "
            )
            instances_str += (
                f"Position=[x={center[0]:.2f}, y={center[1]:.2f}, z={center[2]:.2f}], "
            )
            instances_str += f"Size=[width={size[0]:.2f}, height={size[1]:.2f}, depth={size[2]:.2f}]\n"

        # Build system message with improved guidance
        system_message = """You are an animation assistant. Next you will be provided with object information in a scene and a user prompt.
You need to identify the foreground objects most likely to interact with each other, and generate appropriate animation trajectories for each. Please do not select background objects for animation.

When selecting an object to animate:
1. Pay close attention to any spatial indicators in the user prompt (e.g., "the apple on the left", "the top plate", "the apple falling onto the plate")
2. Consider object descriptions (position, size) when multiple objects of the same category exist
3. Select objects that are mentioned in the user prompt or are likely to participate in the described interaction
4. Most scenes involve 1-3 foreground objects interacting with each other

For each selected object, you should provide animation parameters including:
- Translation: A list of 100 [x,y,z] coordinates representing the translation of the object's center over 100 time steps, note that translation indicate the delta values to object centers.
- Rotation: A list of 100 [roll,pitch,yaw] Euler angles (in degrees) representing the rotation of the object over 100 time steps

Important notes:
- The coordinate system is defined as follows: +x points to the right of the image, +y points upward, and +z points into the scene (i.e., away from the viewer).
- Translation values should stay within reasonable bounds, typically not exceeding 2-3 times the object's size, to avoid objects moving out of frame
- Rotation values should use Euler angles in degrees, with a range of 0-360 degrees
- Animations should transition smoothly, avoiding sudden changes
- Animations should match the actions or interactions described by the user
- For smooth animation, changes between consecutive frames should be gradual

Please use the format below (the output should be JSON format):
{
  "objects": [
    {
      "instance_id": instance_id_1,
      "translation": [
        [dx_0, dy_0, dz_0],
        [dx_1, dy_1, dz_1],
        ...
        [dx_99, dy_99, dz_99]
      ],
      "rotation": [
        [roll_0, pitch_0, yaw_0],
        [roll_1, pitch_1, yaw_1],
        ...
        [roll_99, pitch_99, yaw_99]
      ]
    },
    {
      "instance_id": instance_id_2,
      "translation": [
        [dx_0, dy_0, dz_0],
        [dx_1, dy_1, dz_1],
        ...
        [dx_99, dy_99, dz_99]
      ],
      "rotation": [
        [roll_0, pitch_0, yaw_0],
        [roll_1, pitch_1, yaw_1],
        ...
        [roll_99, pitch_99, yaw_99]
      ]
    }
  ]
}"""

        # Build user message with clear instructions about object selection
        user_message = f"""Object information in the scene:
{instances_str}

User prompt: {user_prompt}

Based on the provided information and user prompt, please identify the foreground objects that are most likely to interact with each other in the described scenario. Don't select any background objects like floors, walls, ground, earth, seas or landscapes.

For each selected object, provide appropriate translation and rotation trajectories. Consider the motion characteristics of each object based on its category and the intended interaction.

Let's think through this step by step: first, determine reasonable animation trajectories, then generate 100 smoothly transitioning keyframes for those trajectories.

Please ensure that:
1. Translation values remain within reasonable bounds (typically not exceeding 2-3 times the object's dimensions from its initial position), translation should change smoothly.
2. Rotation values use Euler angles (degrees) and change smoothly
3. Animation trajectories make physical sense and match the user's description
4. Data for all 100 time steps must be provided"""

        print(user_message)

        # Call API to get response
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        for i in range(10):  # Try up to 10 times
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=messages,
                    timeout=30,
                )

                response_content = response.choices[0].message.content
                print(response_content)

                # Clean response content to ensure it's valid JSON
                if "```json" in response_content:
                    response_content = response_content.split("```json")[1]
                if "```" in response_content:
                    response_content = response_content.split("```")[0]
                response_content = response_content.strip()

                # Parse JSON
                result = json.loads(response_content)

                # Validate result contains all necessary fields
                assert "objects" in result, "Missing objects field"
                assert isinstance(result["objects"], list), "objects should be a list"
                assert len(result["objects"]) > 0, "objects list should not be empty"

                # Validate each object in the list
                for obj in result["objects"]:
                    assert "instance_id" in obj, "Missing instance_id field"
                    assert "translation" in obj, "Missing translation field"
                    assert "rotation" in obj, "Missing rotation field"

                    # Validate each translation and rotation element format
                    for t in obj["translation"]:
                        assert len(t) == 3, (
                            "Each element in translation must be a list of length 3"
                        )

                    for r in obj["rotation"]:
                        assert len(r) == 3, (
                            "Each element in rotation must be a list of length 3"
                        )

                return result

            except Exception as e:
                print(f"API call error: {e}")
                if i < 9:  # If not the last attempt
                    print("Waiting 1 second before retrying...")
                    time.sleep(1)

        # If all attempts fail, return default values
        return {
            "objects": [
                {
                    "instance_id": instance_info[0][0] if instance_info else 0,
                    "translation": [[0.0, 0.0, 0.0] for _ in range(100)],
                    "rotation": [[0.0, 0.0, 0.0] for _ in range(100)],
                }
            ]
        }

    def animate(self, bbox, instance_info, user_prompt):
        """
        Main interface function to process inputs and return animation parameters

        Args:
            bbox: M×3×2 PyTorch tensor, representing bounding boxes
            instance_info: List of length M, each element is (instance_id, category)
            user_prompt: User provided prompt

        Returns:
            Dictionary of animation parameters for multiple objects
        """
        # Ensure input format is correct
        assert isinstance(bbox, torch.Tensor), "bbox must be a PyTorch tensor"
        assert bbox.dim() == 3 and bbox.size(1) == 3 and bbox.size(2) == 2, (
            "bbox should have shape M×3×2"
        )
        assert len(instance_info) == bbox.size(0), (
            "instance_info length should match first dimension of bbox"
        )

        # Generate animation parameters
        result = self.generate_prompt(bbox, instance_info, user_prompt)
        print(result)

        # Get valid instance IDs
        valid_ids = [int(id) for id, _ in instance_info]

        seen_ids = set()
        filtered_objects = []

        for obj in result["objects"]:
            instance_id = int(obj["instance_id"])

            if instance_id in seen_ids:
                continue  # Skip duplicate instance_id
            seen_ids.add(instance_id)

            # Validate instance id exists in instance_info
            if instance_id not in valid_ids:
                print(
                    f"Warning: Instance ID {obj['instance_id']} not found in instance_info, setting to first instance ID"
                )
                obj["instance_id"] = valid_ids[0] if valid_ids else 0

            # Check translation and rotation lengths and interpolate if necessary
            translation_length = len(obj["translation"])
            rotation_length = len(obj["rotation"])

            if translation_length != 100:
                print(
                    f"Warning: Translation length ({translation_length}) is not 100, interpolating..."
                )
                obj["translation"] = self.interpolate_to_length(obj["translation"], 100)

            if rotation_length != 100:
                print(
                    f"Warning: Rotation length ({rotation_length}) is not 100, interpolating..."
                )
                obj["rotation"] = self.interpolate_to_length(obj["rotation"], 100)

            # Apply smoothing filter to interpolated data
            print("Applying smoothing filter to trajectories...")
            obj["translation"] = self.smooth_trajectory(
                obj["translation"], window_size=7, iterations=2
            )
            obj["rotation"] = self.smooth_trajectory(
                obj["rotation"], window_size=7, iterations=2
            )

            # Get object's initial position and size
            idx = valid_ids.index(instance_id)
            center = [
                (bbox[idx, 0, 0] + bbox[idx, 0, 1]) / 2,
                -1
                * (bbox[idx, 1, 0] + bbox[idx, 1, 1])
                / 2,  # Because we flipped y-axis when generating
                (bbox[idx, 2, 0] + bbox[idx, 2, 1]) / 2,
            ]

            size = [
                bbox[idx, 0, 1] - bbox[idx, 0, 0],
                bbox[idx, 1, 1] - bbox[idx, 1, 0],
                bbox[idx, 2, 1] - bbox[idx, 2, 0],
            ]

            # Validate and constrain translation within reasonable bounds
            for i in range(100):
                # Limit x-direction displacement
                # obj['translation'][i][0] = np.clip(
                #     obj['translation'][i][0],
                #     center[0] - 3 * size[0],
                #     center[0] + 3 * size[0]
                # )

                # # Limit y-direction displacement
                # obj['translation'][i][1] = np.clip(
                #     obj['translation'][i][1],
                #     center[1] - 3 * size[1],
                #     center[1] + 3 * size[1]
                # )

                # # Limit z-direction displacement
                # obj['translation'][i][2] = np.clip(
                #     obj['translation'][i][2],
                #     center[2] - 3 * size[2],
                #     center[2] + 3 * size[2]
                # )

                # Ensure rotation angles are within 0-360 degrees
                for j in range(3):
                    obj["rotation"][i][j] = obj["rotation"][i][j] % 360

            # Flip translations on y-axis again to match coordinate system
            for i in range(100):
                obj["translation"][i][1] = -obj["translation"][i][1]

            filtered_objects.append(obj)

        result["objects"] = filtered_objects
        return result

    def smooth_trajectory(self, data, window_size=5, iterations=2):
        """
        对轨迹数据进行平滑滤波，使动画更加流畅

        Args:
            data: List of lists, 数据序列，如translation或rotation
            window_size: 平滑窗口大小，必须为奇数
            iterations: 平滑迭代次数

        Returns:
            平滑后的数据
        """
        if window_size % 2 == 0:
            window_size += 1  # Ensure window size is odd

        # Convert to numpy array for processing
        data_array = np.array(data)
        smoothed = data_array.copy()

        # 在时间维度上应用平滑滤波
        for _ in range(iterations):
            temp = smoothed.copy()
            radius = window_size // 2

            # 对数据中间部分应用窗口平滑
            for i in range(radius, len(smoothed) - radius):
                window = smoothed[i - radius : i + radius + 1]
                temp[i] = np.mean(window, axis=0)

            # 处理边界
            for i in range(radius):
                # 起始边界: 使用可用数据的平均值
                available = smoothed[: i + radius + 1]
                temp[i] = np.mean(available, axis=0)

                # 结束边界: 使用可用数据的平均值
                end_idx = len(smoothed) - 1 - i
                available = smoothed[end_idx - radius :]
                temp[end_idx] = np.mean(available, axis=0)

            smoothed = temp

        # # 确保角度数据保持在合理范围内（如果是旋转数据）
        # # 旋转数据通常是角度，应该保持在0-360度或-180到180度之间
        # if np.max(np.abs(smoothed)) > 10:  # 假设较大的值可能是角度
        #     smoothed = smoothed % 360  # 保持在0-360度范围内

        return smoothed.tolist()
