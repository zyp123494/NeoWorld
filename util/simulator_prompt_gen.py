import json
import os

from openai import OpenAI

os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
import torch

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://openrouter.ai/api/v1",  # "https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class SimulatorPromptGen:
    def __init__(self):
        super(SimulatorPromptGen, self).__init__()
        self.model = "google/gemini-2.5-pro-preview"

    def generate_prompt(self, bbox, instance_info, user_prompt):
        """
        Generate simulator prompt

        Args:
            bbox: M×3×2 PyTorch tensor, representing bounding boxes (x_min, x_max, y_min, y_max, z_min, z_max)
            instance_info: List of length M, each element is (instance_id, category)
            user_prompt: User provided prompt

        Returns:
            Simulator parameters in JSON format
        """
        # Calculate center points from bounding boxes
        centers = []
        sizes = []

        # Prompt enginering, make +y represent up
        bbox[:, 1, :] *= -1

        centers_ = (bbox[:, :, 0] + bbox[:, :, 1]) / 2
        min_value = centers_.min(0).values
        for i in range(bbox.shape[0]):
            # Calculate center as (min + max) / 2 for each dimension
            center_x = (bbox[i, 0, 0] + bbox[i, 0, 1]) / 2 - min_value[0]
            center_y = (bbox[i, 1, 0] + bbox[i, 1, 1]) / 2 - min_value[1]
            center_z = (bbox[i, 2, 0] + bbox[i, 2, 1]) / 2 - min_value[2]

            # Calculate size as max - min for each dimension
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
        system_message = """You are a simulator assistant. Next you will be provided with object information in a scene and a user prompt.
You need to identify the foreground objects most likely to interact with each other, and estimate appropriate MPM attributes for each. Please do not select background objects for simulation.

When selecting an object to simulate:
1. Pay close attention to any spatial indicators in the user prompt (e.g., "the apple on the left", "the top plate", "the apple falling onto the plate")
2. Consider object descriptions (position, size) when multiple objects of the same category exist
3. Select objects that are mentioned in the user prompt or are likely to participate in the described interaction
4. Most scenes involve 1-3 foreground objects interacting with each other

For each selected object, you should provide simulation parameters including:
- Material type should be from the following list: ['jelly', 'sand', 'foam', 'snow', 'plasticine']. Here's a guide to help you select the appropriate material:
  * jelly: For elastic objects that can deform and return to their original shape (like rubber, soft fruits, gelatin-like substances). Best for simulating bouncy, elastic objects. Young's modulus (E): 1e4-1e6, Poisson's ratio (nu): 0.3-0.45
  * sand: For granular materials that can flow but maintain volume (like sand, sugar, rice). Best for simulating grainy substances that pour. Young's modulus (E): 1e6-1e8, Poisson's ratio (nu): 0.2-0.3, friction_angle: 30-45
  * foam: For soft, compressible materials that absorb impact (like cushions, sponges, styrofoam). Young's modulus (E): 1e3-1e5, Poisson's ratio (nu): 0.1-0.3
  * snow: For brittle, lightweight materials that can break apart and accumulate (like snow, powder). Young's modulus (E): 1e4-1e6, Poisson's ratio (nu): 0.2-0.3
  * plasticine: For materials that deform permanently and don't return to original shape (like clay, dough, plasticine). Best for simulating objects that can be molded. Young's modulus (E): 1e5-1e7, Poisson's ratio (nu): 0.3-0.4
Note that when there is a conflict between object's material and the dynamics described in the user prompt, the dynamics in the user prompt take precedence.
  
  For rigid objects like furniture, use 'jelly' with a high Young's modulus (E: 1e5-1e7).
  For soft objects like fruits, pillows, use 'jelly' with low Young's modulus (E: 1e2-1e4).
  For moldable objects like clay or dough, use 'plasticine'.
  For grainy substances like sugar or salt, use 'sand'.

- Density should be set appropriately based on the material and object type:
  * Light objects (paper, feathers): 100-500 kg/m³
  * Medium-weight objects (fruits, plastic toys): 500-1500 kg/m³
  * Heavy objects (rocks, metals): 2000-8000 kg/m³
  * Water and similar liquids: ~1000 kg/m³
  * Wood: 400-800 kg/m³
  * Standard food items: 800-1200 kg/m³

- The coordinate system is defined as follows: +x points to the right of the image, +y points upward, and +z points into the scene (i.e., away from the viewer).
- E (Young's modulus) represents stiffness - higher values mean stiffer materials
- nu (Poisson's ratio) represents how much a material contracts in directions perpendicular to the direction it is stretched
- friction_angle affects how materials slide against each other
- force should be a 3D vector [f_x, f_y, f_z], representing the applied force, and should be set reasonably according to the description of dynamics in the user prompt. Appropriate force magnitude is typically between 5-20 to create visible motion and interaction effects.
- The gravity is set to [0, -9.8, 0], so objects will naturally fall freely. The force only takes effect at the first timestamp and can be set to 0 if not needed.
- The ground has been set, so you do not need to select background objects like floors, walls, seas, ground, earth or landscapes. 

Please use the format below (the output should be JSON format):
{
  "objects": [
    {
      "instance_id": instance_id_1,
      "material_params": {
        "material": material_1,
        "E": E_1,
        "nu": nu_1,
        "friction_angle": friction_angle_1,
        "density": density_1
      },
      "force": [f_x_1, f_y_1, f_z_1]
    },
    {
      "instance_id": instance_id_2,
      "material_params": {
        "material": material_2,
        "E": E_2,
        "nu": nu_2,
        "friction_angle": friction_angle_2,
        "density": density_2
      },
      "force": [f_x_2, f_y_2, f_z_2]
    }
  ]
}"""

        # Build user message with clear instructions about object selection
        user_message = f"""Object information in the scene:
{instances_str}

User prompt: {user_prompt}

Based on the provided information and user prompt, please identify the foreground objects that are most likely to interact with each other in the described scenario. Don't select any background objects like floors, walls, ground, earth, seas or landscapes. 

For each selected object, provide appropriate material parameters and forces. Consider the physical properties of each object based on its category and the intended interaction. If you need to apply a force to create the described interaction, make sure it's appropriately directed and scaled (typically between 5-20 in magnitude for clear visual effects).

Let’s think through this step by step: first, assume a reasonable trajectory, then infer a force that would produce such a motion."""

        print(user_message)

        # Call API to get response
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        for i in range(10):  # Try up to 10 times
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=messages,
                    timeout=10,
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
                    assert "material_params" in obj, "Missing material_params field"
                    assert "material" in obj["material_params"], (
                        "Missing material field"
                    )
                    assert "E" in obj["material_params"], "Missing E field"
                    assert "nu" in obj["material_params"], "Missing nu field"
                    assert "friction_angle" in obj["material_params"], (
                        "Missing friction_angle field"
                    )
                    assert "density" in obj["material_params"], "Missing density field"
                    assert "force" in obj, "Missing force field"
                    assert len(obj["force"]) == 3, "force must be a list of length 3"

                return result

            except Exception as e:
                print(f"API call error: {e}")
                # if i < 2:  # If not the last attempt
                #     print("Waiting 1 second before retrying...")
                #     time.sleep(1)
                # else:
                #     # If all attempts fail, return default values
                #     return {
                #         "instance_idx": 0,
                #         "material_params": {
                #             "material": "jelly",
                #             "E": 1e6,
                #             "nu": 0.3,
                #             "friction_angle": 30
                #         },
                #         "force": [0.0, -9.8, 0.0]
                #     }

    def simulate(self, bbox, instance_info, user_prompt):
        """
        Main interface function to process inputs and return simulation parameters

        Args:
            bbox: M×3×2 PyTorch tensor, representing bounding boxes
            instance_info: List of length M, each element is (instance_id, category)
            user_prompt: User provided prompt

        Returns:
            Dictionary of simulator parameters for multiple objects
        """
        # Ensure input format is correct
        assert isinstance(bbox, torch.Tensor), "bbox must be a PyTorch tensor"
        assert bbox.dim() == 3 and bbox.size(1) == 3 and bbox.size(2) == 2, (
            "bbox should have shape M×3×2"
        )
        assert len(instance_info) == bbox.size(0), (
            "instance_info length should match first dimension of bbox"
        )

        # Generate simulation parameters
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

            # Cap force values
            obj["force"] = [min(max(x, -60), 60) for x in obj["force"]]

            # Flip on y axis, since we have flipped in the prompt
            obj["force"][1] = -obj["force"][1]

            # Validate instance id exists in instance_info
            if instance_id not in valid_ids:
                print(
                    f"Warning: Instance ID {obj['instance_id']} not found in instance_info, setting to first instance ID"
                )
                obj["instance_id"] = valid_ids[0] if valid_ids else 0

            # Validate material is in allowed list
            valid_materials = ["jelly", "metal", "sand", "foam", "snow", "plasticine"]
            if obj["material_params"]["material"] not in valid_materials:
                print(
                    f"Warning: Material {obj['material_params']['material']} not in allowed list, setting to jelly"
                )
                obj["material_params"]["material"] = "jelly"

            # Cap density to reasonable values
            obj["material_params"]["density"] = min(
                max(obj["material_params"]["density"], 100), 10000
            )
            obj["material_params"]["E"] = min(1e6, obj["material_params"]["E"])

            filtered_objects.append(obj)

        result["objects"] = filtered_objects
        return result
