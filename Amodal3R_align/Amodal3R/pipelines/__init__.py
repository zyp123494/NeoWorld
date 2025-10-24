from . import samplers
from .image_to_3d import Amodal3RImageTo3DPipeline


def from_pretrained(path: str):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
    """
    import json
    import os

    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download

        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file) as f:
        config = json.load(f)
    return globals()[config["name"]].from_pretrained(path)
