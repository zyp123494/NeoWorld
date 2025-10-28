# NeoWorld

[Project Page](https://zyp123494.github.io/NeoWorld.github.io/) | [Paper](https://arxiv.org/abs/2509.24441)

Official code repository for:  
**NeoWorld: Neural Simulation of Explorable Virtual Worlds via Progressive 3D Unfolding**  
Yanpeng Zhao, [Shanyan Guan](https://syguan96.github.io/), [Yunbo Wang](https://wyb15.github.io/)<sup>†</sup>, Yanhao Ge, Wei Li, [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ&hl=zh-CN)

## Getting Started

### Installation

```bash
git clone https://github.com/zyp123494/NeoWorld.git && cd NeoWorld
conda create -n neoworld python=3.10
conda activate neoworld

# Install PyTorch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install submodules
pip install submodules/depth-diff-gaussian-rasterization-min-features/
pip install submodules/simple-knn/
```

Install the rest of the requirements:

```bash
pip install -r requirements.txt
cd ./RepViT/sam && pip install -e . && cd ../..
python -m spacy download en_core_web_sm
```

Download the RepViT model and place it in the root directory:
```bash
wget https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt
```
Download our fine-tuned [Amodal3R checkpoint](https://drive.google.com/file/d/1xDwU7dOReeGLUf8WmsOubOFXZJT6VH4U/view?usp=sharing) and place it in the root directory.

We use [OpenRouter](https://openrouter.ai/) for LLM APIs. Please replace the API key in `run_*.sh` files with your own.

### Run Examples

#### Interactive Demo

  We use the tools from [splat](https://github.com/haoyi-duan/splat.git) for local visualization.

  ###### Local Visualization Setup:
  
  On your local laptop, clone this project and open `splat/index_stream.html`.
  
  To enable interactive visualization of your results through this local web browser, follow these steps:
  
  - Ensure you have `'ssh'` installed on your local machine.
  - The main program will run on server user_id@server_name
  
  ```shell
  # On your local machine
  ssh -L 7777:localhost:7777 server_name
  ```
  
  ###### Main Program Running:
  
  On the server, run the main program:
  
  ```bash
  # On user_id@server_name
  bash run_demo.sh
  ```
  More examples are located at `config/more_examples`, feel free to try!
  
  ###### Interactive Generation Step:
  
  Open the `index_stream.html` on your local machine, and you should see the scene in it. You can navigate with `WSAD` and arrow keys.
  
  1. For scene wondering, you can manually input scene description you want in the text box of the local browser. Remember to click 'Next scene is ...' after you are done.  
  2. Next you need to set a proper camera view for the program to generate new scene. You can do this by wondering through the browser to a novel view, then press key `'R'` to let program interactively generate new scene in this view for you. 
  3. For simulation and animation, you need to input corresponding prompt in the text box. Remember to click 'Simulate/Animate' after you are done, then press key 'P' for simulation and 'J' for animation.
  4. If you are not satisfied with the current generation, you can press key `Z` to delete the previous one generation, and follow step 1 and 3 to do a new generation.
  5. Repeat 1-4, you will interactively generate a large-scale connected scene, and you can wonder through the scene freely during the whole process.
  6. After some generation, you can press key `X` to save the current scene. Next time, you can load the generated scene by specifying `load_gen=True` in your configuration file.

#### Local Execution

Alternatively, you can run NeoWorld locally:
- Set `rotation_path` in `config/base-config.yaml`
- Run `bash run.sh` for basic generation
- Run `bash run_simulation.sh` or `bash run_animation.sh` for simulation/animation tasks

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhao2025neoworld,
  title={NeoWorld: Neural Simulation of Explorable Virtual Worlds via Progressive 3D Unfolding},
  author={Zhao, Yanpeng and Guan, Shanyan and Wang, Yunbo and Ge, Yanhao and Li, Wei and Yang, Xiaokang},
  journal={arXiv preprint arXiv:2509.24441},
  year={2025}
}
```

## Acknowledgements

We sincerely appreciate the authors of [Marigold](https://github.com/prs-eth/Marigold), [SyncDiffusion](https://github.com/KAIST-Visual-AI-Group/SyncDiffusion), [RepViT](https://github.com/THU-MIG/RepViT), [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting), [OneFormer](https://github.com/SHI-Labs/OneFormer), [WonderJourney](https://github.com/KovenYu/WonderJourney), and [WonderWorld](https://github.com/KovenYu/WonderWorld) for sharing their excellent work.
