# ComfyUI-TeleStyle

<img width="842" height="553" alt="image" src="https://github.com/user-attachments/assets/ebc5b3e6-eaa9-4a8e-a4a5-fdce5608ea70" />

#

An unofficial, streamlined, and highly optimized ComfyUI implementation of [TeleStyle](https://github.com/Tele-AI/TeleStyle).

This node is specifically designed for **Video Style Transfer** using the **Wan2.1-T2V** architecture and TeleStyle custom weights. Unlike the original repository, this implementation strips away all heavy image-editing components (Qwen weights) to focus purely on video generation with speed/quality.





## ✨ Key Features

- **High Performance**:
  - **Acceleration**: Built-in support for **Flash Attention 2** and **SageAttention** for faster inference.
  - **Fast Mode**: Optimized memory management with aggressive cache cleanup to prevent conflicts between CPU offloading and GPU processing.

- **Simplified Workflow**: No need for complex external text encoding nodes. The model uses pre-computed stylistic embeddings (prompt_embeds.pth) for maximum efficiency.

##

<table style="width: 100%;">
  <tr>
    <td style="width: 50%; text-align: center;">
      <video src="https://github.com/user-attachments/assets/3d6310fa-f1c4-4c04-bf34-1fe0c05d3457" width="100%" controls></video>
    </td>
    <td style="width: 50%; text-align: center;">
      <video src="https://github.com/user-attachments/assets/e65001fc-f181-4da7-bd97-4f36e7700ffe" width="100%" controls></video>
     <td style="width: 50%; text-align: center;">
      <video src="https://github.com/user-attachments/assets/11c504e5-bfee-4847-846f-87f1a567dfb8" width="100%" controls></video>
    </td>
    </td>
  </tr>
</table>





## 📦 Installation

Navigate to your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes/
```

Clone this repository:

```bash
git clone https://github.com/neurodanzelus-cmd/ComfyUI-TeleStyle.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** For SageAttention support, you may need to install `sageattention` manually.

## 📂 Model Setup

This node requires specific weights placed in the `ComfyUI/models/telestyle_models/` directory.

The weights are downloaded automatically at the first run (~6gb)

**Directory Structure:**

```
ComfyUI/
└── models/
    └── telestyle_models/
        ├── weights/
        │   ├── dit.ckpt            # Main Video Transformer weights
        │   └── prompt_embeds.pth   # Pre-computed style embeddings
        └── Wan2.1-T2V-1.3B-Diffusers/
        │   ├── transformer_config.json
        │   ├── vae/
        │   │   │   ├── diffusion_pytorch_model.safetensors
        │   │   │   └── config.json
        │   └── scheduler/
        │       └── scheduler_config.json
```

**Where to get weights:**

https://huggingface.co/Danzelus/TeleStyle_comfy/tree/main

## 🚀 Usage

### 1. TeleStyle Model Loader

This node loads the necessary model components.

| Parameter | Description |
|-----------|-------------|
| `dtype` | Choose between `bf16` (best quality), `fp16` |

### 2. TeleStyle Video Transfer

The main inference node.

| Parameter | Description |
|-----------|-------------|
| `model` | Connect the output from the Loader |
| `video_frames` | Input video batch (from Load Video or VHS_LoadVideo) |
| `style_image` | A reference image to guide the style transfer |
| `steps` | Inference steps (default: 10) |
| `cfg` | Guidance scale (default: 1.5) |
| `scheduler` | Choose your sampler (`FlowMatchEuler`, `DPM++`) |
| `fast_mode` | Keep `True` for speed. Set to `False` for low-VRAM offloading (slower) |
| `acceleration` | `default` - Standard PyTorch attention<br>`flash_attn` - Faster, requires compatible GPU<br>`sage_attn` - Ultra-fast, requires sageattention library |


___

    

Guys, I’d really appreciate any support right now. I’m in a tough spot:

[![Boosty](https://img.shields.io/badge/Boosty-Support-orange?style=for-the-badge)](https://boosty.to/danzelus)
[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/danzelus)


## 📜 Credits

This project is an unofficial implementation based on the amazing work by the original authors.
Please refer to their repository for the original research and model weights.

