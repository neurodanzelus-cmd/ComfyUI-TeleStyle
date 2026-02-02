import torch
import os
import folder_paths
import numpy as np
import time
from contextlib import nullcontext
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import (
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler
)

from .telestylevideo_transformer import WanTransformer3DModel
from .telestylevideo_pipeline import WanPipeline

class TeleStyleLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"dtype": (["fp16", "bf16"], {"default": "bf16"})}}

    RETURN_TYPES = ("TELE_STYLE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_all"
    CATEGORY = "TeleStyle"

    def load_all(self, dtype):
        device = torch.device("cuda")
        repo_id = "Danzelus/TeleStyle_comfy"
        base_path = os.path.join(folder_paths.models_dir, "telestyle_models")
        
        if dtype == "bf16":
            target_dtype = torch.bfloat16
            vae_dtype = torch.bfloat16
        else:
            target_dtype = torch.float16
            vae_dtype = torch.float16
        
        files = [
            "weights/dit.ckpt", "weights/prompt_embeds.pth",
            "Wan2.1-T2V-1.3B-Diffusers/transformer_config.json",
            "Wan2.1-T2V-1.3B-Diffusers/vae/config.json",
            "Wan2.1-T2V-1.3B-Diffusers/vae/diffusion_pytorch_model.safetensors",
            "Wan2.1-T2V-1.3B-Diffusers/scheduler/scheduler_config.json"
        ]
        
        for f in files:
            dest = os.path.join(base_path, f)
            if not os.path.exists(dest):
                try:
                    hf_hub_download(repo_id=repo_id, filename=f, local_dir=base_path, local_dir_use_symlinks=False)
                except Exception as e:
                    print(f"TeleStyle Warning: Could not download {f}: {e}")

        wan_path = os.path.join(base_path, "Wan2.1-T2V-1.3B-Diffusers")
        
        vae = AutoencoderKLWan.from_pretrained(os.path.join(wan_path, "vae"), torch_dtype=vae_dtype).to(device)
        
        config = OmegaConf.to_container(OmegaConf.load(os.path.join(wan_path, "transformer_config.json")))
        transformer = WanTransformer3DModel(**config)
        
        ckpt_path = os.path.join(base_path, "weights/dit.ckpt")
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            if "transformer_state_dict" in sd:
                sd = sd["transformer_state_dict"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            transformer.load_state_dict(sd)
        
        transformer.to(device, dtype=target_dtype)

        embeds_path = os.path.join(base_path, "weights/prompt_embeds.pth")
        if os.path.exists(embeds_path):
            loaded_embeds = torch.load(embeds_path, map_location="cpu")
            if isinstance(loaded_embeds, dict):
                p_embeds = loaded_embeds.get("prompt_embeds", loaded_embeds)
                n_embeds = loaded_embeds.get("negative_prompt_embeds", torch.zeros_like(p_embeds))
            else:
                p_embeds = loaded_embeds
                n_embeds = torch.zeros_like(p_embeds)
        else:
            p_embeds = torch.zeros(1, 1, 4096, dtype=vae_dtype)
            n_embeds = torch.zeros(1, 1, 4096, dtype=vae_dtype)

        scheduler_config_path = os.path.join(wan_path, "scheduler")
        scheduler_config = FlowMatchEulerDiscreteScheduler.load_config(scheduler_config_path)
        try:
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config, shift=3.0)
        except TypeError:
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_config_path)

        return ({
            "vae": vae, 
            "transformer": transformer, 
            "scheduler": scheduler, 
            "p_embeds": p_embeds, 
            "n_embeds": n_embeds, 
            "dtype": vae_dtype,
            "device": device,
            "target_dtype": target_dtype,
            "vae_dtype": vae_dtype,
        },)

class TeleStyleVideoInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TELE_STYLE_MODEL",),
                "video_frames": ("IMAGE",), 
                "steps": ("INT", {"default": 10, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 42}),
                "scheduler": (["FlowMatchEuler", "UniPC", "DPM++"], {"default": "FlowMatchEuler"}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "enable_tiling": ("BOOLEAN", {"default": True}),
                "acceleration": (["default", "flash_attn", "sage_attn", "mem_efficient"], {"default": "default"}),
            },
            "optional": {
                "style_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "TeleStyle"

    def process(self, model, video_frames, steps, cfg, seed, scheduler, fast_mode, enable_tiling, acceleration, style_image=None):
        device = torch.device("cuda")
        m = model
        dtype = m["dtype"]
        
        if enable_tiling or video_frames.shape[0] > 16: 
            m["vae"].enable_tiling()
            m["vae"].enable_slicing()
        else: 
            m["vae"].disable_tiling()
            m["vae"].disable_slicing()

        base_config = m["scheduler"].config
        scheduler_map = {
            "FlowMatchEuler": FlowMatchEulerDiscreteScheduler,
            "UniPC": UniPCMultistepScheduler,
            "DPM++": DPMSolverMultistepScheduler,
        }
        Cls = scheduler_map.get(scheduler, FlowMatchEulerDiscreteScheduler)
        
        if scheduler == "UniPC":
            pipe_scheduler = Cls.from_config(
                base_config,
                num_train_timesteps=1000,
                solver_order=2,
                prediction_type="flow_prediction",
                flow_shift=3.0,
                use_flow_sigmas=True,
            )
        elif scheduler == "DPM++":
            # FIXED: proper DPM++ initialization for flow matching
            # Take config from FlowMatchEuler and adapt for DPM++
            pipe_scheduler = Cls.from_config(
                base_config,
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="scaled_linear",
                solver_order=2,
                prediction_type="flow_prediction",
                algorithm_type="dpmsolver++",
                use_flow_sigmas=True,  # ← KEY parameter!
                flow_shift=3.0,  # ← Required!
            )
        else:
            try:
                pipe_scheduler = Cls.from_config(base_config, shift=3.0)
            except TypeError:
                pipe_scheduler = Cls.from_config(base_config)

        pipe = WanPipeline(transformer=m["transformer"], vae=m["vae"], scheduler=pipe_scheduler)
        
        if fast_mode: 
            pipe.to(device)
        else: 
            pipe.enable_sequential_cpu_offload()

        p_embeds = m["p_embeds"].to(device, dtype=dtype)
        n_embeds = m["n_embeds"].to(device, dtype=dtype)

        F_in = video_frames.shape[0]
        target_frames = ((F_in - 1) // 4 + 1) * 4 + 1
        if ((target_frames - 1) // 4 + 1) == 2: 
            target_frames += 4
        needed = target_frames - F_in
        if needed > 0:
            last_frame = video_frames[-1:].repeat(needed, 1, 1, 1)
            video_frames = torch.cat([video_frames, last_frame], dim=0)

        H, W = video_frames.shape[1], video_frames.shape[2]
        H_new = (H // 16) * 16
        W_new = (W // 16) * 16
        if H_new != H or W_new != W:
            video_frames = video_frames[:, :H_new, :W_new, :]
            if style_image is not None:
                style_image = style_image[:, :H_new, :W_new, :]

        src = video_frames.permute(3, 0, 1, 2).unsqueeze(0).to(device, dtype=dtype)
        src = (src - 0.5) * 2.0 
        
        if style_image is None:
            style_image = torch.rand((1, H_new, W_new, 3), dtype=video_frames.dtype, device=video_frames.device)
        
        ref = style_image.permute(3, 0, 1, 2).unsqueeze(0)
        ref = ref[:, :, :1, :, :].to(device, dtype=dtype)
        ref = (ref - 0.5) * 2.0

        attn_context = nullcontext()
        if acceleration == "sage_attn":
            try: 
                attn_context = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
            except: 
                pass
        elif acceleration == "flash_attn":
            attn_context = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        elif acceleration == "mem_efficient":
            attn_context = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)

        with torch.no_grad():
            s_lat = m["vae"].encode(src).latent_dist.mode()
            f_lat = m["vae"].encode(ref).latent_dist.mode()
            
            vae_config = m["vae"].config
            latents_mean = torch.tensor(vae_config.latents_mean).view(1, -1, 1, 1, 1).to(device, dtype=dtype)
            latents_std = torch.tensor(vae_config.latents_std).view(1, -1, 1, 1, 1).to(device, dtype=dtype)
            
            s_lat = (s_lat - latents_mean) / latents_std
            f_lat = (f_lat - latents_mean) / latents_std

        with torch.no_grad(), attn_context:
            output = pipe(
                source_latents=s_lat,
                first_latents=f_lat,
                neg_first_latents=torch.zeros_like(f_lat),
                prompt_embeds=p_embeds,
                negative_prompt_embeds=n_embeds,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=torch.Generator(device=device).manual_seed(seed),
                output_type="latent"
            )
            latents = output.frames 

        with torch.no_grad():
            latents = latents.to(m["vae"].dtype)
            latents = latents * latents_std + latents_mean
            
            video = m["vae"].decode(latents, return_dict=False)[0]
            video = (video / 2 + 0.5).clamp(0, 1)
            video = video.permute(0, 2, 3, 4, 1).float()
            out = video

        if not fast_mode:
            torch.cuda.empty_cache()

        if out.ndim == 5: 
            out = out.squeeze(0) 
        if out.shape[0] == 3: 
            out = out.permute(1, 2, 3, 0)
        elif out.shape[1] == 3: 
            out = out.permute(0, 2, 3, 1)

        if out.shape[0] > F_in: 
            out = out[:F_in]

        return (torch.clamp(out, 0.0, 1.0).cpu().float(),)

NODE_CLASS_MAPPINGS = {
    "TeleStyleLoader": TeleStyleLoader,
    "TeleStyleVideoInference": TeleStyleVideoInference
}