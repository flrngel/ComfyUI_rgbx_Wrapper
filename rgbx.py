from PIL import Image, ImageFile
import comfy.utils
import comfy.model_management
import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor
from diffusers import DDIMScheduler
import os

# Import the pipelines for both rgb2x and x2rgb
from .rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from .x2rgb.pipeline_x2rgb import StableDiffusionAOVDropoutPipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Globals for pipeline caching ---
rgb2x_pipe_cached = None
x2rgb_pipe_cached = None


# --- Helper function for rgb2x ---
def process_single_aov(torch_image, aov_name='albedo', seed=42, inference_step=50):
    """
    Generates a single AOV map from a torch tensor image.
    Caches the pipeline to avoid reloading on subsequent runs.
    """
    global rgb2x_pipe_cached
    device = comfy.model_management.get_torch_device()

    supported_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    if aov_name.lower() not in supported_aovs:
        raise ValueError(f"Unsupported AOV. Choose from: {', '.join(supported_aovs)}")

    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
    }

    # Load and cache the pipeline
    if rgb2x_pipe_cached is None:
        rgb2x_pipe_cached = StableDiffusionAOVMatEstPipeline.from_pretrained(
            "zheng95z/rgb-to-x",
            torch_dtype=torch.float16,
            cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache"),
        )
        rgb2x_pipe_cached.scheduler = DDIMScheduler.from_config(
            rgb2x_pipe_cached.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
        )
        rgb2x_pipe_cached.set_progress_bar_config(disable=True)
    
    pipe = rgb2x_pipe_cached.to(device)

    # Preprocess image: (B, H, W, C) -> (C, H, W) and convert to linear space
    photo = torch_image[0].permute(2, 0, 1)
    photo = photo ** 2.2

    # Resize for model compatibility
    old_height, old_width = photo.shape[1], photo.shape[2]
    old_aspect_ratio = old_height / old_width
    max_side = 1000
    
    if max(old_height, old_width) > max_side:
        if old_height > old_width:
            new_height = max_side
            new_width = int(new_height / old_aspect_ratio)
        else:
            new_width = max_side
            new_height = int(new_width * old_aspect_ratio)
    else:
        new_height, new_width = old_height, old_width

    new_width = new_width // 8 * 8
    new_height = new_height // 8 * 8

    photo_resized = torchvision.transforms.functional.resize(photo, (new_height, new_width), antialias=True)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt = prompts[aov_name.lower()]

    # Run pipeline
    generated_image_pil = pipe(
        prompt=prompt,
        photo=photo_resized.unsqueeze(0),
        num_inference_steps=inference_step,
        height=new_height,
        width=new_width,
        generator=generator,
        required_aovs=[aov_name.lower()],
    ).images[0][0]

    # Postprocess: PIL to Tensor
    generated_image_tensor = ToTensor()(generated_image_pil)
    
    # Resize back to original dimensions
    if (new_height, new_width) != (old_height, old_width):
        generated_image_tensor = torchvision.transforms.functional.resize(generated_image_tensor, (old_height, old_width), antialias=True)
    
    # Format for ComfyUI: (C, H, W) -> (B, H, W, C)
    output_tensor = generated_image_tensor.permute(1, 2, 0).unsqueeze(0)
    
    comfy.model_management.soft_empty_cache()
    return output_tensor


# --- Helper function for x2rgb ---
def process_x2rgb(prompt, seed, steps, guidance_scale, image_guidance_scale, albedo, normal, roughness, metallic, irradiance):
    """
    Generates an RGB image from AOV maps.
    Caches the pipeline to avoid reloading on subsequent runs.
    """
    global x2rgb_pipe_cached
    device = comfy.model_management.get_torch_device()

    # Load and cache the pipeline
    if x2rgb_pipe_cached is None:
        x2rgb_pipe_cached = StableDiffusionAOVDropoutPipeline.from_pretrained(
            "zheng95z/x-to-rgb",
            torch_dtype=torch.float16,
            cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache"),
        )
        x2rgb_pipe_cached.scheduler = DDIMScheduler.from_config(
            x2rgb_pipe_cached.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
        )
        x2rgb_pipe_cached.set_progress_bar_config(disable=True)
    
    pipe = x2rgb_pipe_cached.to(device)

    input_images = {'albedo': albedo, 'normal': normal, 'roughness': roughness, 'metallic': metallic, 'irradiance': irradiance}
    
    first_image = next((img for img in input_images.values() if img is not None), None)
    if first_image is None:
        raise ValueError("At least one input image (AOV) is required for x2rgb.")
    
    h, w = first_image.shape[1], first_image.shape[2]

    def preprocess_image(image_tensor, image_type):
        if image_tensor is None:
            return None
        
        # Input: (B, H, W, C) from ComfyUI, float [0,1]
        # Output: (1, C, H, W), preprocessed for the pipe
        img = image_tensor[0].permute(2, 0, 1)

        if image_type in ['albedo', 'irradiance']:
            img = torch.clamp(img, 0.0, 1.0) ** 2.2  # sRGB to linear
        elif image_type == 'normal':
            img = img * 2.0 - 1.0  # Normalize to [-1, 1]
        elif image_type in ['roughness', 'metallic']:
            img = torch.clamp(img, 0.0, 1.0)
        
        return img.unsqueeze(0)  # Add batch dim

    pipe_inputs = {name: preprocess_image(tensor, name) for name, tensor in input_images.items()}
    generator = torch.Generator(device=device).manual_seed(seed)

    # Run pipeline
    generated_image_np = pipe(
        prompt=prompt,
        height=h,
        width=w,
        albedo=pipe_inputs['albedo'],
        normal=pipe_inputs['normal'],
        roughness=pipe_inputs['roughness'],
        metallic=pipe_inputs['metallic'],
        irradiance=pipe_inputs['irradiance'],
        num_inference_steps=steps,
        generator=generator,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        output_type="np"
    ).images[0]
    
    # Format for ComfyUI: numpy (H, W, C) -> torch (B, H, W, C)
    output_tensor = torch.from_numpy(generated_image_np).unsqueeze(0)
    
    comfy.model_management.soft_empty_cache()
    return output_tensor


# --- Node Classes ---
class rgb2x:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "aov": (("albedo", "normal", "roughness", "metallic", "irradiance"), {"default": "albedo"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", { "default": 50, "min": 1, "max": 100, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "rgbx"
        
    def execute(self, image: torch.Tensor, aov, seed, steps):
        output = process_single_aov(image, aov, seed, steps)
        return (output,)


class x2rgb:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "masterpiece, best quality, photorealistic"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", { "default": 50, "min": 1, "max": 100, "step": 1, }),
                "guidance_scale": ("FLOAT", { "default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1, }),
                "image_guidance_scale": ("FLOAT", { "default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1, }),
            },
            "optional": {
                "albedo": ("IMAGE",),
                "normal": ("IMAGE",),
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
                "irradiance": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "rgbx"

    def execute(self, prompt, seed, steps, guidance_scale, image_guidance_scale, albedo=None, normal=None, roughness=None, metallic=None, irradiance=None):
        output = process_x2rgb(
            prompt=prompt,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            albedo=albedo,
            normal=normal,
            roughness=roughness,
            metallic=metallic,
            irradiance=irradiance
        )
        return (output,)


# --- Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "rgb2x": rgb2x,
    "x2rgb": x2rgb,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "rgb2x": "RGB to AOV",
    "x2rgb": "AOV to RGB"
}