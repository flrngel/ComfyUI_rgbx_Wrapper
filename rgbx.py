from PIL import Image, ImageFile
import comfy.utils
import numpy as np
import cv2
import sys
from pathlib import Path
from nodes import MAX_RESOLUTION, SaveImage, common_ksampler

import os
import torch
import torchvision
from torchvision.transforms import ToTensor
from diffusers import DDIMScheduler
from .rgb2x.load_image import load_exr_image, load_ldr_image
from .rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True


def process_single_aov(torch_image, aov_name='albedo', seed=42, inference_step=50):
    """
    단일 Torch 텐서 이미지의 특정 AOV 맵을 생성하여 torch 텐서로 반환합니다.
    입력 텐서가 BWHC 형식일 경우, RGB 확인 및 변환 후 결과도 BWHC 형식으로 반환합니다.
    
    Args:
        torch_image (torch.Tensor): 처리할 입력 이미지 (B, H, W, C 형식).
        aov_name (str): 생성할 AOV 맵의 이름 (기본값: 'albedo').
        seed (int): 랜덤 시드 값.
        inference_step (int): 모델 추론 단계 수.
    
    Returns:
        torch.Tensor: 생성된 AOV 맵 텐서 (B, H, W, C 형식).
    """
    # 지원되는 AOV 목록
    supported_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    
    # AOV 유효성 검사
    if aov_name.lower() not in supported_aovs:
        raise ValueError(f"지원되지 않는 AOV입니다. 다음 중 하나를 선택하세요: {', '.join(supported_aovs)}")

    # 프롬프트 정의
    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
    }

    # 입력 텐서 확인
    if len(torch_image.shape) != 4:
        raise ValueError("input tensor must B, H, W, C ")

    # BWHC -> BCHW로 변환
    torch_image = torch_image.permute(0, 3, 1, 2)  # (B, C, H, W)

    # 배치에서 첫 번째 이미지만 사용
    photo = torch_image[0]  # 첫 번째 배치 선택 (C, H, W)
    
    photo = photo**2.2

    # 이미지 크기 조정 (8로 나누어떨어지도록 설정)
    old_height, old_width = photo.shape[1], photo.shape[2]
    old_aspect_ratio = old_height / old_width
    max_side = 1000

    if old_height > old_width:
        new_height = max_side
        new_width = int(new_height / old_aspect_ratio)
    else:
        new_width = max_side
        new_height = int(new_width * old_aspect_ratio)

    # 8의 배수로 크기 조정
    new_width = new_width // 8 * 8
    new_height = new_height // 8 * 8

    resize_transform = torchvision.transforms.Resize((new_height, new_width))
    photo = resize_transform(photo.unsqueeze(0)).squeeze(0)  # 크기 조정

    # 랜덤 시드 설정
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # 선택된 AOV 이미지 생성
    prompt = prompts[aov_name.lower()]
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=torch.float16,
        cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache"),
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    generated_image = pipe(
        prompt=prompt,
        photo=photo.unsqueeze(0).to("cuda"),  # (B=1, C, H, W)
        num_inference_steps=inference_step,
        height=new_height,
        width=new_width,
        generator=generator,
        required_aovs=[aov_name.lower()],
    ).images[0][0]

    # PIL 이미지를 torch 텐서로 변환
    generated_image_tensor = ToTensor()(generated_image)  # (C, H, W)
    

    # BCHW -> BWHC로 변환하여 반환
    generated_image_tensor = generated_image_tensor.permute(1, 2, 0).unsqueeze(0)  # (B=1, H, W, C)
    
    return generated_image_tensor




class rgb2x:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "aov": (("albedo", "normal", "roughness", "metallic", "irradiance"), {"default": "albedo"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", { "default": 50, "min": 1, "max": 0xffffffffffffffff, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "ToyxyzTestNodes"
        
    def execute(self, image: torch.Tensor, aov, seed, steps):
        
        output = process_single_aov(image, aov, seed, steps)

        return(output, )


NODE_CLASS_MAPPINGS = {
    "rgb2x": rgb2x,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "rgb2x": "rgb2x"
}

