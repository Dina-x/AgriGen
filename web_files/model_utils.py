from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

TRIGGER_WORD = "agrigenstyle"
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

COMMON_OBJECTS = {
    "apple", "banana", "cherry", "tomato", "orange", "lemon", "strawberry", "mango",
    "pear", "peach", "kiwi", "pineapple", "watermelon", "grape", "pomegranate",
    "avocado", "cucumber", "carrot", "potato", "onion", "pepper", "corn", "eggplant",
    "plum", "apricot", "cantaloupe", "raspberry", "blueberry", "cabbage", "cauliflower",
    "broccoli", "walnut", "hazelnut", "almond", "almonds", "pistachio", "peanut",
}


def _find_root(models_dir: str | Path = "models") -> Path:
    models_dir = Path(models_dir)
    nested = models_dir / "AgriGen_Lite_Fast"
    if nested.exists():
        return nested
    return models_dir


def _find_lora(root: Path) -> tuple[Path, str]:
    candidates = [
        root / "lora_weights" / "pytorch_lora_weights.safetensors",
        root / "lora_weights" / "checkpoint-600" / "pytorch_lora_weights.safetensors",
    ]
    for p in candidates:
        if p.exists():
            return p.parent, p.name
    matches = list(root.rglob("pytorch_lora_weights.safetensors"))
    if matches:
        p = matches[0]
        return p.parent, p.name
    raise FileNotFoundError("Could not find pytorch_lora_weights.safetensors inside the models folder.")


def load_supported_prompts(models_dir: str | Path = "models") -> list[str]:
    root = _find_root(models_dir)
    path = root / "metadata" / "supported_prompts.json"
    if not path.exists():
        return sorted(COMMON_OBJECTS)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_lora_pipeline(models_dir: str | Path = "models"):
    root = _find_root(models_dir)
    lora_folder, lora_weight_name = _find_lora(root)

    local_base = Path(models_dir) / "sd15_model"
    base_model = str(local_base) if local_base.exists() else BASE_MODEL_ID

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.load_lora_weights(str(lora_folder), weight_name=lora_weight_name)

    supported = load_supported_prompts(models_dir)
    return pipe, supported


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resolve_object(text: str, supported_prompts: Iterable[str]) -> str:
    text = clean_text(text)
    supported = [clean_text(x) for x in supported_prompts]

    if text in supported:
        return text

    for obj in sorted(supported, key=len, reverse=True):
        if text == obj or text in obj or obj in text:
            return obj

    for obj in sorted(COMMON_OBJECTS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(obj) + r"\b", text):
            if obj == "almond":
                return "almonds"
            return obj
    
    raise ValueError(
        "Unsupported prompt. Please enter an agricultural object such as apple, banana, tomato, carrot, or almond."
    )


def build_prompt(text: str, supported_prompts: Iterable[str]) -> tuple[str, str, str]:
    obj = resolve_object(text, supported_prompts)
    prompt = (
        f"a realistic close-up photo of {TRIGGER_WORD} {obj} "
        "on a plain white background, centered composition, sharp details, studio lighting, high quality"
    )
    negative = (
        "blurry, low quality, cartoon, drawing, painting, illustration, distorted, deformed, "
        "multiple fruits, duplicate, extra objects, text, watermark, logo, hands, people, noisy background"
    )
    return prompt, negative, obj


def apply_style(image: Image.Image, style: str = "Natural") -> Image.Image:
    style = str(style).lower().strip()
    img = image.convert("RGB")

    if style in ["natural", "none", "normal", "original", ""]:
        return img

    if style in ["soft grayscale", "gray", "grey", "grayscale"]:
        gray = ImageOps.grayscale(img)
        gray = ImageOps.autocontrast(gray)
        return gray.convert("RGB")

    if style in ["sketch", "pencil", "drawing"]:
        gray = ImageOps.grayscale(img)
        inverted = ImageOps.invert(gray)
        blurred = inverted.filter(ImageFilter.GaussianBlur(radius=12))
        gray_np = np.array(gray).astype(np.float32)
        blur_np = np.array(blurred).astype(np.float32)
        sketch_np = gray_np * 255.0 / (255.0 - blur_np + 1e-6)
        sketch_np = np.clip(sketch_np, 0, 255).astype(np.uint8)
        sketch = Image.fromarray(sketch_np)
        sketch = ImageOps.autocontrast(sketch)
        return sketch.convert("RGB")

    if style in ["bright", "lighting", "light", "enhance"]:
        img = ImageEnhance.Brightness(img).enhance(1.12)
        img = ImageEnhance.Contrast(img).enhance(1.12)
        img = ImageEnhance.Sharpness(img).enhance(1.35)
        img = ImageEnhance.Color(img).enhance(1.05)
        return img.convert("RGB")

    return img


@torch.no_grad()
def generate_image(
    prompt: str,
    pipe,
    supported_prompts: list[str],
    style: str = "Natural",
    seed: int | None = None,
    steps: int = 25,
    guidance_scale: float = 7.5,
) -> tuple[Image.Image, str]:
    full_prompt, negative, obj = build_prompt(prompt, supported_prompts)

    device = pipe.device
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    generator = torch.Generator(device=device).manual_seed(int(seed))

    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        width=512,
        height=512,
        generator=generator,
    ).images[0]

    return apply_style(image, style), obj
