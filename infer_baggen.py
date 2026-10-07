"""Generate bags from one texture source and an optional structure condition."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 1024
PATCH_SIZE = 256
PROMPT_TEMPLATE = "A {category}, detailed, centered, high-quality, product photo, white background"
NEGATIVE_PROMPT = "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdxl_path", type=Path, default=None, help="Local sdxl path; must be provided.")
    parser.add_argument("--vae_path", type=Path, default=None, help="Local vae path; must be provided.")
    parser.add_argument("--ip_adapter_weight_path", type=Path, default=None, help="Local ip adapter (baggen) weight path; must be provided.")
    parser.add_argument("--image_encoder_path", type=Path, default=None, help="Local image encoder path; must be provided.")
    parser.add_argument("--texture_image_path", type=Path, required=True, help="Source image; its central 256x256 texture patch is used.")
    category_group = parser.add_mutually_exclusive_group(required=True)
    category_group.add_argument("--property_json", type=Path, help='JSON object containing a nonempty "category" string.')
    category_group.add_argument("--category", help="Bag category, inserted into the original product-photo prompt.")
    parser.add_argument("--controlnet_path", type=Path, default=None, help="ControlNet directory; requires --structure_image_path.")
    parser.add_argument("--structure_image_path", type=Path, help="Contour or scribble image; requires --controlnet_path.")
    parser.add_argument("--ipadapter_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_try", type=int, default=5, help="Sequential generations for this same input.")
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed; generation i uses seed+i (i starts at 0).")
    parser.add_argument("--output_path", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args(argv)

    if (args.controlnet_path is None) != (args.structure_image_path is None):
        parser.error("--controlnet_path and --structure_image_path must be provided together")
    if args.num_try < 1 or args.num_inference_steps < 1:
        parser.error("--num_try and --num_inference_steps must be positive")
    if args.seed is not None and not 0 <= args.seed <= 2**64 - args.num_try:
        parser.error("--seed through seed+num_try-1 must be in [0, 2**64-1]")

    for name in ("sdxl_path", "vae_path", "ip_adapter_weight_path", "image_encoder_path"):
        if getattr(args, name) is None:
            parser.error(f"--{name} must be provided; download the model and specify its local path")

    for name in ("sdxl_path", "vae_path", "image_encoder_path", "controlnet_path"):
        path = getattr(args, name)
        if path is not None and not path.is_dir():
            parser.error(f"--{name}: directory does not exist: {path}")
    for name in ("ip_adapter_weight_path", "texture_image_path", "property_json", "structure_image_path"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            parser.error(f"--{name}: file does not exist: {path}")

    if args.property_json is not None:
        try:
            with args.property_json.open(encoding="utf-8") as handle:
                properties = json.load(handle)
            if not isinstance(properties, dict):
                parser.error("--property_json must contain a JSON object")
            args.category = properties.get("category")
        except (OSError, ValueError) as error:
            parser.error(f"cannot read --property_json: {error}")
    if not isinstance(args.category, str) or not args.category.strip():
        parser.error("category must be a nonempty string")
    return args


def prepare_texture(texture_image_path):
    """Keep the resize rounding, LANCZOS sampling and center crops of the old scripts."""
    from PIL import Image

    with Image.open(texture_image_path) as image:
        source = image.convert("RGB")
    width, height = source.size
    if width < height:
        new_width = IMAGE_SIZE
        new_height = int(height * IMAGE_SIZE / width)
        resized = source.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        top = (new_height - IMAGE_SIZE) // 2
        resized = resized.crop((0, top, IMAGE_SIZE, top + IMAGE_SIZE))
    else:
        new_height = IMAGE_SIZE
        new_width = int(width * IMAGE_SIZE / height)
        resized = source.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        left = (new_width - IMAGE_SIZE) // 2
        resized = resized.crop((left, 0, left + IMAGE_SIZE, IMAGE_SIZE))
    left = (resized.width - PATCH_SIZE) // 2
    top = (resized.height - PATCH_SIZE) // 2
    return resized, resized.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))


def build_pipeline(args):
    import torch
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        EulerAncestralDiscreteScheduler,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLPipeline,
    )
    from transformers import CLIPVisionModelWithProjection

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        str(args.image_encoder_path), torch_dtype=torch.float16
    )
    if image_encoder.config.projection_dim != 1024:
        raise ValueError("BagGEN needs a 1024-dimensional ViT-H image encoder (models/image_encoder).")
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(str(args.sdxl_path), subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(str(args.vae_path), torch_dtype=torch.float16)
    pipeline_kwargs = dict(
        vae=vae,
        image_encoder=image_encoder,
        safety_checker=None,
        torch_dtype=torch.float16,
        scheduler=scheduler,
    )
    pipeline_class = StableDiffusionXLPipeline
    if args.controlnet_path is not None:
        pipeline_kwargs["controlnet"] = ControlNetModel.from_pretrained(
            str(args.controlnet_path), torch_dtype=torch.float16
        )
        pipeline_class = StableDiffusionXLControlNetPipeline
    pipe = pipeline_class.from_pretrained(str(args.sdxl_path), **pipeline_kwargs).to("cuda")
    # Encoder and adapter live in separate repositories. The encoder is already
    # registered; load_ip_adapter creates the same default CLIPImageProcessor.
    pipe.load_ip_adapter(
        str(args.ip_adapter_weight_path.parent),
        subfolder="",
        weight_name=args.ip_adapter_weight_path.name,
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(args.ipadapter_conditioning_scale)
    return pipe


def main(args):
    import torch
    from PIL import Image
    from tqdm import tqdm

    _, patch = prepare_texture(args.texture_image_path)
    structure = None
    if args.structure_image_path is not None:
        with Image.open(args.structure_image_path) as image:
            structure = image.convert("RGB")

    prompt = PROMPT_TEMPLATE.format(category=args.category)
    print(f"Mode: {'texture + structure' if structure is not None else 'texture only'}")
    print(f"Prompt: {prompt}")
    print(f"Output: {args.output_path}")
    pipe = build_pipeline(args)
    args.output_path.mkdir(parents=True, exist_ok=True)
    inference_kwargs = dict(
        negative_prompt=NEGATIVE_PROMPT,
        ip_adapter_image=patch,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    if structure is not None:
        inference_kwargs.update(image=structure, controlnet_conditioning_scale=args.controlnet_conditioning_scale)

    with torch.inference_mode():
        for index in tqdm(range(args.num_try), desc="Inference BagGEN"):
            if args.seed is not None:
                inference_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(args.seed + index)
            result = pipe(prompt, **inference_kwargs).images[0]
            result.save(args.output_path / f"result_{index:03d}.png")


if __name__ == "__main__":
    main(parse_args())
