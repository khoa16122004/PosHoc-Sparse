from __future__ import annotations

import json
from constant import IMAGENET_PROMPT_PATH, CLIP_PARAMS, OPENCLIP_PARAMS, SIGLIP_PARAMS, VIT_PARAMS
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder
import clip
import open_clip
import torchvision.models as tv_models

from transformers import (
    AutoModel, 
    AutoTokenizer, 
    SiglipModel, 
    ViTForImageClassification,
    CLIPModel,
)



_DATASET_NUM_CLASSES = {
    "imagenet": 1000,
    "imagenet1k": 1000,
    "cifar10": 10,
    "cifar100": 100,
    "mnist": 10,
    "fashionmnist": 10,
    "svhn": 10,
    "caltech101": 101,
    "caltech256": 256,
}


def split_transform_from_weights(weights):

    resize = weights.transforms().resize_size
    crop = weights.transforms().crop_size
    mean = weights.transforms().mean
    std = weights.transforms().std

    spatial = T.Compose([
        T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(crop),
        T.ToTensor()
    ])

    normalize = T.Normalize(mean=mean, std=std)

    return spatial, normalize


def split_transform_from_ViT(param):
    spatial = T.Compose([
        T.Resize((param['size'], param['size']), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])

    normalize = T.Normalize(mean=param['mean'], std=param['std'])

    return spatial, normalize


def split_VLMs_transform(
    param
):
    # CLIP, OPENCLIP, SIGLIP
    spatial = T.Compose([
        T.Resize(param['size'], interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(param['crop_size']),
        T.ToTensor()
    ])

    normalize = T.Normalize(mean=param['mean'], std=param['std'])

    return spatial, normalize

def split_SIGLIP_transform(
    param
):
    # CLIP, OPENCLIP, SIGLIP
    spatial = T.Compose([
        T.Resize((param['size'], param['size']), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])

    normalize = T.Normalize(mean=param['mean'], std=param['std'])

    return spatial, normalize





def get_torchvision_model(
        model_name,
    ):
    """
    Get vision model
    """

    model_fn = getattr(tv_models, model_name)
    weights_enum = tv_models.get_model_weights(model_name).IMAGENET1K_V1
    model = model_fn(weights=weights_enum)
    spatial, normalize = split_transform_from_weights(weights_enum)
    model.eval()
    return model, spatial, normalize

def get_ViT_model(
    model_name
):
    model = ViTForImageClassification.from_pretrained(model_name, attn_implementation="eager").cuda()
    spatial, normalize = split_transform_from_ViT(VIT_PARAMS[model_name])
    model.eval()
    return model, spatial, normalize


# def get_CLIP_model(
#     model_name,
#     ):
    
#     model, preprocess = clip.load(model_name.replace("_", "/"))
#     model = model.cuda()
#     spatial, normalize = split_VLMs_transform(CLIP_PARAMS[model_name])
#     model.eval()
#     return model, spatial, normalize

def get_CLIP_model(
    model_name,
    ):

    model = CLIPModel.from_pretrained(
        model_name,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.cuda()
    model.eval()
    
    spatial, normalize = split_VLMs_transform(
        CLIP_PARAMS[model_name]
    )
    return model, spatial, normalize, tokenizer

def get_OPENCLIP_model(
    model_name,
    ):
    
    model_name_ = model_name.replace("_", "-")
    if model_name == "ViT-B_32":
        pretrained = 'laion2b_s34b_b79k'
    elif model_name == "ViT-B_16":
        pretrained = "laion2b_s34b_b88k"
    elif model_name == "ViT-L_14":
        pretrained = "laion2b_s32b_b82k"
        
    model, _, preprocess = open_clip.create_model_and_transforms(model_name_, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name_)
    model = model.cuda()
    spatial, normalize = split_VLMs_transform(OPENCLIP_PARAMS[model_name])
    return model, spatial, normalize, tokenizer


def get_SIGLIP_model(
    model_name,
    ):
    model = SiglipModel.from_pretrained(model_name, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    spatial, normalize = split_SIGLIP_transform(SIGLIP_PARAMS[model_name])
    model = model.cuda()
    model.eval()
    return model, spatial, normalize, tokenizer
    
    
def get_BEIT3_model(
    model_name,
    ):
    pass


class ImageNetVal(ImageFolder):
	def __getitem__(self, index: int):
		sample, target = super().__getitem__(index)
		path, _ = self.samples[index]
		return sample, target, path




def compose_grid_image(components: list[dict[str, Any]], cell_size: int) -> Image.Image:
    if cell_size <= 0:
        raise ValueError("cell_size must be greater than 0.")

    grid_image = Image.new("RGB", (cell_size * 2, cell_size * 2))
    positions = [(0, 0), (cell_size, 0), (0, cell_size), (cell_size, cell_size)]

    for component, position in zip(components, positions):
        img_path = component.get("img_path")
        if not img_path:
            raise ValueError("Each component must include an img_path.")

        source_path = Path(img_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Image not found: {source_path}")

        with Image.open(source_path) as source_image:
            tile = source_image.convert("RGB").resize((cell_size, cell_size))
            grid_image.paste(tile, position)

    return grid_image

def compute_grid_scores(
    explain_map : torch.Tensor,
    single_shape: int,
) -> torch.Tensor:
    if single_shape <= 0:
        raise ValueError("single_shape must be greater than 0.")

    if explain_map.ndim == 3:
        explain_map = explain_map.unsqueeze(1)
    elif explain_map.ndim != 4:
        raise ValueError(
            f"explain_map must have shape [B, H, W] or [B, 1, H, W], got {tuple(explain_map.shape)}"
        )

    positive_maps = explain_map.clamp(min=0)
    pooled = F.avg_pool2d(
        positive_maps,
        kernel_size=single_shape,
        stride=single_shape,
    ).flatten(start_dim=1)
    totals = pooled.sum(dim=1, keepdim=True)
    return torch.where(totals > 0, pooled / totals, torch.zeros_like(pooled))

if __name__ == "__main__":
    # model = get_torchvision_model("resnet18", pretrained=True)
    
    
    model, spatial, normalize, tokenizer = get_SIGLIP_model(
        model_name="google/siglip-base-patch16-224"
    )
    
    