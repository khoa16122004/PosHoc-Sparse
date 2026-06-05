from __future__ import annotations

import json
from constant import IMAGENET_PROMPT_PATH, CLIP_PARAMS, OPENCLIP_PARAMS, SIGLIP_PARAMS
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder
import clip
import open_clip
import torchvision.models as tv_models
from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

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
        T.Resize(param['size'], interpolation=T.InterpolationMode.BILINEAR),
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






def get_CLIP_model(
    model_name,
    ):
    
    model, preprocess = clip.load(model_name.replace("_", "/"))
    model = model.cuda()
    spatial, normalize = split_VLMs_transform(CLIP_PARAMS[model_name])
    model.eval()
    return model, spatial, normalize


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
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModel.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    spatial, normalize = split_VLMs_transform(SIGLIP_PARAMS[model_name])
    model = model.cuda()
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





if __name__ == "__main__":
    # model = get_torchvision_model("resnet18", pretrained=True)
    
    
    model, spatial, normalize, tokenizer = get_SIGLIP_model(
        model_name="google/siglip-base-patch16-224"
    )
    
    