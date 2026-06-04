from __future__ import annotations

import json
from constant import IMAGENET_PROMPT_PATH, CLIP_PARAMS, OPENCLIP_PARAMS
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
    
    spatial = T.Compose([
        T.Resize(param['size'], interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(param['crop_size']),
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
    model_name = model_name.replace("_", "/")
    model, preprocess = clip.load(model_name)
    model = model.cuda()
    spatial, normalize = split_VLMs_transform(CLIP_PARAMS[model_name])
    model.eval()
    print(preprocess)
    return model, spatial, normalize


def get_OPENCLIP_model(
    model_name,
    ):
    
    
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.cuda()
    spatial, normalize = split_VLMs_transform(OPENCLIP_PARAMS[model_name])
    return model, spatial, normalize, tokenizer


class ImageNetVal(ImageFolder):
	def __getitem__(self, index: int):
		sample, target = super().__getitem__(index)
		path, _ = self.samples[index]
		return sample, target, path





if __name__ == "__main__":
    # model = get_torchvision_model("resnet18", pretrained=True)
    
    
    model, spatial, normalize, tokenizer = get_OPENCLIP_model("ViT-B/32")

    
    x = torch.randn(1, 3, 224, 224)
    with open(IMAGENET_PROMPT_PATH, 'r') as f:
        class_prompts = json.load(f)
    
    img = Image.open(r"/datastore/elo/quanphm/dataset/ImageNet1K/val/n01440764/ILSVRC2012_val_00023559.JPEG").convert("RGB")
    img = spatial(img).unsqueeze(0)
    
    
    model = VLModelWrapper(
        model, 
        normalize,
        class_prompts,
        tokenizer=tokenizer,
        device="cuda"
        )    
    logits = model.predict(img)
    print(logits.shape)
    print(logits.argmax(dim=-1))