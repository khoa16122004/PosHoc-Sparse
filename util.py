from __future__ import annotations

import json
from typing import Optional, Tuple
from torchvision import transforms
from constant import IMAGENET_PROMPT_PATH, CLIP_PARAMS
import torchvision.models as tv_models
from torchvision.models import get_model_weights
import torchvision.transforms as T
from LossFunctions import UnTargeted, Targeted
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from constant import CLIP_PARAMS
import clip
from tqdm import tqdm

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
        T.Resize(resize),
        T.CenterCrop(crop),
        T.ToTensor()
    ])

    normalize = T.Normalize(mean=mean, std=std)

    return spatial, normalize


def split_VLMs_transform(
    param
):
    
    spatial = T.Compose([
        T.Resize(param['size']),
        T.CenterCrop(param['crop_size']),
        T.ToTensor()
    ])

    normalize = T.Normalize(mean=param['mean'], std=param['std'])

    return spatial, normalize




def get_torchvision_model(
        model_name,
        pretrained=True,
        num_classes=None,
    ):
    """
    Get vision model
    """
    

    model_fn = getattr(tv_models, model_name)

    if pretrained:
        weights_enum = get_model_weights(model_name).DEFAULT
        model = model_fn(weights=weights_enum)

        spatial, normalize = split_transform_from_weights(weights_enum)

        return model, spatial, normalize

    kwargs = {}
    if num_classes is not None:
        kwargs["num_classes"] = num_classes

    model = model_fn(weights=None, **kwargs)

    return model, None, None

def get_CLIP_model(
    model_name,
    ):
    
    import clip
    
    model, _ = clip.load(model_name)
    model = model.cuda()
    spatial, normalize = split_VLMs_transform(CLIP_PARAMS[model_name])
    return model, spatial, normalize






class VisionModelWrapper:
    "Vison Wrapper for vision-only models, e.g., ResNet, ViT, etc."
    
    
    def __init__(self, model, normalize, device):
        self.model = model
        self.normalize = normalize
        self.device = device

    def predict(self, x):
        x = x.to(self.device)
        x = self.normalize(x)
        with torch.no_grad():
            logits = self.model(x)
        return logits.detach().cpu()
    
    
class VLModelWrapper:
    def __init__(self, model, normalize, class_prompts, device):
        self.model = model
        self.normalize = normalize
        self.class_prompts = class_prompts
        self.device = device
        
        # extract text_feature
        textual_class_features = []
        print("Extract class_text_features...")
        for class_name in self.class_prompts:
            text_features = []            
            for prompt in self.class_prompts[class_name]:
                fea = self.text_encode([prompt])
                text_features.append(fea)
            text_features = torch.stack(text_features, dim=0).mean(dim=0)
        
            textual_class_features.append(text_features)
        
        self.class_text_features = torch.stack(textual_class_features).to(self.device)
        print("Class text feautures shape: ", self.class_text_features.shape)
        
    def predict(self, x):
        x = x.to(self.device)
        x = self.normalize(x)
        visual_features = self.vision_encode(x)
        logits = visual_features @ self.class_text_features.T       
        return logits.detach().cpu()
    
    
    def vision_encode(self, x):
        vision_features = self.model.encode_image(x)
        vision_features = vision_features / vision_features.norm(dim=-1, keepdim=True)
        return vision_features
    
    def text_encode(self, t): # t lists
        t = clip.tokenize(t).cuda(self.device)
        text_features = self.model.encode_text(t)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features




def get_intersection(clean_map, adv_map):
    clean_map = np.asarray(clean_map, dtype=np.float32)
    adv_map = np.asarray(adv_map, dtype=np.float32)
    inter = np.minimum(clean_map, adv_map).sum()
    union = np.maximum(clean_map, adv_map).sum() + 1e-12
    return float(inter / union)


if __name__ == "__main__":
    # model = get_torchvision_model("resnet18", pretrained=True)
    
    
    model, spatial, normalize = get_CLIP_model("ViT-B/32")
    x = torch.randn(1, 3, 224, 224)
    with open(IMAGENET_PROMPT_PATH, 'r') as f:
        class_prompts = json.load(f)
    
    img = Image.open(r"imgs/tabby.jpg")
    img = spatial(img).unsqueeze(0)
    
    
    model = VLModelWrapper(
        model, 
        normalize,
        class_prompts,
        device="cuda"
        )    
    logits = model.predict(img)
    print(logits.shape)
    print(logits.argmax(dim=-1))