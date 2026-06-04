IMAGENET_PROMPT_PATH = "description/imgnet1k_description.json"
IMAGENET_FOLDER2_CLASSNAME = "description/imgnet1k_label.json"
DEFAULT_VAL_DIR = "/datastore/elo/quanphm/dataset/ImageNet1K/val"


VISION_MODELS = [
    'resnet18',
    'vgg16',
    'densenet121',
    'vit_b_32',    
]


CLIP_MODELS = [
    'ViT-B_32',
    'ViT-B_16',
    'ViT-L_14',
]

CLIP_PARAMS = {
    'ViT-B_32': {
        'size': 224,
        'crop_size': 224,
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711]
    },
    
    'ViT-B_16': {
        'size': 224,
        'crop_size': 224,
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711]
    },
    
    
    'ViT-L_14': {
        'size': 224,
        'crop_size': 224,
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711]
    }
    
    # ....

    
 
}


OPENCLIP_MODELS = [
    'ViT-B_32',
    'ViT-B_16',
    'ViT-L_14',
]


OPENCLIP_PARAMS = {
    'ViT-B_32': {
        'size': 224,
        'crop_size': 224,
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711]
    },
    
    'ViT-B_16': {
        'size': 224,
        'crop_size': 224,
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711]
    },
    
    
    'ViT-L_14': {
        'size': 224,
        'crop_size': 224,
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711]
    }
    
    # ....

    
 
}


SIGLIP_MODELS = [
    "google/siglip-base-patch16-224",
    "google/siglip-large-patch16-256",
]


SIGLIP_PARAMS = {
    "google/siglip-base-patch16-224": {
        "size": 224,
        "crop_size": 224,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    },
    "google/siglip-large-patch16-256": {
        "size": 256,
        "crop_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5] 
    }
}
    


