IMAGENET_PROMPT_PATH = "description/imagenet10k.json"



VISION_MODELS = [
    'resnet18',
    'vgg16',
    'densenet121',
    'ViT-B-32',    
]


CLIP_MODELS = [
    'ViT-B/32',
    'ViT-B/16',
    'ViT-L/14',
]

CLIP_PARAMS = {
    'ViT-B/32': {
        'size': 224,
        'crop_size': 224,
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711]
    },
    
    'ViT-B/16': {
        'size': 224,
        'crop_size': 224,
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711]
    },
    
    
    'ViT-L/14': {
        'size': 224,
        'crop_size': 224,
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711]
    }
    
    # ....

    
 
}


OPENCLIP_MODELS = [
    'ViT-B/32',
    'ViT-B/16',
    'ViT-L/14',
]



SIGLIP_MODELS = [
    "google/siglip-base-patch16-224",
    "google/siglip-large-patch16-256",
]




