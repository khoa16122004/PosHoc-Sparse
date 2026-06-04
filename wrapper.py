import torch
from util import split_transform_from_weights, split_VLMs_transform
from torchvision.models import get_model_weights
import clip

class VisionModelWrapper:
    "Vison Wrapper for vision-only models, e.g., ResNet, ViT, etc."
    
    
    def __init__(self, model, normalize, device='cuda'):
        self.model = model.to(device)
        self.normalize = normalize
        self.device = device

    def predict(self, x):
        x = self.normalize(x)
        logits = self.model(x)
        return logits
    
    
class VLModelWrapper:
    def __init__(self, model, normalize, class_prompts, tokenizer=None, device='cuda'):
        self.model = model.to(device)
        self.normalize = normalize
        self.class_prompts = class_prompts
        self.tokenizer = tokenizer
        self.device = device
    
    def set_fodler_class(self, folder_class_list, folder_2_class_name):
        self.folder_class_list = folder_class_list
        self.folder_2_class_name = folder_2_class_name
    
    def extract_class_text_features(self):
        textual_class_features = []
        # print("Extract class_text_features...")
        for class_name in self.folder_class_list:
            class_real_name = self.folder_2_class_name[class_name][1].replace("_", " ")
            prompts_ = self.class_prompts[class_name]
            prompts = [
                f"{class_real_name} which has {prompt}" for prompt in prompts_
            ]
            textual_class_features.append(self.text_encode(prompts).mean(dim=0))        
        
        self.class_text_features = torch.stack(textual_class_features).to(self.device)
        # print("Class text feautures shape: ", self.class_text_features.shape)
        
    def predict(self, x):
        x = self.normalize(x)
        visual_features = self.vision_encode(x)
        logits = visual_features @ self.class_text_features.T       
        return logits
    
    
    def vision_encode(self, x):
        vision_features = self.model.encode_image(x)
        vision_features = vision_features / vision_features.norm(dim=-1, keepdim=True)
        return vision_features
    
    def text_encode(self, t): # t lists
        if self.tokenizer is not None:
            t = self.tokenizer(t).cuda(self.device)
        else:
            t = clip.tokenize(t).cuda(self.device)
        text_features = self.model.encode_text(t)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach().cpu()
    
    
class SIGLIPWrapper(VLModelWrapper):
    def __init__(self, model, normalize, class_prompts, tokenizer=None, device='cuda'):
        super().__init__(model, normalize, class_prompts, tokenizer=tokenizer, device=device)

    def vision_encode(self, x):
        image_features = self.model.get_image_features(**x, return_dict=False)
        print(image_features)
        raise
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def text_encode(self, t):
        inputs = self.tokenizer(t, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_features = self.model.get_text_features(**inputs, return_dict=False)
        a, b = text_features
        print(a.shape)
        print(b.shape)
        print(text_features)
        raise
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach().cpu()
    
    
class BEIT3Wrapper:
    def __init__(self):
        pass
    
    def predict(self, x):
        pass

    def vision_encode(self, x):
        pass
    
    def text_encode(self, t):
        pass