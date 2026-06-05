import torch

class VisionModelWrapper:
    "Vison Wrapper for vision-only models, e.g., ResNet, ViT, etc."
    
    
    def __init__(self, model, normalize, device='cuda'):
        self.model = model.to(device)
        self.normalize = normalize
        self.device = device
        
    def set_posthoc_xai(self, method_name):
        pass

    def predict(self, x):
        x = self.normalize(x)
        logits = self.model(x)
        return logits
    
    def predict_and_map(self, x, class_id=None):
        if self.method_name == "Grad":
            logits, saliency = self.grad_explain(x, class_id)
        elif self.method_name == "Grad_Input":
            logits, saliency = self.grad_input_explain(x, class_id)
        elif self.method_name == "Int_Grad":
            logits, saliency = self.int_grad_explain(x, class_id)
        elif self.method_name == "GradCAM":
            logits, saliency = self.gradcam_explain(x, class_id)
        
        return logits, saliency
        
    def grad_explain(self, x, class_id):
        x = x.clone().detach() # B x 3 x w x h
        x.requires_grad = True
        logits = self.predict(x)
        scores = logits[:, class_id]
        scores.backward()
        gradients = x.grad
        saliency = gradients.abs().sum(dim=1)    # B x w x h
        saliency = saliency / (
        saliency.mean(dim=(1,2), keepdim=True) + 1e-8)
        return logits, saliency.detach()
    
    def grad_input_explain(self, x, class_id):
        logits, saliency = self.grad_explain(x, class_id)
        return logits, saliency * x
    
    def int_grad_explain(self, x, class_id, steps=50):
        pass
    
    def gradcam_explain(self, x, class_id):
        pass
        
           
          
        
        
    
    
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
        image_features = self.model.get_image_features(pixel_values=x).pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def text_encode(self, t):
        inputs = self.tokenizer(t, padding="max_length", truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_features = self.model.get_text_features(**inputs).pooler_output # pooler or last hidden state
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach().cpu()
    

class GradExplain:
    def __init__(self):
        pass
    
    def method_explain(self, x, class_id):
        x = x.clone().detach()
        x.requires_grad = True
        
        
        pass
    
    