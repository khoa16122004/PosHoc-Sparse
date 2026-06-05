import torch
import clip
import torch.nn.functional as F
from torchvision import transforms as T
from types import MethodType

class BaseWrapper:
    def __init__(self, model, normalize, device="cuda"):
        self.model = model.to(device)
        self.normalize = normalize
        self.device = device

    def predict(self, x):
        pass
    
    def set_posthoc_xai(self, method_name):
        self.method_name = method_name

    
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
        scores = logits[:, class_id].sum()    
        scores.backward()
        gradients = x.grad
        saliency = gradients.abs().sum(dim=1)   # B x w x h
        saliency = saliency / (
        saliency.mean(dim=(1,2), keepdim=True) + 1e-8) # B x w x h
        return logits, saliency.detach()
    
    def grad_input_explain(self, x, class_id):
        x = x.clone().detach() # B x 3 x w x h
        x.requires_grad = True
        logits = self.predict(x)
        scores = logits[:, class_id].sum()    
        scores.backward()
        gradients = x.grad
        saliency = gradients * x
        saliency = saliency.abs().sum(dim=1)    # B x w x h
        saliency = saliency / (
        saliency.mean(dim=(1,2), keepdim=True) + 1e-8) # B x w x h
        return logits, saliency.detach()
    
    def int_grad_explain(self, x, class_id, steps=50):
        x = x.clone().detach() # B x 3 x w x h
        baseline = torch.zeros_like(x).to(self.device)
        grads = torch.zeros_like(x)
        for i in range(steps):
            alpha = (i + 1) / steps
            inp = baseline + alpha * (x - baseline)
            inp = inp.detach().requires_grad_(True)
            logits = self.predict(inp)
            scores = logits[:, class_id].sum()  
            scores.backward()
            grads += inp.grad.detach()
            inp.grad.zero_()
            
        avg_grads = grads / steps
        ig = x * avg_grads
        saliency = ig.abs().sum(dim=1)    # B x w x h
        saliency = saliency / (
        saliency.mean(dim=(1,2), keepdim=True) + 1e-8) # B x w x h
        return logits, saliency.detach()
    
    def gradcam_explain(self, x, class_id):
        pass
        


class VisionModelWrapper(BaseWrapper):
    "Vison Wrapper for vision-only models, e.g., ResNet, etc."
    
    def __init__(self, model, normalize, device='cuda'):
        super().__init__(model, normalize, device=device)
        
    def predict(self, x):
        x = self.normalize(x)
        logits = self.model(x)
        return logits
    
class VisionViTModelWrapper(VisionModelWrapper):
    "Vison Wrapper for vision ViT, etc."
    
    def __init__(self, model, normalize, device='cuda'):
        super().__init__(model, normalize, device=device)
        self.vit_discard_ratio = 0.9
        self._attention_buffers = []
        self._init_attention_hooks()
        
        
    def predict_and_map(self, x, class_id=None):
        if self.method_name == "attn_grad":
            logits, saliency = self.attention_grad(x, class_id)
        else:
            return super().predict_and_map(x, class_id)

        return logits, saliency

    def _init_attention_hooks(self):
        encoder = getattr(self.model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is None:
            return

        for layer in layers:
            attention_module = getattr(layer, "self_attention", None)
            if attention_module is None or hasattr(attention_module, "_posthoc_original_forward"):
                continue

            original_forward = attention_module.forward

            def forward_with_weights(module, *args, _original_forward=original_forward, **kwargs):
                kwargs.setdefault("need_weights", True)
                kwargs.setdefault("average_attn_weights", False)
                return _original_forward(*args, **kwargs)

            attention_module._posthoc_original_forward = original_forward
            attention_module.forward = MethodType(forward_with_weights, attention_module)
            attention_module.register_forward_hook(self._capture_attention_weights)

    def _capture_attention_weights(self, module, inputs, output):
        if not isinstance(output, tuple) or len(output) < 2:
            return

        attention_weights = output[1]
        if attention_weights is None:
            return

        attention_weights.retain_grad()
        self._attention_buffers.append(attention_weights)

    def _attention_grad_rollout(self, image_size):
        if not self._attention_buffers:
            raise RuntimeError("No ViT attention weights were captured for gradient rollout.")

        attentions = list(self._attention_buffers)
        batch_size = attentions[0].size(0)
        num_tokens = attentions[0].size(-1)
        result = torch.eye(num_tokens, device=attentions[0].device).unsqueeze(0).repeat(batch_size, 1, 1)

        for attention in attentions:
            grad = attention.grad
            if grad is None:
                raise RuntimeError("Missing gradients for ViT attention weights.")

            fused_attention = (attention * grad).mean(dim=1)
            fused_attention = fused_attention.clamp(min=0)

            flat = fused_attention.view(batch_size, -1)
            discard_count = int(flat.size(-1) * self.vit_discard_ratio)
            if discard_count > 0:
                _, indices = flat.topk(discard_count, dim=-1, largest=False)
                discard_mask = torch.zeros_like(flat, dtype=torch.bool)
                discard_mask.scatter_(1, indices, True)
                discard_mask[:, 0] = False
                flat = flat.masked_fill(discard_mask, 0)
                fused_attention = flat.view_as(fused_attention)

            identity = torch.eye(num_tokens, device=fused_attention.device).unsqueeze(0)
            fused_attention = (fused_attention + identity) / 2
            fused_attention = fused_attention / fused_attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            result = torch.matmul(fused_attention, result)

        saliency = result[:, 0, 1:]
        width = int(saliency.size(-1) ** 0.5)
        saliency = saliency.view(batch_size, 1, width, width)
        saliency = F.interpolate(saliency, size=image_size, mode="bilinear", align_corners=False).squeeze(1)
        saliency = saliency / (saliency.mean(dim=(1, 2), keepdim=True) + 1e-8)
        return saliency.detach()

    
    def attention_grad(self, x, class_id):
        x = x.clone().detach()
        self._attention_buffers = []
        self.model.zero_grad()

        logits = self.predict(x)
        if class_id is None:
            target_scores = logits.gather(1, logits.argmax(dim=1, keepdim=True)).sum()
        else:
            target_scores = logits[:, class_id].sum()
        target_scores.backward()

        saliency = self._attention_grad_rollout(x.shape[-2:])
        return logits, saliency
    


    
           
          
        
        
    
    
class VLModelWrapper(BaseWrapper):
    def __init__(self, model, normalize, class_prompts, tokenizer=None, device='cuda'):
        super().__init__(model, normalize, device=device)
        self.class_prompts = class_prompts
        self.tokenizer = tokenizer
    
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

    