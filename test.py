from PIL import Image
import torch
import torchvision
from matplotlib import pyplot as plt
from util import ImageNetVal, get_CLIP_model, get_OPENCLIP_model, get_SIGLIP_model, get_torchvision_model
from wrapper import SIGLIPWrapper, VisionModelWrapper, VLModelWrapper
from constant import DEFAULT_VAL_DIR, IMAGENET_PROMPT_PATH, IMAGENET_FOLDER2_CLASSNAME
from torch.utils.data import DataLoader
import json


with open(IMAGENET_PROMPT_PATH, 'r') as f:
    class_prompts = json.load(f)    
with open(IMAGENET_FOLDER2_CLASSNAME, 'r') as f:
    folder_2_class_name = json.load(f)

explain_methods = [
    'Grad',
    'Grad_Input',
    'Int_Grad',
    # 'GradCAM'
]
type = "CLIP"

# vision
if type == "torchvision":
    model, spatial, normalize = get_torchvision_model("resnet18")
    model = VisionModelWrapper(model, normalize)
    
elif type == "CLIP":
    model, spatial, normalize = get_CLIP_model("ViT-B_32")
    model = VLModelWrapper(model, normalize, class_prompts)

    
elif type == "OPENCLIP":
    model, spatial, normalize, tokenizer = get_OPENCLIP_model("ViT-B_32")
    model = VLModelWrapper(model, normalize, class_prompts, tokenizer)

elif type == "SIGLIP":
    model, spatial, normalize, tokenizer = get_SIGLIP_model("google/siglip-base-patch16-224")
    model = SIGLIPWrapper(model, normalize, class_prompts, tokenizer)






# Dataset and dataloader    
dataset = ImageNetVal(DEFAULT_VAL_DIR, transform=spatial)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
folder_class_list = dataset.classes
if type in ["CLIP", "OPENCLIP", "SIGLIP"]:
    model.set_fodler_class(folder_class_list, folder_2_class_name)
    model.extract_class_text_features()
    

img = Image.open(
    "imgs/tench.jpg"
)

img = spatial(img).unsqueeze(0).cuda()
imgs = img.repeat(2,1,1,1)
torchvision.utils.save_image(img, "input.png")

for method in explain_methods:
    model.set_posthoc_xai(method)

    print(f"Running {method}...")

    logits, saliency = model.predict_and_map(imgs, class_id=0)
    sal = saliency[0].detach().cpu().numpy()

    plt.imshow(sal, cmap='hot')
    plt.axis('off')
    plt.savefig(
        f"saliency_{method}.png",
        bbox_inches='tight',
        pad_inches=0
    )