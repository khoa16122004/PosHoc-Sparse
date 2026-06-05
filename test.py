from util import get_torchvision_model
from wrapper import VisionModelWrapper
from PIL import Image
import torch
import torchvision
from matplotlib import pyplot as plt

explain_methods = [
    'Grad',
    'Grad_Input',
    'Int_Grad',
    # 'GradCAM'
]

model, spatial, normalize = get_torchvision_model('densenet121')
model = VisionModelWrapper(model, normalize)
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