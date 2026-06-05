from util import get_torchvision_model
from wrapper import VisionModelWrapper
from PIL import Image
import torch
import torchvision
from matplotlib import pyplot as plt

model, spatial, normalize = get_torchvision_model('resnet18')
model = VisionModelWrapper(model, normalize)
model.set_posthoc_xai(
    "Int_Grad"
)

img = Image.open(
    "imgs/tench.jpg"
)

img = spatial(img).unsqueeze(0).cuda()
# save image
torchvision.utils.save_image(img, "input.png")

imgs = img.repeat(2,1,1,1)

logits, saliency = model.predict_and_map(imgs, class_id=0)
print(saliency.shape)
sal = saliency[0].detach().cpu().numpy()

plt.imshow(sal, cmap='hot')
plt.axis('off')
plt.savefig(
    "saliency.png",
    bbox_inches='tight',
    pad_inches=0
)