from util import get_torchvision_model
from wrapper import VisionModelWrapper
from PIL import Image
import torch
import torchvision
from matplotlib import pyplot as plt

model, spatial, normalize = get_torchvision_model('resnet18')
model = VisionModelWrapper(model, normalize)
model.set_posthoc_xai(
    "Grad"
)

img = Image.open(
    "imgs/tabby.jpg"
)

img = spatial(img).unsqueeze(0).cuda()
imgs = img.repeat(2,1,1,1)

logits, saliency = model.predict_and_map(imgs, class_id=281)
print(saliency.shape)
sal = saliency[0,0].detach().cpu().numpy()
plt.axis('off')
plt.savefig(
    "saliency.png",
    bbox_inches='tight',
    pad_inches=0
)