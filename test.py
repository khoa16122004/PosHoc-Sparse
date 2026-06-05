from util import get_torchvision_model
from wrapper import VisionModelWrapper
from PIL import Image
import torch
import torchvision

model, spatial, normalize = get_torchvision_model('resnet18')
model = VisionModelWrapper(model, normalize)

img = Image.open(
    "imgs/tabby.jpg"
)

img = spatial(img).unsqueeze(0)

logits, saliency = model.predict_and_map(img, class_id=283)

print(torch.argmax(logits, dim=1))
# save saliency, torchvision.utils.save_image(saliency, "saliency.png")
# 1 x 1 x w x h
torchvision.utils.save_image(saliency, "saliency.png")
