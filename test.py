from util import ImageNetVal
import json
dataset = ImageNetVal("E:\\ImageNet1K\\imagenet\\ImageNet1K\\val", transform=None)

image_net_classes = dataset.classes
with open("D:\\PosHoc-Sparse\\description\\imgnet1k_description.json", 'r') as f:
    class_prompts = list(json.load(f).keys())

union = set(image_net_classes) & set(class_prompts)
print("Union: ", len(union))

for cls_1, cls_2 in zip(image_net_classes, class_prompts):
    if cls_1 != cls_2:
        print(cls_2)
