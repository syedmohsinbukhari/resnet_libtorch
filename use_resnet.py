import time

import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

weight_file = './models/resnet.weight'

net = torchvision.models.resnet50(pretrained=False)
net = net.cuda()

net.load_state_dict(torch.load(weight_file))
net.eval()

inp_img = cv2.imread('./imgs/jif2.jpg', 1)

inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)

# Apply standard transforms to image and convert to Tensor
resnet_eval_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

pil_img = Image.fromarray(inp_img)
img_tensor = resnet_eval_transform(pil_img)
img_tensor = img_tensor.unsqueeze(0)
img_tensor = img_tensor.cuda()

with torch.no_grad():
    for i in range(10):
        start = time.time()
        output_probabilities = net(img_tensor)
        end = time.time()

    value, index = torch.max(output_probabilities, 1)
    print(f"value: {value}")
    print(f"index: {index}")
    print(f"time taken: {round(1000*(end-start))} ms")
