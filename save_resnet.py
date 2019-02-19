import time

import torch
import torchvision

weight_file = './models/resnet.weight'

net = torchvision.models.resnet50(pretrained=False)
net.cuda()

net.load_state_dict(torch.load(weight_file))
net.eval()

example = torch.rand(8, 3, 224, 224).cuda()

with torch.no_grad():
    traced_script_module = torch.jit.trace(net, example)

traced_script_module.save('./models/resnet.pt')
