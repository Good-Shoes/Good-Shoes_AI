import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from utils import init_weights
from model import Generator

image_dir = './input/input.jpg'

input_dir = './input'
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Generator()
init_weights(model)
model.load_state_dict(torch.load("./netG_epoch_2_200.pth", map_location=device))
model.eval()
model.to(device)

result_dir = './test'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean

image = Image.open(image_dir)
image = transforms.ToTensor()(image)
image = torch.unsqueeze(image[:, :, 256:], 0)

#transform = transforms.Compose()

output = model(image)
save_image(output, './test/input_2_200.png')


