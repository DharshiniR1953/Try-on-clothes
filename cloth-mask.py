import os
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from networks.u2net import U2NET 

device = 'cpu'  

base_dir = r'C:\Users\Home\Downloads\Virtual_try_on_skratch\Virtual_try_on_skratch'

image_dir = os.path.join(base_dir, 'assets', 'cloth')  
result_dir = os.path.join(base_dir, 'assets', 'cloth-mask') 
checkpoint_path = os.path.join(base_dir, 'cloth_segm_u2net_latest.pth') 

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def load_checkpoint_mgpu(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model

class Normalize_image(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)

    def __call__(self, image_tensor):
        return self.normalize_3(image_tensor)

def get_palette(num_cls):
    palette = [0] * (num_cls * 3)
    palette[1*3:1*3+3] = [255, 0, 0]  
    palette[2*3:2*3+3] = [0, 255, 0]   
    palette[3*3:3*3+3] = [0, 0, 255]   
    return palette

transforms_list = [transforms.ToTensor(), Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(device)
net = net.eval()

palette = get_palette(4)

images_list = sorted(os.listdir(image_dir))
for image_name in images_list:
    img_path = os.path.join(image_dir, image_name)
    img = Image.open(img_path).convert('RGB')
    
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)

    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)

    print("Model output shape:", output_tensor[0].shape)  

    output_arr = output_tensor.cpu().numpy()
    print(f"{image_name}: unique classes in prediction = {np.unique(output_arr)}")

    output_img = Image.fromarray(output_arr.astype('uint8'), mode='L')
    output_img = output_img.resize(img_size, Image.BICUBIC)

    output_img.putpalette(palette)
    output_img = output_img.convert('RGB')  
    output_img.save(os.path.join(result_dir, image_name[:-4] + '_segmentation.jpg'))

print("Segmentation completed for all images.")
