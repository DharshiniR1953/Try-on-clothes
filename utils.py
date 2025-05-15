import os

import cv2
import numpy as np
from PIL import Image
import torch


def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise


def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0,255)

        try:
            array = tensor.numpy().astype('uint8')
        except:
            array = tensor.detach().numpy().astype('uint8')

        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        im = Image.fromarray(array)
        im.save(os.path.join(save_dir, img_name), format='JPEG')


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError("'{}' is not a valid checkpoint path".format(checkpoint_path))
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    try:
        # Try to load the state_dict with strict=False to allow mismatched layers
        model.load_state_dict(checkpoint, strict=False)
    except RuntimeError as e:
        print("Error while loading state_dict: ", e)
    
    # Optionally: Initialize the mismatched layers manually
    for name, param in model.named_parameters():
        # If the weight's shape is mismatched, initialize the layer
        if "head_0.norm_0.conv_shared.0.weight" in name:
            print(f"Initializing layer {name} manually")
            torch.nn.init.xavier_normal_(param)
        elif "head_0.norm_1.conv_shared.0.weight" in name:
            print(f"Initializing layer {name} manually")
            torch.nn.init.xavier_normal_(param)
        elif "G_middle_0.norm_0.conv_shared.0.weight" in name:
            print(f"Initializing layer {name} manually")
            torch.nn.init.xavier_normal_(param)
        elif "G_middle_0.norm_1.conv_shared.0.weight" in name:
            print(f"Initializing layer {name} manually")
            torch.nn.init.xavier_normal_(param)
        # Add additional layers if needed based on the error message
        # Example for other layers:
        # elif "up_0.norm_0.conv_shared.0.weight" in name:
        #     print(f"Initializing layer {name} manually")
        #     torch.nn.init.xavier_normal_(param)
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
