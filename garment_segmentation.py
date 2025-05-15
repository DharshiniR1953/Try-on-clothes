import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from u2net import U2NET
import sys
sys.path.append("D:\Dharshini_ZZZ\Virtual_try_on_skratch\U-2-Net\model\u2net.py")

from u2net import U2NET

model = U2NET()
model.load_state_dict(torch.load('u2net.pth'))
model.eval()

# Define the transform for input image
def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def segment_garment(image_path):
    # Apply transformations and run through the model
    input_image = transform_image(image_path)
    with torch.no_grad():
        output = model(input_image)  # Get the segmentation mask
    
    # Convert output to binary mask
    mask = output[0, 0].cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)  # Thresholding the mask

    # Convert mask to an image
    segmented_image = Image.fromarray(mask * 255)
    return segmented_image

# Example usage:
segmented_garment = segment_garment("C:\\Users\\Admin\\Downloads\\assets\\cloth\\02783_00.jpg")
segmented_garment.show()
