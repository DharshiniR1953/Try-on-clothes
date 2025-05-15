import os
from PIL import Image
import numpy as np
from rembg import remove


class PreprocessInput:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def remove_bg(self, file_path: str):
        base_name = os.path.splitext(os.path.basename(file_path))[0] 
        self.png_path = os.path.join(self.output_dir, base_name + '_nobg.png')

        pic = Image.open(file_path).convert("RGBA")
        self.o_image = remove(pic)
        self.o_image.save(self.png_path)
        print(f"✅ Saved transparent PNG: {self.png_path}")
        return np.asarray(self.o_image)

    def transform(self, width=768, height=1024):
        img = self.o_image.resize((width, height))
        background = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])

        base_name = os.path.splitext(os.path.basename(self.png_path))[0]
        jpeg_path = os.path.join(self.output_dir, base_name + '_transformed.jpg')
        background.convert('RGB').save(jpeg_path, 'JPEG')
        print(f"✅ Saved transformed JPEG: {jpeg_path}")
        return np.asarray(background.convert('RGB'))


input_dir = r"C:\Users\Home\Downloads\Virtual_try_on_skratch\Virtual_try_on_skratch\assets\image"
output_dir = r"C:\Users\Home\Downloads\Virtual_try_on_skratch\Virtual_try_on_skratch\assets\image-nobg"

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(input_dir, filename)
        print(f"🚀 Processing: {file_path}")
        processor = PreprocessInput(output_dir)
        processor.remove_bg(file_path)
        processor.transform(768, 1024)
