from PIL import Image
import os
import shutil
import subprocess

def resize_img(path):
    im = Image.open(path)
    im = im.resize((768, 1024))
    im.save(path)

# Base path
base_dir = r'C:\Users\Home\Downloads\Virtual_try_on_skratch\Virtual_try_on_skratch'
cloth_dir = r'C:\Users\Home\Downloads\Virtual_try_on_skratch\Virtual_try_on_skratch\assets\cloth'
image_dir = r'C:\Users\Home\Downloads\Virtual_try_on_skratch\Virtual_try_on_skratch\assets\image'

for filename in os.listdir(cloth_dir):
    file_path = os.path.join(cloth_dir, filename)
    if os.path.isfile(file_path):
        resize_img(file_path)

checkpoints_dir = os.path.join(cloth_dir, '.ipynb_checkpoints')
if os.path.exists(checkpoints_dir):
    shutil.rmtree(checkpoints_dir)

subprocess.run(['python', 'cloth-mask.py'], cwd=base_dir, shell=True)

print()
print("Cloth masking is done")
print()

subprocess.run(['python', 'remove_bg.py'], cwd=base_dir, shell=True)

print()
print("background removal process is done")
print()

parse_script = os.path.join(base_dir, 'Self-Correction-Human-Parsing', 'simple_extractor.py')
model_restore_path = os.path.join(base_dir, 'Self-Correction-Human-Parsing', 'checkpoints', 'final.pth')
output_parse_dir = os.path.join(base_dir, 'assets', 'image-parse')

subprocess.run([
    'python', parse_script,
    '--dataset', 'lip',
    '--model-restore', model_restore_path,
    '--input-dir', image_dir,
    '--output-dir', output_parse_dir
], shell=True)

print()
print("openpose key points generation initiated")
print()

subprocess.run(['python', 'generate_keypoints.py'], cwd=base_dir, shell=True)

print()
print("MediaPipe keypoints generation completed.")
print()

model_images = os.listdir(image_dir)
cloth_images = os.listdir(cloth_dir)
pairs = zip(model_images, cloth_images)

test_pairs_path = os.path.join(base_dir, 'assets', 'test_pairs.txt')
with open(test_pairs_path, 'w') as file:
    for model, cloth in pairs:
        file.write(f"{model} {cloth}\n")

subprocess.run([
    'python', 'test1.py',
    '--name', 'output',
    '--dataset_dir', base_dir,
    '--checkpoint_dir', os.path.join(base_dir, 'checkpoints'),
    '--save_dir', base_dir
], cwd=base_dir, shell=True)

# Step 9: Clean up input and intermediate directories
# if os.path.exists(image_dir):
#     shutil.rmtree(image_dir)

output_checkpoints = os.path.join(base_dir, 'assets', 'output', '.ipynb_checkpoints')
if os.path.exists(output_checkpoints):
    shutil.rmtree(output_checkpoints)
