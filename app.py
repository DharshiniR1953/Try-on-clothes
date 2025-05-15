import os
import os
import gradio as gr
import shutil
from PIL import Image
import subprocess
import numpy as np

base_dir = r"C:\Users\Home\Downloads\Virtual_try_on_skratch\Virtual_try_on_skratch"
cloth_dir = os.path.join(base_dir, "assets", "cloth")
image_dir = os.path.join(base_dir, "assets", "image")
test_pairs_path = os.path.join(base_dir, "assets", "test_pairs.txt")
output_dir = os.path.join(base_dir, "output")

def find_image_file(base_path, filename_wo_ext):
    """
    Tries to find the image file with .jpg, .jpeg, or .png extension.
    Returns the full path if found, else raises an error.
    """
    for ext in ['.jpg', '.jpeg', '.png']:
        candidate = os.path.join(base_path, f"{filename_wo_ext}{ext}")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No image found for {filename_wo_ext} with .jpg/.png/.jpeg in {base_path}")

def clear_dirs():
    for path in [cloth_dir, image_dir, output_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

def save_and_preprocess(model_img, cloth_img):
    clear_dirs()

    if isinstance(model_img, np.ndarray):
        model_img = Image.fromarray(np.uint8(model_img))
    if isinstance(cloth_img, np.ndarray):
        cloth_img = Image.fromarray(np.uint8(cloth_img))

    model_path = os.path.join(image_dir, "model.jpg")
    cloth_path = os.path.join(cloth_dir, "cloth.jpg")
    model_img.save(model_path)
    cloth_img.save(cloth_path)

    subprocess.run(["python", "cloth-mask.py"], cwd=base_dir, shell=True)

    cloth_resized_path = os.path.join(base_dir,"assets","cloth-mask", "cloth_segmentation.jpg")
    if not os.path.exists(cloth_resized_path):
        return f"Error: Resized cloth not found at {cloth_resized_path}"

    subprocess.run(["python", "remove_bg.py"], cwd=base_dir, shell=True)

    model_bg_removed_path = os.path.join(base_dir, "assets", "image-nobg", "model_nobg_transformed.jpg")
    if not os.path.exists(model_bg_removed_path):
        return f"Error: Background removed model not found at {model_bg_removed_path}"

    subprocess.run([
        "python", os.path.join("Self-Correction-Human-Parsing", "simple_extractor.py"),
        "--dataset", "lip",
        "--model-restore", os.path.join("Self-Correction-Human-Parsing", "checkpoints", "final.pth"),
        "--input-dir", "assets/image-nobg",
        "--output-dir", "Self-Correction-Human-Parsing/segmentation-results"
    ], cwd=base_dir, shell=True)

    parsed_img_path = os.path.join(base_dir, "Self-Correction-Human-Parsing", "segmentation-results", "model.png")
    if not os.path.exists(parsed_img_path):
        return f"Error: Human parsing result not found at {parsed_img_path}"

    subprocess.run(["python", "generate_keypoints.py"], cwd=base_dir, shell=True)

    keypoints_path = os.path.join(base_dir, "assets", "pose", "model_keypoints.json")
    if not os.path.exists(keypoints_path):
        return f"Error: Pose keypoints not generated at {keypoints_path}"

    with open(test_pairs_path, "w") as f:
        f.write("model.jpg cloth.jpg\n")

    subprocess.run([
        "python", "test1.py",
        "--dataset_dir", base_dir,
        "--checkpoint_dir", os.path.join(base_dir, "checkpoints"),
        "--save_dir", output_dir
    ], cwd=base_dir, shell=True)

    try:
        result_path = find_image_file(os.path.join(output_dir), "model_cloth")
        result_img = Image.open(result_path)
        return result_img
    except FileNotFoundError as e:
        return f"Error: Output not generated! Details: {str(e)}"

import gradio as gr

with gr.Blocks() as interface:
    gr.Markdown("## Virtual Try-On")
    gr.Markdown("Upload a person image and a cloth image to visualize the try-on result.")

    with gr.Row():
        person_input = gr.Image(label="Upload Person Image")
        cloth_input = gr.Image(label="Upload Cloth Image")

    output_image = gr.Image(label="Virtual Try-On Output", type="pil")

    submit_button = gr.Button("Generate")

    submit_button.click(
        fn=save_and_preprocess,
        inputs=[person_input, cloth_input],
        outputs=output_image
    )

interface.launch(debug=True, share=True)
