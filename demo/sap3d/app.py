from PIL import Image
import gradio as gr
import os
import shutil
import uuid
import subprocess
import glob
import pdb
from rembg import remove
import numpy as np


root_dir = os.environ.get('ROOT_DIR')
DEMO_PATH = f'{root_dir}/dataset/data/train/DEMO'
TEMPLATE_PATH = os.path.join(DEMO_PATH, 'TEMPLATE')
Render_3D = True
GPU_INDEX = 4
IMG_SIZE = 512

def preprocess_image(image):
    image.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    if image.mode == 'RGBA':
        no_bg_image = image
    else:
        image_rgb = image.convert('RGB')

        bg_removed_image = remove(image_rgb)

        no_bg_image = bg_removed_image.convert('RGBA')

    img_array = np.array(no_bg_image)
    alpha = img_array[:, :, 3]  

    nonzero_y, nonzero_x = np.nonzero(alpha)
    if nonzero_x.size == 0 or nonzero_y.size == 0:
        print("No foreground object detected.")
        return None, None

    x_min, x_max = nonzero_x.min(), nonzero_x.max()
    y_min, y_max = nonzero_y.min(), nonzero_y.max()
    center_x, center_y = (x_max + x_min) // 2, (y_max + y_min) // 2

    foreground_width = x_max - x_min
    foreground_height = y_max - y_min
    crop_size = max(foreground_width, foreground_height) / 0.8
    crop_size = min(crop_size, img_array.shape[1] * 0.8, img_array.shape[0] * 0.8)

    left = max(0, center_x - crop_size // 2)
    upper = max(0, center_y - crop_size // 2)
    right = min(img_array.shape[1], left + crop_size)
    lower = min(img_array.shape[0], upper + crop_size)

    cropped_image = no_bg_image.crop((left, upper, right, lower))
    cropped_mask = Image.fromarray(alpha).crop((left, upper, right, lower))

    final_image = cropped_image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    final_mask = cropped_mask.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    return final_image, final_mask

def find_last_log(dir_path):
    timestamp_dirs = glob.glob(os.path.join(dir_path, "*"))
    timestamp_dirs.sort(key=os.path.getmtime, reverse=True)
    if timestamp_dirs:
        latest_dir = timestamp_dirs[0]
        return latest_dir
    return None

def create_thumbnail(image_path, size=(256, 256)):
    img = Image.open(image_path).convert("RGBA")
    img.thumbnail(size, Image.ANTIALIAS)

    # Create a white background image
    background = Image.new('RGB', img.size, (255, 255, 255))
    # Paste the image on the white background
    background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
    return background

def prepare_dataset_structure(images):
    # Generate a unique directory name
    # random_dir_name = "test2"  # Use a fixed name for demonstration; in practice, consider using uuid4 for uniqueness
    random_dir_name = str(uuid.uuid4())
    base_dir = os.path.join(DEMO_PATH, random_dir_name)
    print(base_dir)
    images_dir = os.path.join(base_dir, 'images')
    masks_dir = os.path.join(base_dir, 'masks')

    # Create directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Save images and extract masks
    for i, img in enumerate(images):
        image_path = os.path.join(images_dir, f'{i:03}.png')
        mask_path = os.path.join(masks_dir, f'{i:03}.png')

        processed_image, mask = preprocess_image(img)
        processed_image.save(image_path)
        mask.save(mask_path)

    # Copy template files to the new directory
    for item in ['poses', 'query_class', 'valid_paths.json']:
        source_path = os.path.join(TEMPLATE_PATH, item)
        destination_path = os.path.join(base_dir, item)
        if os.path.isdir(source_path):
            # Check if the destination directory already exists
            if os.path.exists(destination_path):
                # If it does, remove it before copying
                shutil.rmtree(destination_path)
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination_path)

    return base_dir, random_dir_name

def process_images(uploaded_imgs):
    images = [Image.open(img) for img in uploaded_imgs]
    base_dir, random_dir_name = prepare_dataset_structure(images)
    print(f"Dataset prepared at: {base_dir}")

    bash_dir = os.path.join(root_dir, 'run_pipeline_demo.sh')
    img_num = len(images)
    print("img_num:", img_num)
    command = ["sh", bash_dir, "DEMO", random_dir_name, str(img_num), str(GPU_INDEX)]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode != 0:
        print("Script execution failed")
    else:
        print("Script executed successfully")
        
    class_name = "DEMO"

    if Render_3D:
        log_path = f'{root_dir}/3D_Recon/threestudio/experiments_{class_name}_view_{img_num}_nerf/{random_dir_name}_ours'
        print("log_path:", log_path)
        latest_dir = find_last_log(log_path)
        video_path = os.path.join(latest_dir, "save", "it4500-val.mp4")
    else:
        video_path = os.path.join(root_dir, "camerabooth", "experiments_nvs", "DEMO", f"{random_dir_name}_view_{img_num}", "preds", f"{random_dir_name}_view_{img_num}_sc.mp4")

    print("video_path:", video_path)
    assert os.path.exists(video_path), "Video path does not exist."

    return video_path


css = """
    button:enabled:hover {
        background-color: white;
        color: white;
    }
    button:enabled {
        background-color: white;
        color: white;
    }
"""


with gr.Blocks(css=css) as demo:
    gr.Markdown("# The More You See in 2D, the More You Perceive in 3D")
    gr.Markdown("""
    **Instructions:**
    - You can upload 3-5 images (**RGBD**) of your own unposed images of an object.
    - The reconstructed video will be shown on the right after processing (around an hour).
    - Enjoy reconstructing your own objects!
    """)

    with gr.Row():
        with gr.Column():
            file_input = gr.File(file_count="multiple", file_types=["image"], label="Upload your image(s)", visible=lambda choice: choice == 'Upload your own')
            image_display = gr.Gallery(
                label="Preprocessed Images",
                columns=4,  
                min_width=200,  
                height=300, 
                object_fit="cover", 
                elem_classes="custom-gallery-class", 
            )

        with gr.Column():
            video_output = gr.Video(label="Rendered video from reconstructed 3D NeRF.")


    def process_uploaded_files(files):
        processed_images = []
        if files is None:
            return processed_images  # No files were uploaded

        for file_path in files:
            if files is None:
                return []  
            img = Image.open(file_path)
            processed_image, _ = preprocess_image(img)
            processed_images.append(processed_image)

        return processed_images

    file_input.change(process_uploaded_files, inputs=file_input, outputs=image_display)
    
    process_button = gr.Button("Run")
    process_button.click(process_images, inputs=[file_input], outputs=video_output)

if __name__ == "__main__":
    demo.launch(share=True)