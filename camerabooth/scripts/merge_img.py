from PIL import Image
import os

def merge_images_vertically(image1_path, image2_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    new_img = Image.new("RGB", (max(image1.width, image2.width), image1.height + image2.height))

    new_img.paste(image1, (0, 0))
    new_img.paste(image2, (0, image1.height))

    return new_img

def main():
    dir1 = '/home/xinyang/scratch/zelin_dev/threetothreed/camerabooth/log_debug_sc'
    dir2 = '/home/xinyang/scratch/zelin_dev/threetothreed/camerabooth/log_debug_sc_sc'

    images1 = sorted([f for f in os.listdir(dir1) if f.endswith(('jpg', 'jpeg', 'png'))])
    images2 = sorted([f for f in os.listdir(dir2) if f.endswith(('jpg', 'jpeg', 'png'))])

    merged_images = []

    for img1, img2 in zip(images1, images2):
        image1_path = os.path.join(dir1, img1)
        image2_path = os.path.join(dir2, img2)

        merged_image = merge_images_vertically(image1_path, image2_path)
        merged_images.append(merged_image)

    total_width = sum(img.width for img in merged_images)
    max_height = max(img.height for img in merged_images)

    final_image = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in merged_images:
        final_image.paste(img, (x_offset, 0))
        x_offset += img.width

    final_image.save('final_output_image.jpg')

if __name__ == "__main__":
    main()