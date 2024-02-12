import os
import shutil

def copy_mp4_files(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print("Input directory does not exist.")
        return

    # Create the output directory structure
    for root, dirs, files in os.walk(input_dir):
        # Determine the path to this directory relative to the input directory
        rel_path = os.path.relpath(root, input_dir)
        # Construct the corresponding path in the output directory
        dest_dir = os.path.join(output_dir, rel_path)
        # Create the directory in the output directory
        os.makedirs(dest_dir, exist_ok=True)

        # Copy mp4 files
        for file in files:
            if file.lower().endswith('.mp4'):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)
                shutil.copy2(src_file, dest_file)

# Example usage:
for i in range(1,6):
    input_dir = f'/shared/xinyang/threetothreed/test_recon/threestudio/experiments_GSO_demo_view_{i}_nerf'
    output_dir = input_dir + '_summarized'

    # Call the function to copy the mp4 files
    copy_mp4_files(input_dir, output_dir)


print("Copying of MP4 files is completed.")
