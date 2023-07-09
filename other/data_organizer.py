import os

import shutil
from PIL import Image

# Directory paths
images_dir = './eyth_dataset/images'
masks_dir = './eyth_dataset/masks'

# Rename and move files in the images directory
for subdir, dirs, files in os.walk(images_dir):
    for filename in files:
        old_path = os.path.join(subdir, filename)
        new_filename = subdir.split(os.path.sep)[-1] + '_' + filename
        new_path = os.path.join(images_dir, new_filename)
        shutil.move(old_path, new_path)

# Convert and move files in the masks directory
for subdir, dirs, files in os.walk(masks_dir):
    for filename in files:
        # Skip processing .DS_Store files
        if filename == '.DS_Store':
            continue

        old_path = os.path.join(subdir, filename)

        # Check if the file extension is missing
        if '.' not in filename:
            filename = filename + '.png'

        new_filename = subdir.split(os.path.sep)[-1] + '_' + os.path.splitext(filename)[0] + '.jpg'
        new_path = os.path.join(masks_dir, new_filename)

        # Open the mask image using PIL
        mask = Image.open(old_path)

        # Convert the mask to RGB format
        mask_rgb = mask.convert('RGB')

        # Save the mask as a JPEG image
        mask_rgb.save(new_path, 'JPEG')

        print(f'Converted mask: {new_path}')