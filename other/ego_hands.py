import os
import cv2
import scipy.io as sio
import numpy as np

# Path to the Egohands dataset directory
# dataset_dir = './egohands_data'

# # Path to save the generated mask files
# os.makedirs('./ego_out',exist_ok=True)
# output_dir = './ego_out'
file_path = './egohands_data/labelled_samples/CARDS_COURTYARD_B_T/polygons.mat'

polygon_data = sio.loadmat(file_path)
print(polygon_data.)

# # Iterate through each subfolder in the labeled samples directory
# for subdir in os.listdir(os.path.join(dataset_dir, 'labelled_samples')):
#     print('hi')
#     subfolder_path = os.path.join(dataset_dir, 'labelled_samples', subdir)

#     if not os.path.isdir(subfolder_path):
#         continue

#     # Find the polygon.mat file in the subfolder
#     polygon_file = os.path.join(subfolder_path, 'polygon.mat')

#     if not os.path.isfile(polygon_file):
#         continue

#     # Load the polygon annotations from the .mat file
#    #    

#     polygons = polygon_data['polygons']

#     # Iterate through each polygon annotation
#     for i, polygon in enumerate(polygons):
#         # Create a blank mask image
#         mask = np.zeros_like(cv2.imread(os.path.join(subfolder_path, '000000.jpg'), cv2.IMREAD_COLOR))

#         # Fill the polygon region with white color in the mask image
#         cv2.fillPoly(mask, [polygon], (255, 255, 255))

#         # Save the mask as an image file
#         output_mask_path = os.path.join(output_dir, subdir + f'_mask{i:03d}.png')
#         cv2.imwrite(output_mask_path, mask)

#         print(f'Saved mask: {output_mask_path}')
