import re
import os
import shutil


def extract_numbers_from_string(string):
    ranges = re.findall(r"(\d+)-(\d+)", string)
    numbers = []
    for start, stop in ranges:
        numbers.extend(range(int(start), int(stop) + 1))
    return numbers


def move_files(source_folder, destination_folders, file_extensions):
    os.makedirs(destination_folders['other'], exist_ok=True)
    for filename in os.listdir(source_folder):
        if filename.endswith(file_extensions):
            file_number = int(re.search(r"\d+", filename).group())
            for category, folder in destination_folders.items():
                if file_number in category:
                    shutil.move(os.path.join(source_folder, filename), os.path.join(folder, filename))
                    break
            else:
                shutil.move(os.path.join(source_folder, filename), os.path.join(destination_folders['other'], filename))


adding_things = "11-41,129-157,116-189,195-216,227-240,372-393,434-461,468-490,498-520,530-555,565-598,708-734,763-788,906-922"
stiring = "55-85,94-117,266-303,410-425,618-653,808-867,940-970,1188-1200"

add_frames = extract_numbers_from_string(adding_things)
stir_frames = extract_numbers_from_string(stiring)

source_folder = "./results/video_2/masked_image"
destination_folders = {
    'add': "./results/video_2/adding",
    'stir': "./results/video_2/stiring",
    'other': "./results/video_2/other"
}
file_extensions = (".jpg", ".png")

move_files(source_folder, destination_folders, file_extensions)

source_folder = "./results/video_2/predicted_mask"
destination_folders = {
    'add': "./results/video_2/adding_mask",
    'stir': "./results/video_2/stiring_mask",
    'other': "./results/video_2/other_mask"
}

move_files(source_folder, destination_folders, file_extensions)
