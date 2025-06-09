# Takes the dataset PNGs and expands them to a multiple of 100x100 pixels, then cuts them into individual 100x100 squares.

WIDTH = 100

import os
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

# Dates used by the dataset are usually in the format YYYYMMDD
date_format_str = '%Y-%m-%d-%H%M'

# Parse /dataset/coordinates folder

# Fires is a dict of each fire, with each fire being a 
# list of the files associated with that fire, sorted
# by date
fires = {}

for dirpath, dirnames, filenames in os.walk('organized_dataset'):
    for filename in filenames:
        # Search for files with the .png extension
        if filename.endswith('.png'):
            file = os.path.join(dirpath, filename)
            # Insert into fires list
            # Search for last list that contains the same beginning filepath
            fire_found = False
            for fire in fires.keys():
                # Skip any 'squares' folders
                if 'squares' in file.split('/'):
                    continue
                if file.startswith(fire):
                    fires[fire].append(file)
                    fire_found = True
                    break

            # In the case of a new fire being found, add it to the fires dict
            if not fire_found:
                file_dir = file.split('/')
                
                for i in range(len(file_dir)):
                    
                    # All days for a fire are the same except for after the 'IR' or 'KML_KMZ' folder
                    if file_dir[i].upper() in ['IR', 'KML_KMZ', 'KMZ', 'GIS']:
                        file_dir = file_dir[:i]
                        break
                    else:
                        try:
                            # See if the current string is a date, at which point the fire name is found
                            datetime.strptime(file_dir[i], date_format_str)
                            file_dir = file_dir[:i]
                            break
                        except ValueError:
                            continue

                fires['/'.join(file_dir)] = [file]

def expand_image(image, width):
    # Calculate the new dimensions
    bigger_dimension = max(image.shape[0], image.shape[1])

    new_width = ((bigger_dimension // width) + 1) * width

    # Expand the image to the new dimensions equally on all 4 sides
    expanded_image = np.zeros((new_width, new_width, 4), dtype=image.dtype)

    x_offset = (new_width - image.shape[0]) // 2
    y_offset = (new_width - image.shape[1]) // 2
   
    expanded_image[x_offset:x_offset + image.shape[0], y_offset:y_offset + image.shape[1]] = image

    return expanded_image

def cut_into_squares(image, width):
    # Calculate the number of squares in each dimension
    num_squares_x = image.shape[0] // width
    num_squares_y = image.shape[1] // width

    # Cut the image into squares
    squares = []
    for i in range(num_squares_x):
        for j in range(num_squares_y):
            square = image[i * width:(i + 1) * width, j * width:(j + 1) * width]
            squares.append(square)

    return squares

print("Expanding and cutting images...")

for fire_name, data in tqdm(fires.items()):
    for file_name in data:
        # Load the image
        image = Image.open(file_name)
        np_image = np.array(image)
        image_expanded = expand_image(np_image, WIDTH)
        pil_image = Image.fromarray(image_expanded)

        # Create folder for image
        folder = f'{file_name.split('.')[0]}/squares'
        os.makedirs(folder, exist_ok=True)

        # Cut the image into squares
        squares = cut_into_squares(image_expanded, WIDTH)

        # Save squares into folder
        for i, square in enumerate(squares):
            square = Image.fromarray(square)
            square.save(os.path.join(folder, f'square_{i+1},{len(squares)}.png'))
