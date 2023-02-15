# This script creates a json file listing images by pairs

import os
import json

# Get all files in the folder
path = "./"
files = os.listdir(path)

# Sort the files alphabetically
files.sort()

# Create a list of images
images = []

# Loop through all files
for file in files:
    # Get the file extension
    file_extension = os.path.splitext(file)[1]
    # Check if the file is an image
    if file_extension in [".jpg", ".jpeg", ".png"]:
        # Add the image to the list
        images.append(file)

# Count distinct first 7 characters
num_sequences = 1
for i in range(len(images)):
    if images[i][:7] != images[i-1][:7]:
        num_sequences += 1

# Create a list of pairs
pairs = []

# Loop through all images
for i in range(0, len(images), 2):
    # Get the pair
    pair = [images[i], images[i+1]]
    # Add the pair to the list
    pairs.append(pair)

# Create dictionary
data = {
    "control": {
        "num_pairs": len(pairs),
        "num_sequences": num_sequences,
        "path": "images/in/"
    },
    "image_pairs": pairs
}


# Create a json file
with open("pairs.json", "w") as file:
    json.dump(data, file)

# End of script