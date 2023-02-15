# Script for renaming images, replacing part of the name with a new one

import os

# New name
old_name = "HCULB_00033"
new_name = "Seq_001"

# Get all files in the folder
path = "./"
files = os.listdir(path)

# Loop through all files
for file in files:
    # Get the file extension
    file_extension = os.path.splitext(file)[1]
    # Check if the file is an image
    if file_extension in [".jpg", ".jpeg", ".png"]:
        # Replace part of the name
        file_new = file
        file_new = file.replace(old_name, new_name)
        print(file, file_extension, file_new)
    # Rename the file
        os.rename(path + file, path + file_new)

# End of script
