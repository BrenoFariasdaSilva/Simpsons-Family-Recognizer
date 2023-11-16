#!/bin/bash

# Run:
# chmod +x ./Setup-Dataset.sh
# ./Setup-Dataset.sh

# URL of the zip file on Google Drive
file_url="https://drive.google.com/uc?export=download&id=1wVyUmsz150uKjOprxRA_4LtmTXDPRp1o"

# Output zip file name
zip_file="Simpsons-Dataset.zip"

# Download the file using wget
wget "$file_url" -O "$zip_file"

# Extract the contents
unzip $zip_file

# Clean up: remove the downloaded zip file
rm $zip_file

echo "Download and extraction complete."