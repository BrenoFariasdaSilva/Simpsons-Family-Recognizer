#!/bin/bash

# Run:
# chmod +x ./Setup-Dataset.sh
# ./Setup-Dataset.sh

dataset_dir="./Dataset"
zip_file="Simpsons-Dataset.zip"

# Clean up: Remove the downloaded zip file
cleanup() {
	rm "$zip_file"
}

# Verify that the dataset has not already been downloaded
verify_dataset() {
	if [ -d "$dataset_dir" ]; then
		echo "Dataset already exists. Removing it first."
		rm -rf "$dataset_dir"
	fi
}

# Download the dataset
download_dataset() {
	# URL of the zip file on Google Drive
	file_url="https://drive.google.com/uc?export=download&id=1wVyUmsz150uKjOprxRA_4LtmTXDPRp1o"

	# Download the file using wget
	wget "$file_url" -O "$zip_file"
}

# Extract the zip file
extract_dataset() {
	unzip "$zip_file"
}

# Main function
main() {
	verify_dataset # Check if dataset already exists
	download_dataset # Download the dataset
	extract_dataset # Extract the dataset
	cleanup # Remove the downloaded zip file

	echo "Download and extraction complete."
}

# Run the main function
main