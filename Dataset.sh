#!/bin/bash

# Run:
# chmod +x ./Dataset.sh
# ./Dataset.sh

unzip archive.zip -d dataset

# Get the name of the file inside the dataset folder
file_name=$(ls dataset)

# Rename the file inside the dataset folder to Credit-Card-Fraud-2023.csv
mv dataset/"$file_name" dataset/Credit-Card-Fraud-2023.csv

# Clean up: remove the downloaded zip file
rm archive.zip

echo "Download and extraction complete."
