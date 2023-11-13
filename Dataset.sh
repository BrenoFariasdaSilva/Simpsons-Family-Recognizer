#!/bin/bash

# Run:
# chmod +x ./Dataset.sh
# ./Dataset.sh

# Unzip the dataset
unzip archive.zip -d dataset

# Rename the file inside the dataset folder to Credit-Card-Fraud-2023
mv dataset/creditcard_2023.csv dataset/Credit-Card-Fraud-2023.csv

# Clean up: remove the downloaded zip file
rm archive.zip

echo "Download and extraction complete."
