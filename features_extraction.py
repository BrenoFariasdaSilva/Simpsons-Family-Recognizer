import atexit # For playing a sound when the program finishes
import os # For running a command in the terminal
import numpy as np # For saving the features and labels to a text file
import platform # For getting the operating system name
import tensorflow as tf # To load the pre-trained models
from colorama import Style # For coloring the terminal
from tensorflow.keras.preprocessing import image # For loading images
from tensorflow.keras.applications import ( # For loading the pre-trained models
	inception_v3, # Inception V3
	resnet50, # ResNet-50
	vgg16, # VGG-16
	densenet, # DenseNet-201
	mobilenet_v2, # MobileNet V2
	xception, # Xception
	nasnet, # NASNet Mobile
	efficientnet, # EfficientNet B0
)
from tqdm import tqdm # For showing a progress bar

# Macros:
class BackgroundColors: # Colors for the terminal
   CYAN = "\033[96m" # Cyan
   GREEN = "\033[92m" # Green
   YELLOW = "\033[93m" # Yellow
   RED = "\033[91m" # Red
   BOLD = "\033[1m" # Bold
   UNDERLINE = "\033[4m" # Underline
   CLEAR_TERMINAL = "\033[H\033[J" # Clear the terminal

# Sound Constants:
SOUND_COMMANDS = {"Darwin": "afplay", "Linux": "aplay", "Windows": "start"}
SOUND_FILE = "./.assets/NotificationSound.wav" # The path to the sound file

# Dataset Constants:
MODELS = { # Dictionary of pre-trained models -> model name: model constructor
	"densenet201": densenet.DenseNet201,
	"efficientnetb0": efficientnet.EfficientNetB0,
	"mobilenetv2": mobilenet_v2.MobileNetV2,
	"nasnetmobile": nasnet.NASNetMobile,
	"resnet50": resnet50.ResNet50,
	"vgg16": vgg16.VGG16,
	"xception": xception.Xception,
	"inception_v3": inception_v3.InceptionV3,
}

OUTPUT_MODELS = { # Dictionary of output for the pre-trained models -> model name: Output model name
	"densenet201": "DenseNet201",
	"efficientnetb0": "EfficientNetB0",
	"mobilenetv2": "MobileVetV2",
	"nasnetmobile": "NASNetMobile",
	"resnet50": "ResNet50",
	"vgg16": "VGG16",
	"xception": "Xception",
	"inception_v3": "InceptionV3",
}

INPUT_FILES = ["./Dataset/Simpsons/Train/", "./Dataset/Simpsons/Test/"] # The input files
OUTPUT_FILE_FORMAT = ".txt" # The output file format

# Functions:

# This function defines the command to play a sound when the program finishes
def play_sound():
   if os.path.exists(SOUND_FILE):
      if platform.system() in SOUND_COMMANDS: # If the platform.system() is in the SOUND_COMMANDS dictionary
         os.system(f"{SOUND_COMMANDS[platform.system()]} {SOUND_FILE}")
      else: # If the platform.system() is not in the SOUND_COMMANDS dictionary
         print(f"{BackgroundColors.RED}The {BackgroundColors.CYAN}platform.system(){BackgroundColors.RED} is not in the {BackgroundColors.CYAN}SOUND_COMMANDS dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}")
   else: # If the sound file does not exist
      print(f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{SOUND_FILE}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}")

# Register the function to play a sound when the program finishes
atexit.register(play_sound)

# This function creates the input directory if it doesn't exist and deletes
def create_input_directory(input_directory):
	# Create the directory if it doesn't exist
	if not os.path.exists(input_directory):
		os.makedirs(input_directory) # Recursive directory creation function
		
# This function creates the output directory if it doesn't exist and deletes
def create_output_directory(output_file):
	output_dir = os.path.dirname(output_file) # Get the directory name
   
   # Create the directory if it doesn't exist
	if not os.path.exists(output_dir):
		os.makedirs(output_dir) # Recursive directory creation function
	if os.path.exists(f"{output_file}.txt"):
		os.remove(f"{output_file}.txt") # Delete the file if it exists
          
# This function extracts deep features from the specified layer of the pre-trained model
def deep_features(imds, model, layer, labels, output_file):
	# Resize images according to the mode's input size
	input_shape = (299, 299)  # Modify this according to your model"s input shape
	datagen = image.ImageDataGenerator(rescale=1./255)	# Create an image generator that reads images from the dataset directory
	img_generator = datagen.flow_from_directory( # Create a generator that reads images from the dataset directory
		imds,
		target_size=input_shape,
		batch_size=1,
		class_mode=None,
		shuffle=False,
	)

	# Extract features from the specified layer
	feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer).output)
	features = feature_extractor.predict(img_generator)

	# Reshape features to a flat format
	features_flat = features.reshape((features.shape[0], -1))

	# Combine features and labels
	data = np.hstack((features_flat, np.array(labels)[:, np.newaxis]))

	# Save features and labels to a text file
	np.savetxt(output_file, data, fmt="%.6f")

# This is the Main function
def main():
   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Feature Extraction{BackgroundColors.BOLD} program!{Style.RESET_ALL}")
   print(f"{BackgroundColors.GREEN}This program extracts deep features from the specified layer of the pre-trained model.{Style.RESET_ALL}")
   print(f"{BackgroundColors.YELLOW}The outputs generated by this program are text files containing the extracted features and labels that can be used for training a classifier in the {BackgroundColors.CYAN}best_parameters.py{BackgroundColors.GREEN} and {BackgroundColors.CYAN}specific_parameters.py{BackgroundColors.GREEN} programs.{Style.RESET_ALL}")

   # Create the progress bar for the models and datasets
   with tqdm(total=(len(MODELS) * len(INPUT_FILES)), desc=f"{BackgroundColors.GREEN}Processing models and datasets{Style.RESET_ALL}") as pbar:
      # Iterate over each model
      for model_name, model_constructor in MODELS.items():
         # Iterate over each dataset
         for dataset_path in INPUT_FILES:
            # Loading a pre-trained model and extracting features
            model = model_constructor(weights="imagenet", include_top=False)

            layer_name = model.layers[-1].name # Get the name of the last layer

            # Update the progress bar
            pbar.set_description(f"{BackgroundColors.GREEN}Processing model {BackgroundColors.CYAN}{OUTPUT_MODELS[model_name]}{BackgroundColors.GREEN} using the {BackgroundColors.CYAN}{layer_name.capitalize()}{BackgroundColors.GREEN} with dataset {BackgroundColors.CYAN}{dataset_path.split('/')[-1]}{Style.RESET_ALL}")

            # Provide the path and name for the output file
            output_file_name = f"./Dataset/{OUTPUT_MODELS[model_name]}/{dataset_path.split('/')[-1]}{OUTPUT_FILE_FORMAT}"

            # You can assign labels to the images manually or load them from your dataset
            # Classes: 01 (Bart), 02 (Homer), 03 (Lisa), 04 (Maggie) and 05 (Marge)
            labels = [0] * len(os.listdir(os.path.join(dataset_path, "01"))) + \
                     [1] * len(os.listdir(os.path.join(dataset_path, "02"))) + \
                     [2] * len(os.listdir(os.path.join(dataset_path, "03"))) + \
                     [3] * len(os.listdir(os.path.join(dataset_path, "04"))) + \
                     [4] * len(os.listdir(os.path.join(dataset_path, "05"))) # Create a list of labels
            
            create_input_directory(dataset_path) # Create the input directory if it doesn't exist
            create_output_directory(output_file_name) # Create the output directory if it doesn't exist
            
            # Extract deep features from the specified layer of the pre-trained model
            deep_features(dataset_path, model, layer_name, labels, output_file_name)

            pbar.update(1) # Update the progress bar
            print(f"The code was executed successfully for the file {output_file_name} using the model {model_name}.")

   print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}The program finished successfully!{Style.RESET_ALL}")

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
	main() # Call the main function
