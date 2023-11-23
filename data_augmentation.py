import atexit # For playing a sound when the program finishes
import cv2 # For loading images
import imgaug as ia # For image data augmentation
import os # For running a command in the terminal
import platform # For getting the operating system name
from colorama import Style # For coloring the terminal
from imgaug import augmenters as iaa # For image data augmentation
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

# Input Constants:
INPUT_FILES = ["./Dataset/Simpsons/Train/", "./Dataset/Simpsons/Test/"] # The input files
IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"] # The image formats

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

# This function loops through the input files and augments the images
def process_files():
	# Get the number of files
	files_quantity = sum(1 for input_file in INPUT_FILES for root, directories, files in os.walk(input_file) for file in files if file.endswith(tuple(IMAGE_FORMATS)))

	with tqdm(total=files_quantity, desc=f"{BackgroundColors.GREEN}Processing Input Files{Style.RESET_ALL}") as pbar: # Create a progress bar
		for input_file in INPUT_FILES: # Loop through the input files
			for root, directories, files in os.walk(input_file): # Loop through the files in the input file
				for file in files: # Loop through the files in the input file
					if file.endswith(tuple(IMAGE_FORMATS)): # If the file ends with one of the image formats
						image_path = os.path.join(root, file) # Get the image path
						augmented_image_path = image_path[::-1].replace(".", "_augmented."[::-1], 1)[::-1] # Get the augmented image path
						image = cv2.imread(image_path) # Load the image
						image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert the image from BGR to RGB
						augmented_image = augment_image(image) # Augment the image
						cv2.imwrite(augmented_image_path, augmented_image) # Save the augmented image
						pbar.update(1) # Update the progress bar

# Function for image data augmentation
def augment_image(image):
	# Define the augmentation sequence
	augmentation_sequence = iaa.Sequential([
		iaa.Fliplr(0.5), # Horizontal flips
		iaa.Crop(percent=(0, 0.1)), # Random crops
		iaa.GaussianBlur(sigma=(0, 1.0)), # Gaussian blur
		iaa.Add((-10, 10), per_channel=0.5), # Add brightness
		iaa.Multiply((0.5, 1.5), per_channel=0.5), # Multiply brightness
		iaa.Affine( # Affine transformations
			scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # Scale images
			translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # Translate images
			rotate=(-45, 45), # Rotate images
			shear=(-16, 16), # Shear images
		) # End of affine transformations
	]) # End of augmentation sequence
	
	# Augment the image
	augmented_images = augmentation_sequence(images=[image])

	return augmented_images[0] # Return the augmented image

# This is the Main function
def main():
	print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Image Data Augmentation{BackgroundColors.GREEN} program!{Style.RESET_ALL}") # Output the welcome message

	process_files() # Process the files

	print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Thank you for using the {BackgroundColors.CYAN}Image Data Augmentation{BackgroundColors.GREEN} program!{Style.RESET_ALL}") # Output the goodbye message

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
	main() # Call the main function
