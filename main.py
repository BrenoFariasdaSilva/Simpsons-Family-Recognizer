import atexit # For playing a sound when the program finishes
import numpy as np # For the data manipulation
import os # For running a command in the terminal
import platform # For getting the operating system name
import time # For the timer
from colorama import Style # For coloring the terminal
from sklearn import svm # For the SVM classifier
from sklearn import tree # For the decision tree classifier
from sklearn.ensemble import RandomForestClassifier # For the random forest classifier
from sklearn.metrics import classification_report # For the classification report
from sklearn.model_selection import GridSearchCV # For the grid search
from sklearn.naive_bayes import GaussianNB # For the Naive Bayes classifier
from sklearn.neighbors import KNeighborsClassifier # For the k-NN classifier
from sklearn.neural_network import MLPClassifier # For the MLP classifier
from sklearn.pipeline import Pipeline # For the pipeline
from sklearn.preprocessing import StandardScaler # For the standard scaler

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

# Constants:
INPUT_FILES = ["./", "./"] # The input files

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

# This function loads the data from the dataset files and returns the training and test sets
def load_data():
   print(f"{BackgroundColors.BOLD}{BackgroundColors.YELLOW}Remember to remove the header line from the dataset files. They should be in the format: {BackgroundColors.CYAN}label feature1 feature2 ... featureN{Style.RESET_ALL}")
   print(f"\n{BackgroundColors.GREEN}Loading data...{Style.RESET_ALL}")
   tr = np.loadtxt(f"{INPUT_FILES[0]}") # Load the training data
   ts = np.loadtxt(f"{INPUT_FILES[1]}") # Load the test data
   train_label = tr[:, 0] # The first column is the label
   test_label = ts[:, 0] # The first column is the label
   train_features_values = tr[:, 1:] # The second column to the last is the feature vector
   test_features_values = ts[:, 1:] # The second column to the last is the feature vector
   return train_features_values, train_label, test_features_values, test_label # Return the data

# This function creates a k-NN classifier and prints the classification report
def knn(train_features_values, train_label, test_features_values, test_label):
   print(f"{BackgroundColors.GREEN}1º {BackgroundColors.CYAN}K-NN Classifier{BackgroundColors.GREEN}.{Style.RESET_ALL}")
   start_time = time.time() # Start the timer
   neigh = KNeighborsClassifier(n_neighbors=1, metric="euclidean") # Instantiate the classifier
   neigh.fit(train_features_values, train_label) # Train the classifier
   y_pred = neigh.predict(test_features_values) # Predict the test set
   accuracy = neigh.score(test_features_values, test_label) # Calculate the accuracy
   execution_time = time.time() - start_time # Calculate the execution time

   return accuracy, {"N Neighbors": 1, "Metric": "Euclidean", "Execution Time": f"{execution_time:.5f} Seconds"} # Return the Accuracy and the Parameters

# This function creates a Decision Tree classifier with grid search and prints the classification report
def decision_tree(train_features_values, train_label, test_features_values, test_label):
   print(f"{BackgroundColors.GREEN}2º {BackgroundColors.CYAN}Decision Tree Classifier{BackgroundColors.GREEN}.{Style.RESET_ALL}")
   start_time = time.time() # Start the timer
   clf = tree.DecisionTreeClassifier() # Instantiate the classifier
   clf.fit(train_features_values, train_label) # Train the classifier
   y_pred = clf.predict(test_features_values) # Predict the test set
   accuracy = clf.score(test_features_values, test_label) # Calculate the accuracy
   execution_time = time.time() - start_time # Calculate the execution time

   return accuracy, {"Criterion": "Gini", "Splitter": "Best", "Execution Time": f"{execution_time:.5f} Seconds"} # Return the Accuracy and the Parameters

# This is the Main function
def main():
   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Hello, World!{Style.RESET_ALL}") # Output the Welcome message

   print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}") # Output the end of the program message

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
	main() # Call the main function
