import atexit # For playing a sound when the program finishes
import numpy as np # For the data manipulation
import os # For running a command in the terminal
import platform # For getting the operating system name
import time # For the timer
import tqdm # For the progress bar
from collections import Counter # For the majority voting
from colorama import Style # For coloring the terminal
from sklearn import svm # For the SVM classifier
from sklearn import tree # For the decision tree classifier
from sklearn.ensemble import RandomForestClassifier # For the random forest classifier
from sklearn.metrics import classification_report # For the classification report
from sklearn.metrics import confusion_matrix # For the confusion matrix
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

# Input Constants:
INPUT_DEEP_LEARNING_MODEL = "ResNet18" # The deep learning model to use
INPUT_FILES = {
   "NasNetLarge": ["./Dataset/NasNetLarge/Train.txt", "./Dataset/NasNetLarge/Test.txt"],
   "ResNet18": ["./Dataset/ResNet18/Train.txt", "./Dataset/ResNet18/Test.txt"],
}

# Output Constants:
SHOW_CONFUSION_MATRIX = False # If True, show the confusion matrix
SHOW_CLASSIFICATION_REPORT = False # If True, show the classification report

# Classifiers Constants:
CLASSIFIERS = { # The classifiers
   "K-Nearest Neighbors": "k_nearest_neighbors",
   "Decision Tree": "decision_tree",
   "Support Vector Machine": "support_vector_machine",
   "Multilayer Perceptron": "multilayer_perceptron",
   "Random Forest": "random_forest",
   "Naive Bayes": "naive_bayes",
}

BEST_PARAMETERS = { # The best parameters for each classifier
   "K-Nearest Neighbors": {"metric": "euclidean", "n_neighbors": 1},
   "Decision Tree": {"criterion": "gini", "max_depth": 30, "splitter": "best"},
   "Support Vector Machine": {"C": 10, "Gamma": 0.001, "kernel": "linear"},
   "Multilayer Perceptron": {"alpha": 0.0001, "hidden_layer_sizes": (500, 500, 500, 500), "solver": "adam"},
   "Random Forest": {"max_depth": None, "n_estimators": 500},
   "Naive Bayes": {"priors": None, "var_smoothing": 1e-05},
}

# Define the best combination of classifiers to use
BEST_COMBINATION = ["K-Nearest Neighbors", "Multilayer Perceptron", "Random Forest"]

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
   tr = np.loadtxt(f"{INPUT_FILES[INPUT_DEEP_LEARNING_MODEL][0]}") # Load the training data
   ts = np.loadtxt(f"{INPUT_FILES[INPUT_DEEP_LEARNING_MODEL][1]}") # Load the test data
   train_labels = tr[:, -1] # Get the training labels
   train_features = tr[:, :-1] # Get the training features
   test_labels = ts[:, -1] # Get the test labels
   test_features = ts[:, :-1] # Get the test features
   return train_features, train_labels, test_features, test_labels # Return the data

# This function creates a k-NN classifier and prints the classification report
def k_nearest_neighbors(train_features, train_labels, test_features, test_labels):
   best_params = BEST_PARAMETERS["K-Nearest Neighbors"] # Get the best parameters for the k-NN classifier
   knn = KNeighborsClassifier(**best_params) # Create the k-NN classifier
   start_time = time.time() # Start the timer
   knn.fit(train_features, train_labels) # Train the k-NN classifier
   y_pred = knn.predict(test_features) # Predict the test labels
   execution_time = time.time() - start_time # Calculate the execution time
   accuracy = knn.score(test_features, test_labels) # Calculate the accuracy

   if SHOW_CLASSIFICATION_REPORT: # Show the classification report if it is set to True
      print(f"{classification_report(test_labels, y_pred)}{Style.RESET_ALL}") # Print the classification report

   if SHOW_CONFUSION_MATRIX: # Show the confusion matrix if it is set to True
      conf_matrix = confusion_matrix(test_labels, y_pred) # Calculate the confusion matrix
      print(f"{BackgroundColors.GREEN}K-Nearest Neighbors Confusion Matrix:\n{BackgroundColors.CYAN}{conf_matrix}{Style.RESET_ALL}") # Print the confusion matrix
   
   return accuracy, y_pred, {"Predefined Parameters": best_params, "Execution Time": f"{execution_time:.5f} Seconds"}

# This function creates a Decision Tree classifier with grid search and prints the classification report
def decision_tree(train_features, train_labels, test_features, test_labels):
   best_params = BEST_PARAMETERS["Decision Tree"] # Get the best parameters for the decision tree classifier
   dt = tree.DecisionTreeClassifier(**best_params) # Create the decision tree classifier
   start_time = time.time() # Start the timer
   dt.fit(train_features, train_labels) # Train the decision tree classifier
   y_pred = dt.predict(test_features) # Predict the test labels
   execution_time = time.time() - start_time # Calculate the execution time
   accuracy = dt.score(test_features, test_labels) # Calculate the accuracy

   if SHOW_CLASSIFICATION_REPORT: # Show the classification report if it is set to True
      print(f"{classification_report(test_labels, y_pred)}{Style.RESET_ALL}")
   
   if SHOW_CONFUSION_MATRIX: # Show the confusion matrix if it is set to True
      conf_matrix = confusion_matrix(test_labels, y_pred) 
      print(f"{BackgroundColors.GREEN}Decision Tree Confusion Matrix:\n{BackgroundColors.CYAN}{conf_matrix}{Style.RESET_ALL}")

   return accuracy, y_pred, {"Predefined Parameters": best_params, "Execution Time": f"{execution_time:.5f} Seconds"}

# This function creates a SVM classifier with grid search and prints the classification report
def support_vector_machine(train_features, train_labels, test_features, test_labels):
   best_params = BEST_PARAMETERS["Support Vector Machine"] # Get the best parameters for the SVM classifier
   # Create the SVM classifier
   svm_clf = svm.SVC(probability=True, C=best_params["C"], gamma=best_params["Gamma"], kernel=best_params["kernel"])
   ss = StandardScaler() # Create the standard scaler
   pipeline = Pipeline([("scaler", ss), ("svm", svm_clf)]) # Create the pipeline
   start_time = time.time() # Start the timer
   pipeline.fit(train_features, train_labels) # Train the SVM classifier
   y_pred = pipeline.predict(test_features) # Predict the test labels
   execution_time = time.time() - start_time # Calculate the execution time
   accuracy = pipeline.score(test_features, test_labels) # Calculate the accuracy

   if SHOW_CLASSIFICATION_REPORT: # Show the classification report if it is set to True
      print(f"{classification_report(test_labels, y_pred)}{Style.RESET_ALL}")

   if SHOW_CONFUSION_MATRIX: # Show the confusion matrix if it is set to True
      conf_matrix = confusion_matrix(test_labels, y_pred)
      print(f"{BackgroundColors.GREEN}Support Vector Machine Confusion Matrix:\n{BackgroundColors.CYAN}{conf_matrix}{Style.RESET_ALL}")

   return accuracy, y_pred, {"Predefined Parameters": best_params, "Execution Time": f"{execution_time:.5f} Seconds"}

# This function creates a Multilayer Perceptron classifier and prints the classification report
def multilayer_perceptron(train_features, train_labels, test_features, test_labels):
   best_params = BEST_PARAMETERS["Multilayer Perceptron"] # Get the best parameters for the MLP classifier
   mlp = MLPClassifier(**best_params, random_state=1, max_iter=2000) # Create the MLP classifier
   start_time = time.time() # Start the timer
   mlp.fit(train_features, train_labels) # Train the MLP classifier
   y_pred = mlp.predict(test_features) # Predict the test labels
   execution_time = time.time() - start_time # Calculate the execution time
   accuracy = mlp.score(test_features, test_labels) # Calculate the accuracy

   if SHOW_CLASSIFICATION_REPORT: # Show the classification report if it is set to True
      print(f"{classification_report(test_labels, y_pred)}{Style.RESET_ALL}") # Print the classification report

   if SHOW_CONFUSION_MATRIX: # Show the confusion matrix if it is set to True
      conf_matrix = confusion_matrix(test_labels, y_pred)
      print(f"{BackgroundColors.GREEN}Multilayer Perceptron Confusion Matrix:\n{BackgroundColors.CYAN}{conf_matrix}{Style.RESET_ALL}") # Print the confusion matrix

   return accuracy, y_pred, {"Predefined Parameters": best_params, "Execution Time": f"{execution_time:.5f} Seconds"}

# This function creates a Random Forest classifier and prints the classification report
def random_forest(train_features, train_labels, test_features, test_labels):
   best_params = BEST_PARAMETERS["Random Forest"] # Get the best parameters for the random forest classifier
   rf = RandomForestClassifier(**best_params, random_state=1) # Create the random forest classifier
   start_time = time.time() # Start the timer
   rf.fit(train_features, train_labels) # Train the random forest classifier
   y_pred = rf.predict(test_features) # Predict the test labels
   execution_time = time.time() - start_time # Calculate the execution time
   accuracy = rf.score(test_features, test_labels) # Calculate the accuracy

   if SHOW_CLASSIFICATION_REPORT: # Show the classification report if it is set to True
      print(f"{classification_report(test_labels, y_pred)}{Style.RESET_ALL}") # Print the classification report

   if SHOW_CONFUSION_MATRIX: # Show the confusion matrix if it is set to True
      conf_matrix = confusion_matrix(test_labels, y_pred) # Calculate the confusion matrix
      print(f"{BackgroundColors.GREEN}Random Forest Confusion Matrix:\n{BackgroundColors.CYAN}{conf_matrix}{Style.RESET_ALL}") # Print the confusion matrix

   return accuracy, y_pred, {"Predefined Parameters": best_params, "Execution Time": f"{execution_time:.5f} Seconds"}

# This function trains the Naive Bayes classifier and prints the classification report
def naive_bayes(train_features, train_labels, test_features, test_labels):
   best_params = BEST_PARAMETERS["Naive Bayes"] # Get the best parameters for the Naive Bayes classifier
   nb = GaussianNB(**best_params) # Create the Naive Bayes classifier
   start_time = time.time() # Start the timer
   nb.fit(train_features, train_labels) # Train the Naive Bayes classifier
   y_pred = nb.predict(test_features) # Predict the test labels
   execution_time = time.time() - start_time # Calculate the execution time
   accuracy = nb.score(test_features, test_labels) # Calculate the accuracy

   if SHOW_CLASSIFICATION_REPORT: # Show the classification report if it is set to True
      print(f"{classification_report(test_labels, y_pred)}{Style.RESET_ALL}")

   if SHOW_CONFUSION_MATRIX: # Show the confusion matrix if it is set to True
      conf_matrix = confusion_matrix(test_labels, y_pred) # Calculate the confusion matrix
      print(f"{BackgroundColors.GREEN}Naive Bayes Confusion Matrix:\n{BackgroundColors.CYAN}{conf_matrix}{Style.RESET_ALL}") # Print the confusion matrix

   return accuracy, y_pred, {"Predefined Parameters": best_params, "Execution Time": f"{execution_time:.5f} Seconds"}

# This function trains and evaluates a classifier
def train_and_evaluate_classifier(classifier_function, train_features, train_labels, test_features, test_labels):
   classifier_function = globals()[classifier_function] # Use globals() to get the function object from its name
   accuracy, y_pred, parameters = classifier_function(train_features, train_labels, test_features, test_labels) # Train the classifier and get the accuracy, predictions and parameters
   return accuracy, y_pred, parameters # Return the accuracy, predictions and parameters

# This function trains the selected classifiers
def train_all_classifiers(classifiers, selected_classifiers, train_features, train_labels, test_features, test_labels):
   classifiers_execution = {} # Initialize the classifiers execution dictionary
   classifiers_predictions = {} # Initialize the classifiers predictions dictionary

   # Create progress bar that shows the training progress and the current classifier
   with tqdm.tqdm(total=len(selected_classifiers), desc=f"{BackgroundColors.GREEN}Training Classifiers{Style.RESET_ALL}", bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}") as pbar:
      # Loop through the selected classifiers
      for classifier_name in selected_classifiers:
         classifier_function = classifiers[classifier_name] # Get the classifier function
         accuracy, y_pred, parameters = train_and_evaluate_classifier(classifier_function, train_features, train_labels, test_features, test_labels) # Train and evaluate the classifier
         classifiers_execution[classifier_name] = (accuracy, parameters) # Add the classifier execution to the dictionary
         classifiers_predictions[classifier_name] = y_pred # Add the classifier predictions to the dictionary
         pbar.update(1) # Update the progress bar

   return classifiers_execution, classifiers_predictions # Return the classifiers execution and predictions dictionaries

# This function performs majority voting on the classifiers predictions
def majority_vote_predictions(classifiers_predictions):
   transposed_predictions = list(map(list, zip(*classifiers_predictions.values()))) # Transpose the predictions
   final_predictions = [Counter(instance_predictions).most_common(1)[0][0] for instance_predictions in transposed_predictions] # Calculate the majority vote predictions
   return final_predictions # Return the majority vote predictions

# This function trains the best combination of classifiers
def train_best_combination(classifiers, train_features, train_labels, test_features, test_labels):
   classifiers_execution = {} # Initialize the classifiers execution dictionary
   classifiers_predictions = {} # Initialize the classifiers predictions dictionary
   start_time = time.time() # Start the timer

   # Create progress bar that shows the training progress and the current classifier
   with tqdm.tqdm(total=len(BEST_COMBINATION), desc=f"{BackgroundColors.GREEN}Training Classifiers{Style.RESET_ALL}", bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}") as pbar:
      # Loop through the selected classifiers
      for classifier_name in BEST_COMBINATION:
         classifier_function = classifiers[classifier_name] # Get the classifier function
         accuracy, y_pred, parameters = train_and_evaluate_classifier(classifier_function, train_features, train_labels, test_features, test_labels) # Train and evaluate the classifier
         classifiers_execution[classifier_name] = (accuracy, parameters) # Add the classifier execution to the dictionary
         classifiers_predictions[classifier_name] = y_pred # Add the classifier predictions to the dictionary
         pbar.update(1) # Update the progress bar

   execution_time = time.time() - start_time # Calculate the execution time
   classifiers_execution["Best Combination"] = (None, {"Predefined Parameters": BEST_COMBINATION, "Execution Time": f"{execution_time:.5f} Seconds"}) # Add the best combination to the classifiers execution dictionary

   return classifiers_execution, classifiers_predictions # Return the classifiers execution and predictions dictionaries

# This function sort the classifiers by accuracy
def sort_classifiers_execution(classifiers_execution):
   # Sort the classifiers by accuracy and return the sorted dictionary
   return dict(sorted(classifiers_execution.items(), key=lambda item: item[1][0], reverse=True))

# This function prints the classifiers accuracy, parameters and execution time
def print_classifiers_execution(sorted_classifiers_execution):
   print(f"\n{BackgroundColors.GREEN}Classifiers Results for {BackgroundColors.CYAN}{INPUT_DEEP_LEARNING_MODEL}{BackgroundColors.GREEN}:{BackgroundColors.CYAN}") # Print the classifiers results

   # loop through the classifiers name, accuracy and parameters
   for classifier, accuracy in sorted_classifiers_execution.items():
      print(f"{BackgroundColors.GREEN}{classifier}: {BackgroundColors.CYAN}{accuracy[0]*100:.2f}%{Style.RESET_ALL}")
      for parameter, value in accuracy[1].items():
         print(f"{BackgroundColors.GREEN}{parameter}: {BackgroundColors.CYAN}{value}{Style.RESET_ALL}")
      print(f"{Style.RESET_ALL}")

# This is the main function. It calls the other functions, building the project workflow
def main():
   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the Simpsons Family Characters Classifier!{Style.RESET_ALL}") # Output the welcome message

   train_features, train_labels, test_features, test_labels = load_data() # Load the data

   # Train every classifier and get the classifiers execution and predictions
   selected_classifiers_execution, selected_classifiers_predictions = train_all_classifiers(CLASSIFIERS, CLASSIFIERS, train_features, train_labels, test_features, test_labels)

   # Train the best combination of classifiers and get the classifiers execution and predictions
   best_combination_execution, best_combination_predictions = train_best_combination(CLASSIFIERS, train_features, train_labels, test_features, test_labels)

   # Perform majority voting on the classifiers predictions for the best combination
   majority_vote = majority_vote_predictions(best_combination_predictions)

   # Calculate the accuracy of the majority vote
   majority_vote_accuracy = sum(majority_vote == test_labels) / len(test_labels)

   # Add the majority vote to the selected_classifiers_execution dictionary
   selected_classifiers_execution["Majority Vote"] = (majority_vote_accuracy, {"Predefined Parameters": BEST_COMBINATION, "Execution Time": best_combination_execution["Best Combination"][1]["Execution Time"]})

   # Sort the classifiers by accuracy
   selected_classifiers_execution = sort_classifiers_execution(selected_classifiers_execution)

   print_classifiers_execution(selected_classifiers_execution) # Print the classifiers accuracy, parameters and execution time

   print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Thank you for using the Simpsons Family Characters Classifier!{Style.RESET_ALL}") # Output the goodbye message

# This is the standard boilerplate that calls the main() function.
if __name__ == "__main__":
	main() # Call the main function
