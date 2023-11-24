<div align="center">
  
# [Simpsons-Family-Recognizer.](https://github.com/BrenoFariasdaSilva/Simpsons-Family-Recognizer.git) <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg"  width="3%" height="3%">

</div>

<div align="center">
  
---

This project focuses on using the K-Nearest Neighbors (K-NN), Decision Trees (DT), Support Vector Machines (SVM), MultiLayer Perceptrons (MLP), Random Forest, Naive Bayes and Combining the Supervised Learning Algorithms to recognize The Simpsons Family Members. The dataset used for this project can be downloaded from [here](https://drive.google.com/uc?export=download&id=1wVyUmsz150uKjOprxRA_4LtmTXDPRp1o) url.

---

</div>

<div align="center">

![GitHub Code Size in Bytes](https://img.shields.io/github/languages/code-size/BrenoFariasdaSilva/Simpsons-Family-Recognizer)
![GitHub Last Commit](https://img.shields.io/github/last-commit/BrenoFariasdaSilva/Simpsons-Family-Recognizer)
![GitHub](https://img.shields.io/github/license/BrenoFariasdaSilva/Simpsons-Family-Recognizer)
![wakatime](https://wakatime.com/badge/user/28f6c354-43b3-4634-8ec1-631893fe27d0/project/018bd988-d987-4ea0-955e-ba21bce66b1c.svg)

</div>

<div align="center">
  
![RepoBeats Statistics](https://repobeats.axiom.co/api/embed/c1d9310ac47ca95fa592c68214c0b91a81154eda.svg "Repobeats analytics image")

</div>

## Table of Contents
- [Simpsons-Family-Recognizer. ](#simpsons-family-recognizer-)
	- [Table of Contents](#table-of-contents)
	- [Introduction](#introduction)
	- [Machine Learning Supervised Classifiers](#machine-learning-supervised-classifiers)
		- [K-Nearest Neighbors (K-NN)](#k-nearest-neighbors-k-nn)
		- [Decision Trees (DT)](#decision-trees-dt)
		- [Support Vector Machines (SVM)](#support-vector-machines-svm)
		- [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
		- [Random Forest](#random-forest)
		- [Naive Bayes](#naive-bayes)
	- [Combining the Supervised Learning Algorithms](#combining-the-supervised-learning-algorithms)
	- [Dataset](#dataset)
		- [Data Description](#data-description)
	- [Feature Extraction](#feature-extraction)
	- [Algorithms](#algorithms)
		- [Best Parameters Python Algorithm](#best-parameters-python-algorithm)
			- [Usage](#usage)
			- [Output Example](#output-example)
		- [Data Augmentation Python Algorithm](#data-augmentation-python-algorithm)
			- [Usage](#usage-1)
			- [Output Example](#output-example-1)
		- [Feature Extraction Python Algorithm](#feature-extraction-python-algorithm)
			- [Usage](#usage-2)
			- [Output Example](#output-example-2)
		- [Specific Parameters Python Algorithm](#specific-parameters-python-algorithm)
			- [Usage](#usage-3)
			- [Output Example](#output-example-3)
	- [Requirements](#requirements)
	- [Setup](#setup)
		- [Clone the repository](#clone-the-repository)
		- [Install Dependencies](#install-dependencies)
		- [Setup Dataset](#setup-dataset)
	- [How to run](#how-to-run)
	- [Results](#results)
		- [Results Analysis](#results-analysis)
	- [Important Notes](#important-notes)
	- [Contributing](#contributing)
	- [License](#license)

## Introduction

Classification problems in AI involve assigning predefined labels or categories to input data based on its features. The goal is to train models that can generalize patterns from labeled examples to accurately predict the class of new, unseen instances.  
In this project, we use the K-Nearest Neighbors (K-NN), Decision Trees (DT), Support Vector Machines (SVM), MultiLayer Perceptrons (MLP), Random Forest and Combine those Supervised Learning Algorithms to recognize The Simpsons Family Members. So, this project will extract features from Bart, Homer, Lisa Maggie e Marge characters of the Simpsons family, in order to use all of the previously mentioned algorithms to train them to recognize them from the features we extract with the labels we provide, which character is in the image is in the image name.  
The system will try to predict which image represents which member and will produce as output the accuracy and F1-Score [0 to 100%] and a confusion matrix (N × N) indicating the percentage of system hits and errors among the N classes.

## Machine Learning Supervised Classifiers

This project employs various machine learning supervised classifiers to recognize the Simpsons family members. Each classifier has unique characteristics and applications. Below are brief descriptions of the classifiers used in this project.

### K-Nearest Neighbors (K-NN)

K-Nearest Neighbors (K-NN) is a simple and widely used machine learning algorithm that falls under the category of supervised learning. It is a non-parametric and instance-based method used for classification and regression tasks. The fundamental idea behind K-NN is to make predictions based on the majority class or average value of the `k` nearest data points in the feature space. In other words, the algorithm classifies or predicts the output of an instance by considering the labels of its nearest neighbors that are the data points closest to that instance in a Cartesian plane. The K-NN doesn't have an explicit training and classification progress, so its classification time could be very slow, depending on the size of the dataset.  
The impleted K-NN is using the Grid Search to find the best parameters for the model and the parameters used are:
- `metric`: The distance metric to use for the tree. The selected metrics are `"euclidean", "manhattan" and "minkowski"`.
- `n_neighbors`: Number of neighbors to use by default for kneighbors queries -> `1, 3, 5 and 7`.

### Decision Trees (DT)

Decision Trees are a versatile machine learning algorithm used for both classification and regression tasks. They make decisions based on a set of rules learned from the training data.  
Decision Trees recursively split the data into subsets based on the most significant features, creating a tree-like structure. They are interpretable and can capture complex relationships in the data.  
The implemented Decision Tree is using the Grid Search to find the best parameters for the model and the parameters used are:
- `criterion`: The function to measure the quality of a split. The selected criterion are `"gini" and "entropy"`.
- `max_depth`: The maximum depth of the tree. The selected max_depth are `None, 10, 20, 30`.
- `splitter`: The strategy used to choose the split at each node. The selected splitter are `"best" and "random"`.

### Support Vector Machines (SVM)

Support Vector Machines (SVM) is a powerful supervised learning algorithm used for classification and regression tasks. SVM aims to find a hyperplane that best separates data into different classes. It works by maximizing the margin between classes, and it is effective in high-dimensional spaces.  
SVM can handle non-linear relationships through the use of kernel functions.  
The implemented SVM is using the Grid Search to find the best parameters for the model and the parameters used are:
- `C`: Regularization parameter: It test the values from `0.01, 0.1, 1, 10, 100`
- `gamma`: Kernel coefficient. The selected gamma define the influence of input vectors on the margins. The values are from `0.001, 0.01, 0.1, 1, 10`.
- `kernel`: Specifies the kernel type to be used in the algorithm. The selected kernel are `"linear", "poly", "rbf" and "sigmoid"`.

### Multi-Layer Perceptron (MLP)

Multi-Layer Perceptron (MLP) is a type of artificial neural network commonly used for classification and regression tasks. It consists of multiple layers of interconnected nodes (neurons) with each layer having a set of weights. MLPs can capture complex relationships in data and are known for their ability to model non-linear functions.  
The implemented MLP is using the Grid Search to find the best parameters for the model and the parameters used are:
- `alpha`: L2 penalty (regularization term) parameter. The selected alpha are `1e-5, 1e-4 and 1e-3`.
- `hidden_layer_sizes`: The number of neurons in the hidden layers. The selected hidden_layer_sizes are `(100,), (100, 100), (500, 500, 500, 500)`.
- `solver`: The solver for weight optimization. The selected solver are `"adam" and "lbfgs"`.

### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve predictive accuracy and control overfitting. Each tree in the forest is trained on a random subset of the data, and the final prediction is based on the majority vote or average of the individual tree predictions. Random Forest is robust and effective for both classification and regression tasks.  
The implemented Random Forest is using the Grid Search to find the best parameters for the model and the parameters used are:
- `max_depth`: The maximum depth of the tree. The selected max_depth are `None, 10, 20, 30`.
- `n_estimators`: The number of trees in the forest. The selected n_estimators are `100, 500 and 1000`.

### Naive Bayes

Naive Bayes is a simple yet powerful machine learning algorithm commonly used for classification tasks. It is a probabilistic classifier that makes use of Bayes' Theorem, which states that the probability of A given B is equal to the probability of B given A times the probability of A divided by the probability of B. The algorithm it self is simple and easy to implement, and it is effective in high-dimensional spaces. It is also fast and can be used for both binary and multi-class classification tasks. It has a few drawbacks, such as the assumption of independent features and the zero-frequency problem. Also, it requires a parameter called "var_smoothing" to be set, which is a smoothing parameter that accounts for features not present in the learning samples and prevents zero probabilities in the prediction. "prior" is another parameter that can be set to specify the prior probabilities of the classes, and be aware that the sum of the priors must be 1.
The implemented Naive Bayes is using the Grid Search to find the best parameters for the model and the parameters used are:
- `priors`: Prior probabilities of the classes. The selected priors are `None and [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]`.
- `var_smoothing`: Portion of the largest variance of all features that is added to variances for calculation stability. The selected var_smoothing are `1e-9, 1e-8, 1e-7, 1e-6, 1e-5`.

## Combining the Supervised Learning Algorithms

The idea of combining the Supervised Learning Algorithms is to use the best parameters found by the Grid Search for each algorithm and combine them to create a new model. The new model will be trained with the best parameters found by the Grid Search for each algorithm and the prediction will be made by the majority vote of the predictions of each algorithm.

## Dataset

The dataset used for this project can be found [here](https://drive.google.com/uc?export=download&id=1wVyUmsz150uKjOprxRA_4LtmTXDPRp1o).

### Data Description

The dataset contains two folders, one named "Train" and another named "Test". Each of this directories have .bmp photos of the Simpsons family members. The "Train" folder contains 226 photos of the Simpsons family members as follows:

- Bart: 78 Photos.
- Home: 61 Photos.
- Lisa: 33 Photos.
- Maggie: 30 Photos.
- Marge: 24 Photos.

The "Test" folder contains 95 photos of the Simpsons family members.

- Bart: 35 Photos.
- Home: 25 Photos.
- Lisa: 13 Photos.
- Maggie: 12 Photos.
- Marge: 10 Photos.
  
All of the photos are named with the name of the character in the photo followed by a number, so that will be considered the label.

## Feature Extraction

The feature extraction process was tested using various deep learning models, each with a specific number of features. The following list provides an overview:

1. `DenseNet201`: 155520 features.
2. `EfficientNetB0`: 103680 features.
3. `InceptionV3`: 1000 features.
4. `MobileNetV2`: 103680 features.
5. `NASNetLarge`: 487872 features.
6. `ResNet18`: 512 features.
7. `ResNet50`: 204800 features.
8. `VGG16`: 1000 features.
9. `Xception`: 204800 features.

The chosen layer for extraction is always the last layer of the model. The resulting output is stored in `.txt` files, where all columns, except the last one, correspond to the extracted features, and the last column represents the label of the instance. The main difference between the models used in feature extraction is the quantity of features, as indicated in the list above.

You can find the extracted features datasets from any of the mentioned models [here](https://drive.google.com/drive/folders/12TjdYNLIml8E-k9G5HwZ4wxmeNwXNRCo?usp=share_link). Please note that the size of the extracted features file is generally proportional to the number of features extracted from a model.

## Algorithms

### Best Parameters Python Algorithm

This Python script automates the search for optimal parameters across a set of classifiers for character classification in the Simpsons Family dataset. It conducts a grid search for each classifier, identifying the best hyperparameters, and outputs the optimal combination of classifiers. It also outputs the accuracy, F1-Score [0 to 100%], Confusion Matrix [N × N], indicating the percentage of system hits and errors among the N classes, the execution time for each classifier, as well as it's Best Params found by the Grid Search.
This streamlined process is crucial for the final project step, where the specific_parameters.py algorithm combines classifiers with the best parameters using a majority vote for predictions.

#### Usage

1. **Deep Learning Model**: The script loads the dataset based on the selected deep learning model (`INPUT_DEEP_LEARNING_MODEL` variable).
2. **Classifiers**: It performs a grid search for each classifier specified in the `CLASSIFIERS` dictionary to find the best hyperparameters.
3. **Outputs**: Modify the outputs constants, such as, `SHOW_CLASSIFICATION_REPORT`, `SHOW_CONFUSION_MATRIX`, and `SHOW_DATASET_INFORMATION`.
So, Adjust the script's constants and parameters to fit your specific use case.

	```bash
	make best_parameters
	```

#### Output Example

```bash
Best Combination: ('K-Nearest Neighbors', 'Multilayer Perceptron' and 'Random Forest')
Majority Vote Accuracy: 85.2%
Execution Time: 120.256 Seconds

Decision Tree: 86.5%
Criterion: gini
Max Depth: None
Splitter: best
Execution Time: 32.543 Seconds

Random Forest: 84.8%
Max Depth: 30
Number of Estimators: 100
Execution Time: 45.678 Seconds

Support Vector Machine: 82.3%
C: 10
Gamma: 0.1
Kernel: rbf
Execution Time: 42.035 Seconds
```

### Data Augmentation Python Algorithm

This Python script performs image data augmentation on a set of input images. It utilizes the `imgaug` library to apply various transformations to the images, creating augmented versions for training purposes. The program is designed to work with image datasets used in machine learning projects.  
Data Augmentation is a technique used to increase the size of a dataset by applying various transformations to the images. This is useful for training machine learning models, as it allows the model to learn from a larger number of images. The script performs a series of transformations on the input images, creating augmented versions of the images. The augmented images are saved in the same directory as the original images, with the file name followed by the `_augmented` string.

#### Usage

1. **Input Files:** Specify the input directories (`INPUT_FILES`) containing the images you want to augment.
2. **Image Formats:** Define the accepted image formats (`IMAGE_FORMATS`) to filter the files in the input directories.
3. **Image Filters:** Specify the image filters/transformations to be applied in the `augmentation_sequence` variable.
So, Adjust the script's constants and parameters to fit your specific use case.

	```bash
	make data_augmentation
	```

#### Output Example

The terminal will not show any really useful output, rather than the progress of the execution. The output will be the augmented images, which will have the file name followed by the `_augmented` string, indicating that the image is an augmented image.

### Feature Extraction Python Algorithm

This Python script automates the extraction of deep features from athe last layer of pre-trained models for the Simpsons Family dataset. The features are saved to text files, which can be used for training classifiers in subsequent steps like `best_parameters.py` and `specific_parameters.py`.  
The script loads pre-trained models like `DenseNet201`, `EfficientNetB0`, `InceptionV3`, `ResNet50`, `VGG16`, and `Xception`, extracting deep features from the last layer of each model for character classification.

#### Usage

1. **Input Files:** Specify the input directories (`INPUT_FILES`) containing the images you want to augment.
2. **Deep Learning Models:** Define the deep learning models (`MODELS`) variable to be used for feature extraction.
So, Adjust the script's constants and parameters to fit your specific use case.

	```bash
	make feature_extraction
	```

#### Output Example

The terminal will not show any really useful output, rather than the progress of the execution. The output will be the txt files with the extracted features, where the last column represents the label of the instance and the other columns represent the extracted features. Those files will be saved in the `Dataset/ModelName/` directory.

### Specific Parameters Python Algorithm

This Python script is designed for training and evaluating various classifiers on a dataset of Simpsons family characters. It performs runs each classifier with the specified parameters, as well as the best Classifier Combination found by the `best_parameters.py` algorithm, and outputs the accuracy, F1-Score [0 to 100%], Confusion Matrix [N × N], indicating the percentage of system hits and errors among the N classes, the execution time for each classifier, as well as it's Best Params found by the Grid Search.

#### Usage

1. **Input Files:** Adjust the `INPUT_FILES` dictionary with the correct paths to your dataset files.
2. **Deep Learning Model:** The script loads the dataset based on the selected deep learning model (`INPUT_DEEP_LEARNING_MODEL` variable).
3. **Classifiers:** The script loads the classifiers based on the selected classifiers (`CLASSIFIERS` dictionary).
4. **Best Parameters:** The script loads the classifiers parameters based on the selected classifiers parameters (`BEST_PARAMETERS` dictionary).
5. **Best Combination:** The script loads the classifiers combination based on the selected classifiers combination (`BEST_COMBINATION` dictionary).
6. **Outputs:** Modify the outputs constants, such as, `SHOW_CLASSIFICATION_REPORT`, `SHOW_CONFUSION_MATRIX`, and `SHOW_DATASET_INFORMATION`.

So, Adjust the script's constants and parameters to fit your specific use case.

	```bash
	make specific_parameters
	```

#### Output Example

The script will display the results of each classifier, including accuracy, parameters, and execution time. The final section will present the best combination of classifiers and their performance.

## Requirements

- Python 3.x
- Makefile
- Colorama
- Collections
- ImgAug
- NumPy
- OpenCV-Python
- Scikit-learn
- TensorFlow
- TQDM
- Threadpoolctl (requires upgrade)

You can install all of the requirements (except the Python and the Makefile) by running the following command:

```bash
make dependencies
```

## Setup

### Clone the repository

1. Clone the repository with the following command:

```bash
git clone https://github.com/BrenoFariasdaSilva/Simpsons-Family-Recognizer.git
cd Simpsons-Family-Recognizer
```

### Install Dependencies

1. Install the project dependencies with the following command:

```bash
make dependencies
```

### Setup Dataset

1. Download any of the processed datasets from [here](https://drive.google.com/drive/folders/12TjdYNLIml8E-k9G5HwZ4wxmeNwXNRCo?usp=share_link) and place it in this project directory `(/Simpsons-Family-Recognizer/Dataset)` and extract it. After that, feel free to delete the `.zip` file.

## How to run

In order to run the project, make sure you already have installed the dependencies using the `make dependencies` command. In case you already did that and have the dataset right placed as well as the variables setup for each algorithm, first we must run the Data Augmentation code by following command:

```bash
make data_augmentation
```

Now we have the dataset size increased by 2 times, as each image received an augmented version of itself. With that in mind, we must extract the features from the images by running the following command:

```bash
make feature_extraction
```

Now we have the features extracted from the images, so we can run the algorithms to find the best parameters for each algorithm, as well as the best combination of them. To do that, run the following command:

```bash
make best_parameters
```

Now, with the output of the modify the `specific_parameters.py` algorithm with the best parameters found by the Grid Search for each algorithm  that was executed in the `best_parameters.py` code, so now we are finally able to classify the images. To do that, run the following command:

```bash
make specific_parameters
```

## Results

The results of the K-Nearest Neighbors (K-NN), Decision Tree (DT), Support Vector Machine (SVM), Multilayer Perceptron (MLP), Random Forest, Naive Bayes classifiers, as well as the classifiers combination will produce as output the accuracy, F1-Score [0 to 100%], it's Best Params found by the Grid Search, the execution time and a confusion matrix (N × N) indicating the percentage of system hits and errors among the N classes. That results will be shown in the terminal and saved in the `Results` directory.  

You can find all of my results in the `Results` directory, but i will show the results of the methods who gave the best accuracy, which is the following case:

1. Deep Learning Model: `DenseNet201`.
2. Classifiers: `Multilayer Perceptron (MLP)`.
3. Accuracy: `78.95%`.
4. F1-Score: `79%`.
5. Confusion Matrix:  
[[ 9  0  0  0  1]  
 [ 0  9  0  0  1]  
 [ 0  0  9  0  1]  
 [ 0  0  0 10  0]  
 [ 0  0  0  0 10]]  

![DenseNet201](https://github.com/BrenoFariasdaSilva/Simpsons-Family-Recognizer/blob/main/Results/DenseNet201/DenseNet201%20Accuracy.png)

### Results Analysis

An interesting observation is that the `NASNetLarge`, the most resource-intensive model with nearly half a million extracted features, did not perform well, achieving an accuracy of 52.63%.

Unfortunately, even when combining algorithms, the best-performing combination was the combination of K-NN, MLP, and Random Forest. However, none of them yielded satisfactory results, as there is often a significant difference (often greater than 10%) between the classifier with the best result and the second-best.

Regrettably, the attempt to enhance results by creating a data augmentation algorithm (`data_augmentation.py`) proved unsuccessful. The program aimed to generate a noisy copy for each image, but from my experience, it was too noisy. Although the `DenseNet201 model was employed again with the "augmented" data, the effort was futile, as depicted in the following image:

![DenseNet201 with Augmented Data](https://github.com/BrenoFariasdaSilva/Simpsons-Family-Recognizer/blob/main/Results/DenseNet201/AugmentedDenseNet201%20Accuracy.png)

Analyzing the confusion matrix of the best result, which utilized the Multilayer Perceptron (MLP) with non-augmented data and the DenseNet201 feature extraction model, we observe an accuracy of 78.95%. However, accuracy is not a reliable metric for imbalanced datasets. The F1-Score obtained in the best result was 0.79 or 79.00%.

The confusion matrix compares the actual classes of a classification model with the predicted classes. Each cell in the matrix represents the number of instances for a specific class. The values on the main diagonal represent correctly classified instances for each class. Larger values on the diagonal indicate better model performance.

Considering the analysis of the MLP confusion matrix (best result), as shown in Figure ~\ref{fig:DenseNet201MlpRfNb}, we can conclude the following:

- Class 1 (Bart): 3 instances were misclassified out of 35 instances, accounting for 8% error.
- Class 2 (Homer): 5 instances were misclassified out of 25 instances, representing 20% error.
- Class 3 (Lisa): 5 instances were misclassified out of 13 instances, with 4 errors attributed to class 2 and 1 to class 1, resulting in 38% error.
- Class 4 (Maggie): 5 instances were misclassified out of 12 instances, with 4 errors attributed to class 1 and 1 to class 2, resulting in 41% error.
- Class 5 (Marge): 2 instances were misclassified out of 35 instances, with 1 instance attributed to class 1 and 1 to class 2, resulting in 10% error.

This trend indicates that a lower number of samples per class correlates with a higher percentage of error. However, this may be influenced by instances in the dataset where certain images are mixed with other characters, impacting the models' ability to capture relevant class characteristics. Resolving these issues, particularly in classes with fewer instances, would likely improve accuracy.

Lastly, the results were really good, as the dataset was very small, with poluted images and, when compared to the other students, their best results were around 66% accuracy, so i'm very happy with the results i got, as there is a almost 13% difference between my best result and the best result of the other students. Also, once again, in an ideal world, i would have unlimited time and huge hardware resources to test all of the 9 deep learning models with the `best_parameters.py` algorithm, but i didn't have that, so i had to choose one of them and replicate the parameters to the other 8 deep learning models, so i'm sure that if i had the time and the hardware resources, i would have achieved even better results.

## Important Notes

In order to improve the accuracy of the models, i tried to implement the a python algorithm that read each image and generated a new image with the face centralized, but it works well for humans, but as The Simpsons characters are not humans, it didn't work as expected. So, i decided to not use it in the project, but you can find it in the following url: [Face Detection Python Algorithm](https://github.com/BrenoFariasdaSilva/Python/tree/main/Faces%20Detection).

Also, be aware that this is a very hardware intensive project, so if you want to fully run this project, you will need a very good hardware. I do have a Ryzen 3800X with 16 threads and 32GB of RAM and to run the `best_parameters.py` algorithm for all of the 9 deep learning models, it was predicted to take almost 48 hours to finish, so i only tested the `best_parameters.py` algorithm with the `ResNet18` dataset (which took about 3 hours) and replicated the found parameters to every of the others 8 deep learning models, but be aware that the right thing to do is to run the `best_parameters.py` algorithm for all of the 9 deep learning models and then run the `specific_parameters.py` algorithm with the best parameters found by the Grid Search for each algorithm.

Another thing that could improve the classifiers combination is to the majority vote be now only for the predicted label, but for the top 3 predicted labels, so the algorithm could be more accurate, but only the proper execution would tell us if it would be better or not. Futhermore, i could have used the `XGBoost` algorithm, which is a very powerful algorithm, but i didn't have time to implement it, so i left it for future improvements.

## Contributing

Code improvement recommendations are very welcome. In order to contribute, submit a Pull Request describing your code improvements.

## License

This project is licensed under the [Apache License 2.0](LICENSE), which allows you to use, modify, distribute, and sublicense this code, even for commercial purposes, as long as you include the original copyright notice and attribute you as the original author for the repository. See the [LICENSE](LICENSE) file for more details.
