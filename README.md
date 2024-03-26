<div align="center">
  
# [Simpsons-Family-Recognizer.](https://github.com/BrenoFariasdaSilva/Simpsons-Family-Recognizer.git) <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg"  width="3%" height="3%">

</div>

<div align="center">
  
---
This project focuses on using the K-Nearest Neighbors (K-NN), Decision Trees (DT), Support Vector Machines (SVM), MultiLayer Perceptrons (MLP), Random Forest, Naive Bayes and Combining the Supervised Learning Algorithms to recognize The Simpsons Family Members. The `raw dataset` (bmp file format images) used for this project can be downloaded from [here](https://drive.google.com/drive/folders/12TjdYNLIml8E-k9G5HwZ4wxmeNwXNRCo?usp=share_link), and the `features dataset` (txt format file with the extracted features from the raw dataset) can be downloaded from [here](https://drive.google.com/drive/folders/12dlVktROvILU-J8gMT7ee8GUZPfd2yNX?usp=sharing). Lastly, my results can be found in the `Results` directory of this repository or [here](https://github.com/BrenoFariasdaSilva/Simpsons-Family-Recognizer/tree/main/Results).

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
		- [Raw Dataset](#raw-dataset)
		- [Features Dataset](#features-dataset)
		- [Data Description](#data-description)
	- [Feature Extraction Models](#feature-extraction-models)
	- [Algorithms](#algorithms)
		- [Best Parameters Python Algorithm](#best-parameters-python-algorithm)
			- [Usage](#usage)
			- [Output Example](#output-example)
				- [Classification Report](#classification-report)
				- [Confusion Matrix](#confusion-matrix)
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
	- [Next Steps](#next-steps)
	- [Contributing](#contributing)
	- [License](#license)

## Introduction

Classification problems in AI involve assigning predefined labels or categories to input data based on its features. The goal is to train models that can generalize patterns from labeled examples to accurately predict the class of new, unseen instances.  
In this project, we use the K-Nearest Neighbors (K-NN), Decision Trees (DT), Support Vector Machines (SVM), MultiLayer Perceptrons (MLP), Random Forest, Naive Bayes and Combine those Supervised Learning Algorithms to recognize The Simpsons Family Members. So, this project will extract features from Bart, Homer, Lisa Maggie e Marge characters of the Simpsons family, in order to use all of the previously mentioned algorithms to train them to recognize them from the features we extract with the labels we provide, which character is in the image is in the image name.  
The system will try to predict which image represents which member and will produce as output the accuracy and F1-Score [0 to 100%] and a confusion matrix (N × N) indicating the percentage of system hits and errors among the N classes.

## Machine Learning Supervised Classifiers

This project employs various machine learning supervised classifiers to recognize the Simpsons family members. Each classifier has unique characteristics and applications. Below are brief descriptions of the classifiers used in this project.

Supervised learning classifiers play a crucial role in machine learning, enabling systems to make predictions or decisions based on labeled training data. While these classifiers offer several advantages, they also come with inherent limitations. Let's explore some of the key advantages and disadvantages:

Advantages:

1. **Accuracy:**
   - Supervised learning classifiers, once trained on labeled data, can provide accurate predictions on new, unseen data.

2. **Interpretability:**
   - Many classifiers, such as Decision Trees, offer interpretability, allowing users to understand the decision-making process.

3. **Versatility:**
   - Supervised learning supports a wide range of applications, including classification, regression, and even complex tasks like image and speech recognition.

4. **Efficiency:**
   - These classifiers can efficiently process large datasets and make predictions in real-time, making them suitable for various real-world scenarios.

5. **Adaptability:**
   - Supervised learning models can adapt and learn from new data, making them dynamic and applicable to evolving environments.

Disadvantages:

1. **Dependency on Labeled Data:**
   - Supervised learning classifiers heavily rely on labeled training data, and the quality of predictions is directly linked to the quality and representativeness of this data.

2. **Overfitting:**
   - Some classifiers, like Decision Trees, are prone to overfitting, where the model memorizes the training data but struggles with generalization to new data.

3. **Computational Complexity:**
   - Certain algorithms, such as Support Vector Machines, can be computationally expensive, especially with large datasets, impacting their scalability.

4. **Bias:**
   - Supervised learning classifiers may inherit biases present in the training data, leading to biased predictions, especially when the training data is not diverse or representative.

5. **Limited Handling of Noise:**
   - Noisy or irrelevant features in the training data can negatively impact the performance of classifiers, as they might learn patterns from the noise.

Understanding these advantages and disadvantages is essential for selecting the most suitable supervised learning classifier for a specific task and mitigating potential challenges in real-world applications.

### K-Nearest Neighbors (K-NN)

K-Nearest Neighbors (K-NN) is a simple and widely used machine learning algorithm that falls under the category of supervised learning. It is a non-parametric and instance-based method used for classification and regression tasks. The fundamental idea behind K-NN is to make predictions based on the majority class or average value of the `k` nearest data points in the feature space. In other words, the algorithm classifies or predicts the output of an instance by considering the labels of its nearest neighbors that are the data points closest to that instance in a Cartesian plane. The K-NN doesn't have an explicit training and classification progress, so its classification time could be very slow, depending on the size of the dataset.  
The implemented K-NN is using Grid Search to find the best parameters for the model, and the parameters used are:
- `metric`: The distance metric to use for the tree. The selected metrics are `"euclidean", "manhattan" and "minkowski"`.
- `n_neighbors`: Number of neighbors to use by default for kneighbors queries -> `1, 3, 5 and 7`.

**Advantages:**
- Simple and easy to understand.
- No training phase; predictions are made directly using the training dataset.

**Disadvantages:**
- Can be computationally expensive during the classification phase, especially with large datasets.
- Sensitive to irrelevant or redundant features.

### Decision Trees (DT)

Decision Trees are a versatile machine learning algorithm used for both classification and regression tasks. They make decisions based on a set of rules learned from the training data.  
Decision Trees recursively split the data into subsets based on the most significant features, creating a tree-like structure. They are interpretable and can capture complex relationships in the data.  
The implemented Decision Tree is using Grid Search to find the best parameters for the model, and the parameters used are:
- `criterion`: The function to measure the quality of a split. The selected criteria are `"gini" and "entropy"`.
- `max_depth`: The maximum depth of the tree. The selected max_depths are `None, 10, 20, 30`.
- `splitter`: The strategy used to choose the split at each node. The selected splitters are `"best" and "random"`.

**Advantages:**
- Simple to understand and interpret.
- Can handle both numerical and categorical data.

**Disadvantages:**
- Prone to overfitting, especially with deep trees.
- Sensitive to small variations in the data.

### Support Vector Machines (SVM)

Support Vector Machines (SVM) is a powerful supervised learning algorithm used for classification and regression tasks. SVM aims to find a hyperplane that best separates data into different classes. It works by maximizing the margin between classes, and it is effective in high-dimensional spaces.  
SVM can handle non-linear relationships through the use of kernel functions.  
The implemented SVM is using Grid Search to find the best parameters for the model, and the parameters used are:
- `C`: Regularization parameter: It tests values from `0.01, 0.1, 1, 10, 100`.
- `gamma`: Kernel coefficient. The selected gamma defines the influence of input vectors on the margins. The values are from `0.001, 0.01, 0.1, 1, 10`.
- `kernel`: Specifies the kernel type to be used in the algorithm. The selected kernels are `"linear", "poly", "rbf" and "sigmoid"`.

**Advantages:**
- Effective in high-dimensional spaces.
- Versatile and can handle non-linear decision boundaries.

**Disadvantages:**
- Can be sensitive to the choice of kernel and hyperparameters.
- Computationally expensive, especially with large datasets.

### Multi-Layer Perceptron (MLP)

Multi-Layer Perceptron (MLP) is a type of artificial neural network commonly used for classification and regression tasks. It consists of multiple layers of interconnected nodes (neurons) with each layer having a set of weights. MLPs can capture complex relationships in data and are known for their ability to model non-linear functions.  
The implemented MLP is using Grid Search to find the best parameters for the model, and the parameters used are:
- `alpha`: L2 penalty (regularization term) parameter. The selected alphas are `1e-5, 1e-4 and 1e-3`.
- `hidden_layer_sizes`: The number of neurons in the hidden layers. The selected hidden_layer_sizes are `(100,), (100, 100), (500, 500, 500, 500)`.
- `solver`: The solver for weight optimization. The selected solvers are `"adam" and "lbfgs"`.

**Advantages:**
- Capable of learning complex patterns and relationships.
- Effective for tasks with large amounts of data.

**Disadvantages:**
- Requires careful tuning of hyperparameters.
- Computationally intensive, especially with large networks.

### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve predictive accuracy and control overfitting. Each tree in the forest is trained on a random subset of the data, and the final prediction is based on the majority vote or average of the individual tree predictions. Random Forest is robust and effective for both classification and regression tasks.  
The implemented Random Forest is using Grid Search to find the best parameters for the model, and the parameters used are:
- `max_depth`: The maximum depth of the tree. The selected max_depths are `None, 10, 20, 30`.
- `n_estimators`: The number of trees in the forest. The selected n_estimators are `100, 500 and 1000`.

**Advantages:**
- Robust and less prone to overfitting.
- Can handle large datasets with high dimensionality.

**Disadvantages:**
- Lack of interpretability compared to individual decision trees.
- Training can be computationally expensive.

### Naive Bayes

Naive Bayes is a simple yet powerful machine learning algorithm commonly used for classification tasks. It is a probabilistic classifier that makes use of Bayes' Theorem, which states that the probability of A given B is equal to the probability of B given A times the probability of A divided by the probability of B. The algorithm itself is simple and easy to implement, and it is effective in high-dimensional spaces. It is also fast and can be used for both binary and multi-class classification tasks. It has a few drawbacks, such as the assumption of independent features and the zero-frequency problem. Also, it requires a parameter called "var_smoothing" to be set, which is a smoothing parameter that accounts for features not present in the learning samples and prevents zero probabilities in the prediction. "prior" is another parameter that can be set to specify the prior probabilities of the classes, and be aware that the sum of the priors must be 1.
The implemented Naive Bayes is using Grid Search to find the best parameters for the model, and the parameters used are:
- `priors`: Prior probabilities of the classes. The selected priors are `None and [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]`.
- `var_smoothing`: Portion of the largest variance of all features that is added to variances for calculation stability. The selected var_smoothing are `1e-9, 1e-8, 1e-7, 1e-6, 1e-5`.

**Advantages:**
- Simple and efficient for classification tasks.
- Performs well in high-dimensional spaces.

**Disadvantages:**
- Assumes independence between features, which may not hold in some cases.
- Sensitivity to the quality of the input data and the distribution of features.

## Combining the Supervised Learning Algorithms

The idea of combining the Supervised Learning Algorithms is to use the best parameters found by the Grid Search for each algorithm and combine them to create a new model. The new model will be trained with the best parameters found by the Grid Search for each algorithm and the prediction will be made by the majority vote of the predictions of each algorithm.

## Dataset

There is two types of datasets used in this project, the `raw dataset` and the `features dataset`.  

### Raw Dataset

The raw dataset is a set of images of the Simpsons family members. The images are in the `.bmp` format and are named with the name of the character in the photo followed by a number, so that will be considered the label. Those images are used to extract the features from them, so we can use those features to train the classifiers and then use them to predict the character in the image. You are able to download the raw dataset from [here](https://drive.google.com/drive/folders/12TjdYNLIml8E-k9G5HwZ4wxmeNwXNRCo?usp=share_link). Keep in mind that the raw dataset is very small and to extract the features from the images, you must select or run multiple (if not all) deep learning models that are available in the `feature_extraction.py`, which is very hardware and time intensive, so be aware of that. For comparison, the `DenseNet201` model took about 3 hours to extract the features from the images, so if you want to extract the features from all of the 9 deep learning models, it will took me about 24 hours to finish, using a Ryzen 3800X (8 Cores and 16 Threads) with 32GB of RAM. So, if you are time and hardware limited, i recommend you jump the feature extraction process and download the features dataset, which are explained in the next section. In case you chose to use the raw dataset and extract the features from it, you must place the raw dataset in the `Dataset` directory and extract it. After that, feel free to delete the `.zip` file.

### Features Dataset

The features dataset is a set of `.txt` files with the extracted features from the raw dataset. The features are extracted from the last layer of the pre-trained models, so we can use those features to train the classifiers and then use them to predict the character in the image. If you chose not to extract the features from the raw dataset, you don't need to run the `feature_extraction.py` algorithm, but you must download the features dataset from [here](https://drive.google.com/drive/folders/12dlVktROvILU-J8gMT7ee8GUZPfd2yNX?usp=sharing), place it in the `Dataset` directory and extract it. After that, feel free to delete the `.zip` file.

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

## Feature Extraction Models

The feature extraction process was conducted using various deep learning models, each yielding a specific number of features. This section details the models used and clarifies how the feature dimensions were obtained, especially highlighting the selection of layers that are not 1x1, where applicable. Below is an overview:

1. **DenseNet201**: 155,520 features. Feature extraction was performed using the output of the penultimate dense layer, which is not a 1x1 layer, allowing for a richer and more detailed representation.

2. **EfficientNetB0**: 103,680 features. Features were obtained from a specific intermediate layer preceding the classification layer, providing a high level of detail and diversity in features.

3. **InceptionV3**: 1,000 features. Features were extracted directly from the output layer, which is a 1x1 layer, suitable for representing the final classification.

4. **MobileNetV2**: 103,680 features. Similar to EfficientNetB0, extraction occurred from an intermediate layer that offers a detailed representation of image characteristics.

5. **NASNetLarge**: 487,872 features. This model exhibits a large number of features due to the use of a deep layer before the classification layer, capturing a broad range of details.

6. **ResNet18**: 512 features. Feature extraction was done using the last global average pooling layer, following standard practice for this model, ensuring the capture of essential information for classification.

7. **ResNet50**: 204,800 features. Features were extracted from a specific deep layer, providing a rich representation of the processed images.

8. **VGG16**: 1,000 features. Extraction was performed from the output layer, which is a 1x1 layer, focusing on the final characteristics for classification.

9. **Xception**: 204,800 features. A similar approach to ResNet50 was used, extracting features from an advanced layer to obtain a detailed representation.

The chosen layer for extraction is always the last layer of the model. The resulting output is stored in `.txt` files, where all columns, except the last one, correspond to the extracted features, and the last column represents the label of the instance. The main difference between the models used in feature extraction is the quantity of features, as indicated in the list above.

You can find the extracted features datasets from any of the mentioned models [here](https://drive.google.com/drive/folders/12TjdYNLIml8E-k9G5HwZ4wxmeNwXNRCo?usp=share_link). Please note that the size of the extracted features file is generally proportional to the number of features extracted from a model.

This approach ensures that each model contributes a unique perspective in the feature extraction process, optimizing the performance of the image recognition system. The specific dimensions of the features were chosen based on each model's ability to capture relevant information for the task at hand, considering both the depth of the representations and the specificity of the used layers.

## Algorithms

This section describes the algorithms used in this project and how to use them. Remember that, this project can use two different types of dataset. The first is the `raw dataset` and the second is the `features dataset`. Read the [Dataset](#dataset) section to understand the difference between them, as it impacts the usage of the algorithms.

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

Random Forest: 84.8%
Max Depth: 30
Number of Estimators: 100
Execution Time: 45.678 Seconds

Support Vector Machine: 82.3%
C: 10
Gamma: 0.1
Kernel: rbf
Execution Time: 42.035 Seconds

Multilayer Perceptron: 81.4%
Alpha: 0.0001
Hidden Layer Sizes: (500, 500, 500, 500)
Solver: adam
Execution Time: 320.035 Seconds

K-Nearest Neighbors: 80.3%
K-Nearbors: 1
Metric: Euclidean
Execution Time: 6.035 Seconds

Decision Tree: 65.5%
Criterion: gini
Max Depth: None
Splitter: best
Execution Time: 1.543 Seconds

Naive Bayes: 65.5%
Priors: None
Var Smoothing: 1e-09
Execution Time: 0.543 Seconds
```

Also, it can output, if the constants `SHOW_CLASSIFICATION_REPORT`, `SHOW_CONFUSION_MATRIX` and `SHOW_DATASET_INFORMATION` are set to `True`, the Classification Report, the Confusion Matrix and the Dataset Information, as shown below:

##### Classification Report

The classification report provides a comprehensive summary of a model's performance for each class in a classification problem. It includes the following metrics:

- **Precision**: The ratio of true positive predictions to the total predicted positives. It indicates the accuracy of positive predictions.
- **Recall (Sensitivity)**: The ratio of true positive predictions to the total actual positives. It measures the model's ability to capture all positive instances.
- **F1-Score**: The harmonic mean of precision and recall. It balances precision and recall, providing a single metric for model performance.
- **Support**: The number of actual occurrences of the class in the specified dataset. It is the number of true instances for each class.
- **Accuracy**: The overall accuracy of the model, indicating the proportion of correctly predicted instances.
- **Macro Avg**: The average of precision, recall, and F1-Score for all classes.
- **Weighted Avg**: The weighted average of precision, recall, and F1-Score for all classes.

```bash
          precision    recall  f1-score   support

Class 0       0.80      0.85      0.82       100
Class 1       0.75      0.70      0.72        80

accuracy                           0.78       180
macro avg      0.78      0.77      0.77       180
weighted avg   0.78      0.78      0.78       180
```

##### Confusion Matrix

The confusion matrix compares the actual classes of a classification model with the predicted classes. Each cell in the matrix represents the number of instances for a specific class. The values on the main diagonal represent correctly classified instances for each class. Larger values on the diagonal indicate better model performance.

```bash
Actual/Predicted   Class 0   Class 1   Class 2   Class 3   Class 4
   Class 0             9         0        0         0         1
   Class 1             0         9        0         0         1
   Class 2             0         0        9         0         1
   Class 3             0         0        0        10         0
   Class 4             0         0        0         0        10
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

Unfortunately, even when combining algorithms, the best-performing combination was the combination of K-Nearest Neighbors, Multilayer Perceptron, and Random Forest. However, none of them yielded satisfactory results, as there is often a significant difference (often greater than 10%) between the classifier with the best result and the second-best.

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

## Next Steps

- [ ] Implement the `XGBoost`, `Ada Boost`, `ExtraTrees`, and many more algorithms found in [here](https://scikit-learn.org/stable/modules/ensemble.html).
- [ ] Implement the `Face Detection` algorithm, with something like the `YOLO` algorithm to recognize the faces of the characters and use that info to cut the image and use only the face of the character.
- [ ] Improve the `Data Augmentation` algorithm.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. If you have suggestions for improving the code, your insights will be highly welcome.
In order to contribute to this project, please follow the guidelines below or read the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details on how to contribute to this project, as it contains information about the commit standards and the entire pull request process.
Please follow these guidelines to make your contributions smooth and effective:

1. **Set Up Your Environment**: Ensure you've followed the setup instructions in the [Setup](#setup) section to prepare your development environment.

2. **Make Your Changes**:
   - **Create a Branch**: `git checkout -b feature/YourFeatureName`
   - **Implement Your Changes**: Make sure to test your changes thoroughly.
   - **Commit Your Changes**: Use clear commit messages, for example:
     - For new features: `git commit -m "FEAT: Add some AmazingFeature"`
     - For bug fixes: `git commit -m "FIX: Resolve Issue #123"`
     - For documentation: `git commit -m "DOCS: Update README with new instructions"`
     - For refactors: `git commit -m "REFACTOR: Enhance component for better aspect"`
     - For snapshots: `git commit -m "SNAPSHOT: Temporary commit to save the current state for later reference"`
   - See more about crafting commit messages in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

3. **Submit Your Contribution**:
   - **Push Your Changes**: `git push origin feature/YourFeatureName`
   - **Open a Pull Request (PR)**: Navigate to the repository on GitHub and open a PR with a detailed description of your changes.

4. **Stay Engaged**: Respond to any feedback from the project maintainers and make necessary adjustments to your PR.

5. **Celebrate**: Once your PR is merged, celebrate your contribution to the project!

## License

This project is licensed under the [Apache License 2.0](LICENSE), which allows you to use, modify, distribute, and sublicense this code, even for commercial purposes, as long as you include the original copyright notice and attribute you as the original author for the repository. See the [LICENSE](LICENSE) file for more details.
