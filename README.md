<div align="center">
  
# [Simpsons-Family-Recognizer.](https://github.com/BrenoFariasdaSilva/Simpsons-Family-Recognizer.git) <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg"  width="3%" height="3%">

</div>

<div align="center">
  
---

This project focuses on using the K-Nearest Neighbors (K-NN), Decision Trees (DT), Support Vector Machines (SVM), MultiLayer Perceptrons (MLP), Random Forest and Naive Bayes Supervised Learning Algorithms to recognize The Simpsons Family Members. The dataset used for this project can be downloaded from [here](https://drive.google.com/uc?export=download&id=1wVyUmsz150uKjOprxRA_4LtmTXDPRp1o) url.

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
		- [Dataset](#dataset)
		- [Data Description](#data-description)
	- [Requirements](#requirements)
	- [Setup](#setup)
		- [Clone the repository](#clone-the-repository)
		- [Install Dependencies](#install-dependencies)
		- [Get Dataset](#get-dataset)
	- [Usage](#usage)
	- [Results](#results)
	- [Contributing](#contributing)
	- [License](#license)

## Introduction

Classification problems in AI involve assigning predefined labels or categories to input data based on its features. The goal is to train models that can generalize patterns from labeled examples to accurately predict the class of new, unseen instances.  
In this project, we use the K-Nearest Neighbors (K-NN), Decision Trees (DT), Support Vector Machines (SVM), MultiLayer Perceptrons (MLP) and Random Forest Supervised Learning Algorithms to recognize The Simpsons Family Members. So, this project will extract features from Bart, Homer, Lisa Maggie e Marge characters of the Simpsons family, in order to use all of the previously mentioned algorithms to train them to recognize them from the features we extract with the labels we provide, which character is in the image is in the image name.  
The system will try to predict which image represents which member and will produce as output the accuracy and F1-Score [0 to 100%] and a confusion matrix (N × N) indicating the percentage of system hits and errors among the N classes.

### Machine Learning Supervised Classifiers

This project employs various machine learning supervised classifiers to recognize the Simpsons family members. Each classifier has unique characteristics and applications. Below are brief descriptions of the classifiers used in this project.

### K-Nearest Neighbors (K-NN)

K-Nearest Neighbors (K-NN) is a simple and widely used machine learning algorithm that falls under the category of supervised learning. It is a non-parametric and instance-based method used for classification and regression tasks. The fundamental idea behind K-NN is to make predictions based on the majority class or average value of the `k` nearest data points in the feature space. In other words, the algorithm classifies or predicts the output of an instance by considering the labels of its nearest neighbors that are the data points closest to that instance in a Cartesian plane. The K-NN doesn't have an explicit training and classification progress, so its classification time could be very slow, depending on the size of the dataset.  
The impleted K-NN is using the Grid Search to find the best parameters for the model and the parameters used are:
- `n_neighbors`: Number of neighbors to use by default for kneighbors queries -> `1, 3, 5 and 7`.
- `metric`: The distance metric to use for the tree. The selected metrics are `"euclidean", "manhattan" and "minkowski"`.

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
- C: Regularization parameter: It test the values from `2. ** np.arange(-5, 15, 2)`
- kernel: Specifies the kernel type to be used in the algorithm. The selected kernel are 'linear', 'poly', 'rbf' and 'sigmoid'.
- gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. The selected gamma define the influence of input vectors on the margins. The values are from `2. ** np.arange(3, -15, -2)`.

### Multi-Layer Perceptron (MLP)

Multi-Layer Perceptron (MLP) is a type of artificial neural network commonly used for classification and regression tasks. It consists of multiple layers of interconnected nodes (neurons) with each layer having a set of weights. MLPs can capture complex relationships in data and are known for their ability to model non-linear functions.

### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve predictive accuracy and control overfitting. Each tree in the forest is trained on a random subset of the data, and the final prediction is based on the majority vote or average of the individual tree predictions. Random Forest is robust and effective for both classification and regression tasks.

### Naive Bayes

Naive Bayes is a simple yet powerful machine learning algorithm commonly used for classification tasks. It is a probabilistic classifier that makes use of Bayes' Theorem, which states that the probability of A given B is equal to the probability of B given A times the probability of A divided by the probability of B. The algorithm it self is simple and easy to implement, and it is effective in high-dimensional spaces. It is also fast and can be used for both binary and multi-class classification tasks. It has a few drawbacks, such as the assumption of independent features and the zero-frequency problem. Also, it requires a parameter called "var_smoothing" to be set, which is a smoothing parameter that accounts for features not present in the learning samples and prevents zero probabilities in the prediction.

### Dataset

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

## Requirements

- Python 3.x
- Colorama
- Pandas
- NumPy
- Scikit-learn

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

### Get Dataset

1. Download the dataset from [here](https://drive.google.com/uc?export=download&id=1wVyUmsz150uKjOprxRA_4LtmTXDPRp1o) and place it in this project directory `(/Simpsons-Family-Recognizer)` and run the following command:

```bash
make dataset
```

This command will give execution permission to the `Setup-Dataset.sh` ShellScript and execute it. The `Setup-Dataset.sh` ShellScript will download the dataset from a url and unzip it to the `Dataset` directory and, lastly, remove the zip file. With that in mind, it basically does everything for you.

## Usage

In order to run the project, run the following command:

```bash
make run
```

## Results

The results of the K-NN, Decision Tree, Support Vector Machine, Multilayer Perceptron, Random Forest and Naive Bayes algorithms models will produce as output the accuracy, F1-Score [0 to 100%], it's Best Params found by the Grid Search, the execution time and a confusion matrix (N × N) indicating the percentage of system hits and errors among the N classes. That results will be shown in the terminal and saved in the `Results` directory.

## Contributing

Code improvement recommendations are very welcome. In order to contribute, follow the steps below:

## License

This project is licensed under the [Apache License 2.0](LICENSE), which allows you to use, modify, distribute, and sublicense this code, even for commercial purposes, as long as you include the original copyright notice and attribute you as the original author for the repository. See the [LICENSE](LICENSE) file for more details.
