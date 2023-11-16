<div align="center">
  
# [Simpsons-Family-Recognizer.](https://github.com/BrenoFariasdaSilva/Simpsons-Family-Recognizer.git) <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg"  width="3%" height="3%">

</div>

<div align="center">
  
---

This project focuses on using the K-Nearest Neighbors (K-NN), Decision Trees (DT), Support Vector Machines (SVM), MultiLayer Perceptrons (MLP) and Random Forest Supervised Learning Algorithms to recognize The Simpsons Family Members. The dataset used for this project can be downloaded from [here](https://drive.google.com/uc?export=download&id=1wVyUmsz150uKjOprxRA_4LtmTXDPRp1o) url.

---

</div>

<div align="center">

![GitHub Code Size in Bytes](https://img.shields.io/github/languages/code-size/BrenoFariasdaSilva/Simpsons-Family-Recognizer)
![GitHub Last Commit](https://img.shields.io/github/last-commit/BrenoFariasdaSilva/Simpsons-Family-Recognizer)
![GitHub](https://img.shields.io/github/license/BrenoFariasdaSilva/Simpsons-Family-Recognizer)
![wakatime](https://wakatime.com/badge/github/BrenoFariasdaSilva/Simpsons-Family-Recognizer.svg)

</div>

<div align="center">
  
![RepoBeats Statistics](https://repobeats.axiom.co/api/embed/2a2bfd10cfdfee1520cda5c7aeb0a8555c58334a.svg "Repobeats analytics image")

</div>

## Table of Contents
- [Simpsons-Family-Recognizer. ](#simpsons-family-recognizer-)
	- [Table of Contents](#table-of-contents)
	- [Introduction](#introduction)
		- [Machine Learning Supervised Classifiers](#machine-learning-supervised-classifiers)
		- [K-Nearest Neighbors (K-NN)](#k-nearest-neighbors-k-nn)
		- [Decision Tress (DT)](#decision-tress-dt)
		- [Support Vector Machines (SVM)](#support-vector-machines-svm)
		- [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
		- [Random Forest](#random-forest)
		- [Dataset](#dataset)
		- [Data Description](#data-description)
	- [Requirements](#requirements)
	- [Setup](#setup)
		- [Clone the repository](#clone-the-repository)
		- [Dependencies](#dependencies)
		- [Dataset](#dataset-1)
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

K-Nearest Neighbors (K-NN) is a simple and widely used machine learning algorithm that falls under the category of supervised learning. It is a non-parametric and instance-based method used for classification and regression tasks. The fundamental idea behind K-NN is to make predictions based on the majority class or average value of the 'k' nearest data points in the feature space. In other words, the algorithm classifies or predicts the output of an instance by considering the labels of its nearest neighbors that are the data points closest to that instance in a Cartesian plane. The K-NN doesn't have an explicit training and classification progress, so its classification time could be very slow, depending on the size of the dataset.

### Decision Tress (DT)

Decision Trees are a versatile machine learning algorithm used for both classification and regression tasks. They make decisions based on a set of rules learned from the training data.  
Decision Trees recursively split the data into subsets based on the most significant features, creating a tree-like structure. They are interpretable and can capture complex relationships in the data.

### Support Vector Machines (SVM)

Support Vector Machines (SVM) is a powerful supervised learning algorithm used for classification and regression tasks. SVM aims to find a hyperplane that best separates data into different classes. It works by maximizing the margin between classes, and it is effective in high-dimensional spaces.  
SVM can handle non-linear relationships through the use of kernel functions.

### Multi-Layer Perceptron (MLP)

Multi-Layer Perceptron (MLP) is a type of artificial neural network commonly used for classification and regression tasks. It consists of multiple layers of interconnected nodes (neurons) with each layer having a set of weights. MLPs can capture complex relationships in data and are known for their ability to model non-linear functions.

### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve predictive accuracy and control overfitting. Each tree in the forest is trained on a random subset of the data, and the final prediction is based on the majority vote or average of the individual tree predictions. Random Forest is robust and effective for both classification and regression tasks.

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
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Setup

### Clone the repository
1. Clone the repository with the following command:

```bash
git clone https://github.com/BrenoFariasdaSilva/Simpsons-Family-Recognizer.git
cd Simpsons-Family-Recognizer
```

### Dependencies
1. Install the project dependencies with the following command:

```bash
make dependencies
```

### Dataset
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) and place it in this project directory `(/Repository-Name)` and run the following command:

```bash
make dataset
```
This command will give execution permission to the `Dataset.sh` ShellScript and execute it. The `Dataset.sh` ShellScript will unzip the `archive` file, rename the extracted folder to `dataset` and delete the `archive` file. Also, inside of the `dataset` folder, the `Dataset.sh` ShellScript will rename the `creditcard_2023.csv` file to `Simpsons-Family-Recognizer.csv.

## Usage

In order to run the project, run the following command:

```bash
make run
```

## Results

The results of the K-NN model in terms of accuracy will be outputted to the console and saved to the `results` directory.

## Contributing
Code improvement recommendations are very welcome. In order to contribute, follow the steps below:

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
