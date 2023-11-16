<div align="center">
  
# [Simpsons-Family-Recognizer.](https://github.com/BrenoFariasdaSilva/Simpsons-Family-Recognizer.git) <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg"  width="3%" height="3%">

</div>

<div align="center">
  
---

This project focuses on using the K-Nearest Neighbors (K-NN), Decision Trees (DT), Support Vector Machines (SVM), MultiLayer Perceptrons (MLP) and Random Forest Supervised Learning Algorithms to recognize The Simpsons Family Members. The dataset used for this project is stored in [this](https://drive.google.com/uc?export=download&id=1wVyUmsz150uKjOprxRA_4LtmTXDPRp1o) url. To summary, this project will extract features from Bart, Homer, Lisa Maggie e Marge characters of the Simpsons family, in order to use all of the previously mentioned algorithms to train them to recognize from the features we extract with the labels we provide, which character is in the image is in the image name. The system will try to predict which image represents which member and Twill produce as output the accuracy and F1-Score [0 to 100%] and a confusion matrix (N × N) indicating the percentage of system hits and errors among the N classes.

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
		- [K-Nearest Neighbors - K-NN](#k-nearest-neighbors---k-nn)
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
In this project, we use the K-Nearest Neighbors (K-NN) algorithm to detect credit card fraud, but there are many others supervised learning algorithms that can be used to solve classification problems, such as Decision Trees, Support Vector Machines (SVM), Multilayer Perceptron (MLP), Random Forest, etc.  

### K-Nearest Neighbors - K-NN
K-Nearest Neighbors (K-NN) is a simple and widely used machine learning algorithm that falls under the category of supervised learning. It is a non-parametric and instance-based method used for classification and regression tasks.  
The fundamental idea behind K-NN is to make predictions based on the majority class or average value of the 'k' nearest data points in the feature space. In other words, the algorithm classifies or predicts the output of an instance by considering the labels of its nearest neighbors that are the data points closest to that instance in a cartesian plane.


### Dataset
The dataset used for this project can be found on Kaggle [here](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023).

### Data Description

The dataset contains a mixture of legitimate and fraudulent transactions, with features that have been transformed to maintain confidentiality. Features include time, amount of the transaction, and anonymized numerical input variables, V1-V28. The target variable is a binary variable, Class, which denotes whether a transaction is fraudulent (1) or legitimate (0).  
As for today, day 13/11/2023, the dataset contains 500k+ rows and 31 columns.

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
make data
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
