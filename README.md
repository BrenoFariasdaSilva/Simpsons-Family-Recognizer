<div align="center">
  
# [Credit Card Fraud 2023](https://github.com/BrenoFariasdaSilva/Credit-Card-Fraud-2023.git) <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg"  width="3%" height="3%">

</div>

<div align="center">
  
---

This project focuses on using the K-Nearest Neighbors (K-NN) algorithm to detect credit card fraud. The dataset used for this project is sourced from Kaggle and contains anonymized credit card transactions labeled as either fraudulent or legitimate.
  
---

</div>

<div align="center">

![GitHub Code Size in Bytes](https://img.shields.io/github/languages/code-size/BrenoFariasdaSilva/Credit-Card-Fraud-2023)
![GitHub Last Commit](https://img.shields.io/github/last-commit/BrenoFariasdaSilva/Credit-Card-Fraud-2023)
![GitHub](https://img.shields.io/github/license/BrenoFariasdaSilva/Credit-Card-Fraud-2023)
![wakatime](https://wakatime.com/badge/github/BrenoFariasdaSilva/Credit-Card-Fraud-2023.svg)

</div>

<div align="center">
  
![RepoBeats Statistics](https://repobeats.axiom.co/api/embed/2a2bfd10cfdfee1520cda5c7aeb0a8555c58334a.svg "Repobeats analytics image")

</div>

## Dataset

The dataset used for this project can be found on Kaggle [here](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023).

### Data Description

The dataset contains a mixture of legitimate and fraudulent transactions, with features that have been transformed to maintain confidentiality. Features include time, amount of the transaction, and anonymized numerical input variables.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/BrenoFariasdaSilva/Credit-Card-Fraud-2023.git
   cd Credit-Card-Fraud-2023
	```

2. Install the required packages:

	```bash
	make dependencies
	```

3. Download the dataset from Kaggle and place it in this project directory and run the following command:

	```bash
	make dataset
	```

## Usage

In order to run the project, run the following command:

```bash
make run
```

## Results

The results of the K-NN model in terms of accuracy will be outputted to the console and saved to the `results` directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kaggle for providing the dataset.
- Scikit-learn and other open-source contributors for their valuable libraries.
- The broader data science community for inspiration and guidance.
