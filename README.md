## Overview

This project focuses on using the K-Nearest Neighbors (K-NN) algorithm to detect credit card fraud. The dataset used for this project is sourced from Kaggle and contains anonymized credit card transactions labeled as either fraudulent or legitimate.

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
