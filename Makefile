all: dependencies dataset

best_parameters:
	clear; time python3 ./best_parameters.py

specific_parameters:
	clear; time python3 ./specific_parameters.py

dependencies:
	pip install colorama collection numpy scikit-learn tqdm
	pip install --upgrade threadpoolctl

dataset:
	chmod +x ./Setup-Dataset.sh
	./Setup-Dataset.sh
