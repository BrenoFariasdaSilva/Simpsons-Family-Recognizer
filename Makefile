all: dependencies

best_parameters:
	clear; time python3 ./best_parameters.py

data_augmentation:
	clear; time python3 ./data_augmentation.py

features_extraction:
	clear; time python3 ./features_extraction.py

specific_parameters:
	clear; time python3 ./specific_parameters.py

dependencies:
	pip install colorama collection imgaug numpy scikit-learn tensorflow tqdm
	pip install --upgrade threadpoolctl
