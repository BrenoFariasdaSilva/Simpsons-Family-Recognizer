all: dependencies dataset run

run:
	clear; time python3 ./main.py

dependencies:
	pip install colorama collection numpy scikit-learn tqdm
	pip install --upgrade threadpoolctl

dataset:
	chmod +x ./Setup-Dataset.sh
	./Setup-Dataset.sh
