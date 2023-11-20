all: dependencies dataset run

run:
	clear; time python3 ./main.py

dependencies:
	pip install colorama numpy scikit-learn
	pip install --upgrade threadpoolctl

dataset:
	chmod +x ./Setup-Dataset.sh
	./Setup-Dataset.sh
