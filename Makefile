all: dependencies data run

run:
	clear; time python3 ./main.py

dependencies:
	pip install colorama

dataset:
	chmod +x ./Setup-Dataset.sh
	./Setup-Dataset.sh
