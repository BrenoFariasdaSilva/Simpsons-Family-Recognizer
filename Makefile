all: dependencies data run

run:
	clear; time python3 ./main.py

dependencies:
	pip install colorama

get-dataset:
	chmod +x ./Setup-Dataset.sh
	./Setup-Dataset.sh
