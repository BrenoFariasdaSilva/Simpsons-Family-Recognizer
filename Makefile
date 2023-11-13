all: dependencies data run

run:
	clear; time python3 ./main.py

dependencies:
	pip install colorama

data:
	chmod +x ./Dataset.sh
	./Dataset.sh
