all: depencencies run

run:
	clear; time python3 ./main.py

depencencies:
	pip install colorama

dataset:
	chmod +x ./Dataset.sh
	./Dataset.sh
