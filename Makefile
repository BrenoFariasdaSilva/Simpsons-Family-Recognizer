all: depencencies run

run:
	clear; time python3 ./main.py

depencencies:
	pip install colorama

data:
	chmod +x ./Dataset.sh
	./Dataset.sh
