# Name of the virtual environment directory
VENV := venv
# Python command to use
PYTHON := python3

.PHONY: all venv dependencies

all: venv dependencies

venv: $(VENV)/bin/activate

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	touch $(VENV)/bin/activate

best_parameters:
	clear; time $(VENV)/bin/python ./best_parameters.py

data_augmentation:
	clear; time $(VENV)/bin/python ./data_augmentation.py

features_extraction:
	clear; time $(VENV)/bin/python ./features_extraction.py

specific_parameters:
	clear; time $(VENV)/bin/python ./specific_parameters.py

dependencies:
	$(VENV)/bin/pip install colorama collection imgaug numpy opencv-python scikit-learn tensorflow tqdm
	$(VENV)/bin/pip install --upgrade threadpoolctl
