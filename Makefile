.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################

# install

reinstall_package:
	@pip uninstall -y leukemic-det
	@pip install -e .



# local api testing

run_api:
	@uvicorn leukemic_det.api.fast:app --port 8000

# local streamlit

streamlit:
	@streamlit run leukemic_det/webinterface/Intro.py


clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -f */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc

all: install clean
