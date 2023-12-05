.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y leukemic_cell_detective
	@pip install -e .

run_preprocess_and_train:
	python -c 'from leukemic_det.interface.main import preprocess_and_train; preprocess_and_train()'

run_pred:
	python -c 'from leukemic_det.interface.main import pred; pred()'

run_evaluate:
	python -c 'from leukemic_det.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

run_workflow:
	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m taxifare.interface.workflow


run_api:
	uvicorn leukemic_det.api.fast:app --reload


clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -f */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc
all: install clean


reset_local_files:
	rm -rf ${ML_DIR}
	mkdir ~/.leukemic/mlops/training_outputs
	mkdir ~/.leukemic/mlops/training_outputs/metrics
	mkdir ~/.leukemic/mlops/training_outputs/models
	mkdir ~/.leukemic/mlops/training_outputs/params

show_sources_all:
	-ls -laR ~/.lewagon/mlops/data
	-bq ls ${BQ_DATASET}
	-bq show ${BQ_DATASET}.processed_1k
	-bq show ${BQ_DATASET}.processed_2k
	-bq show ${BQ_DATASET}.processed_all
	-gsutil ls gs://${BUCKET_NAME}
