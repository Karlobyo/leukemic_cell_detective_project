####### 👇 OPTIMIZED SOLUTION (x86)👇 #######

# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
# FROM tensorflow/tensorflow:2.10.0
# OR for apple silicon, use this base image instead
#FROM armswdev/tensorflow-arm-neoverse:r22.09-tf-2.10.0-eigen


FROM python:3.10.6-buster

#WORKDIR /prod

# RUN apt-get update && apt-get install -y \
# libhdf5-dev \
# && rm -rf /var/lib/apt/lists/*


# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
COPY requirements_prod.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY leukemic_det leukemic_det
COPY setup.py setup.py
RUN pip install .

#COPY Makefile Makefile
#RUN make reset_local_files

CMD uvicorn leukemic_det.api.fast:app --host 0.0.0.0 --port $PORT
