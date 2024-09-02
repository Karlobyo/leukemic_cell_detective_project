from setuptools import setup
from setuptools import find_packages


with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='leukemic_det',
      description="package description",
      packages=find_packages(),#include=['leukemic_det', 'leukemic_det.*']),
      install_requires=requirements)
