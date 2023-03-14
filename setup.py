from setuptools import setup
from setuptools import find_packages


with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='leukemic_cell_detective',
      description="package description",
      packages=find_packages(),
      install_requires=requirements)
