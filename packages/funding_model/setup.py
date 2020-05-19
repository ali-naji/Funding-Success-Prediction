import pathlib
from setuptools import setup, find_packages

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent


def list_reqs(fname='requirements.txt'):
    with open(fname) as f:
        return f.read().splitlines()


with open('README.md') as f:
    readme = f.read().strip()

with open(PACKAGE_ROOT / 'funding_model' / 'VERSION') as f:
    _version = f.read().strip()

setup(name='funding-model',
      version=_version,
      packages=find_packages(exclude=('tests', 'docs')),
      install_requires=list_reqs(),
      python_requires='>=3.7.0',
      author='Ali Naji',
      url="https://github.com/ali-naji",
      license='MIT',
      author_email='anaji7@gatech.edu',
      description="Classification model to predict whether a kickstarter project will be successfully funded",
      long_description=readme,
      package_data={'funding_model': ['VERSION']},
      classifiers=['License :: OSI Approved :: MIT License',
                   'Programming Language :: Python',
                   'Programming Language:: Python:: 3.7',
                   'Programming Language:: Python:: 3.8',
                   'Programming Language:: Python:: 3.9']
      )
