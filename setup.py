from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name='bayesianize',
    version='0.0.1',
    url='https://github.com/TODO',
    author=['Hippolyt Ritter', 'Martin Kukla', 'Cheng Zhang', 'Yingzhen Li'],
    description='Lightweight BNN wrapper for pytorch.',
    packages=find_packages(),
    install_requires=requirements,
    license="MIT"
)
