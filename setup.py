from setuptools import find_packages,setup

def get_requirments(file):
    with open(file, 'r') as file:
        requirements = [line.strip() for line in file.readlines()]
    if "-e ." in requirements:
        requirements.remove("-e .")
    return requirements


setup(  
    name="Datapredictor",
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_requirments('requirments.txt'),
    author="Vasu",
    author_email="paliwalvasu8@gmail.com"
)

