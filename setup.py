from setuptools import find_packages, setup


def get_requirements(file_path: str) -> list[str]:
    with open(file_path, "r") as requirements_file:
        requirements = [line.strip() for line in requirements_file.readlines()]
    if "-e ." in requirements:
        requirements.remove("-e .")
    return requirements


setup(
    name="real-estate-ml",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements("requirments.txt"),
    author="Vasu",
    author_email="paliwalvasu8@gmail.com",
)
