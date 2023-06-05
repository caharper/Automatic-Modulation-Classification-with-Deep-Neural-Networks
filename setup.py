from setuptools import setup, find_packages
from setuptools.dist import Distribution
import pathlib


HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()


setup(
    name="amc_neural_networks",
    version="0.1",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/caharper/Automatic-Modulation-Classification-with-Deep-Neural-Networks.git",
    author="Clay Harper",
    author_email="caharper@smu.edu",
    install_requires=[],
    extras_require={},
    python_requires=">=3.7",
    distclass=Distribution,
    packages=find_packages(exclude=("*_test.py",)),
    include_package_data=True,
)
