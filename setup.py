import setuptools

from simple_diarizer import __version__


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read(

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.read()

setuptools.setup(
    version=__version__,
    long_description=long_description,
    install_requires=install_requires,
)
