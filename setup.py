import setuptools

from simple_diarizer import __version__


with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.read()

setuptools.setup(
    version=__version__,
    install_requires=install_requires,
)
