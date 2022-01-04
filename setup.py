import os

import setuptools

version = os.getenv("GITHUB_REF", "latest")

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.read()

setuptools.setup(
    version=version,
    install_requires=install_requires,
)