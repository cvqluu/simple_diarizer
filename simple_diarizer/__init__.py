import os

__version__ = os.getenv("GITHUB_REF_NAME", "latest")
