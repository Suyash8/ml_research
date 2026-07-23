from setuptools import setup, find_packages

setup(
    name="ml_research",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy"
    ],
)
