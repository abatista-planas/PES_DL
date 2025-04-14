from setuptools import setup

requirements = [
    'importlib-metadata; python_version == "3.12"',
    "scikit-learn>=1.1.1",
    'torch',
]

requirements_dev = [
    "black",
    "ruff",
    "isort",
    "jupyter",
    "pre-commit",
    "pytest",
    "pytest-cov",
]

setup(
    name="PES_DL",
    version="0.2.0",
    description="Deep Learning model to construct Potential Energy Surfaces",
    url="https://github.com/abatista-planas/PES_DL.git",
    author="Adrian Batista",
    packages=["pes_1D"],
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dev": requirements_dev,
    },
)

