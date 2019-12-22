from setuptools import setup, find_packages

install_requires = [
    # "gurobipy",  # install this manually
    "matplotlib>=2.2,<2.3",
    "numpy",
    "click==6.7",
    "pyyaml",
    "jsonpickle",
    "unidecode",
    "networkx",
]

setup(
    name="alib",
    # version="0.1",
    packages=["alib"],
    package_data={"alib": ["data/topologyZoo/*.yml"]},
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "alib = alib.cli:cli",
        ]
    }
)
