from setuptools import find_packages, setup

install_requires = []

setup(
    name="covidprognosis",
    author="Facebook AI Research",
    author_email="fastmri@fb.com",
    version="0.1",
    packages=find_packages(
        exclude=[
            "tests",
            "examples",
            "configs",
        ]
    ),
    setup_requires=["wheel"],
    install_requires=install_requires,
)