"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "pandas",
    "Pillow",
    "pytorch-lightning",
    "pyyaml",
    "scipy",
    "torch",
    "torchvision",
    "tqdm",
]

setup(
    name="covidprognosis",
    author="Facebook AI Research",
    author_email="mmuckley@fb.com",
    version="0.1",
    packages=find_packages(exclude=["tests", "cp_examples", "configs"]),
    setup_requires=["wheel"],
    install_requires=install_requires,
)
