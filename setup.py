"""
Setup script for the FrankaPanda package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="frankapanda",
    version="0.1.0",
    author="Kallol Saha",
    author_email="kallolsaharesearch@gmail.com",
    description="Franka Panda robot control and dual Azure Kinect perception",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pyk4a",
        "zmq",
        "pickle-mixin",
        "open3d",
        "deoxys",
        "robo-utils",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        'console_scripts': [
            'frankapanda-perception=frankapanda.perception.perception_pipeline:main',
            'frankapanda-capture=frankapanda.perception.capture_single_camera:main',
        ],
    },
)
