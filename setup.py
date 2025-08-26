#!/usr/bin/env python3
"""
Setup script for SAR to EO Image Translation using CycleGAN
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "SAR to EO Image Translation using CycleGAN"

# Read requirements from requirements.txt
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = []
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
            return requirements
    except FileNotFoundError:
        return [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "numpy>=1.21.0",
            "rasterio>=1.2.0",
            "Pillow>=8.3.0",
            "tqdm>=4.62.0",
            "matplotlib>=3.4.0",
            "scikit-image>=0.18.0",
        ]

setup(
    name="sar-to-eo-translation",
    version="1.0.0",
    author="Taksh Pal, Ritij Raj",
    author_email="taksh.pal@example.com, ritij.raj@example.com",
    description="SAR to EO Image Translation using CycleGAN",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SAR-to-EO-Translation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/SAR-to-EO-Translation/issues",
        "Source": "https://github.com/yourusername/SAR-to-EO-Translation",
        "Documentation": "https://github.com/yourusername/SAR-to-EO-Translation/wiki",
    },
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    keywords="cyclegan, sar, eo, image-translation, deep-learning, pytorch, satellite-imagery, remote-sensing",
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "pre-commit>=2.15.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.1.0",
            "ipywidgets>=7.6.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "gpu": [
            "torch==1.9.0+cu111",
            "torchvision==0.10.0+cu111",
        ],
    },
    entry_points={
        "console_scripts": [
            "sar-to-eo-train=src.training.trainer:main",
            "sar-to-eo-preprocess=src.data.preprocessing:main",
            "sar-to-eo-evaluate=src.evaluation.metrics:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.txt", "*.md"],
    },
    zip_safe=False,
    platforms=["any"],
    license="MIT",
    test_suite="tests",
)
