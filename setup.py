from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="lci-framework",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A framework for evolutionary experiments on Losslessness-Compression-Invariance trade-offs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lci-framework",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lci-framework=lci_framework.main:main",
        ],
    },
) 