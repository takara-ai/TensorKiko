import os
from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements from requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="tensorkiko",
    version="0.1.0",  # Update this with your current version
    author="takara-ai",
    author_email="jordan@takara.ai",
    description="A fast and intuitive tool for visualizing and analyzing model structures from safetensors files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/takara-ai/TensorKiko",
    packages=find_packages(include=['tensorkiko', 'tensorkiko.*']),
    include_package_data=True,
    package_data={
        'tensorkiko': [
            'static/templates/*.html',
            'static/css/*.css',
            'static/js/*.js',
        ],
    },
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "tensorkiko=tensorkiko.visualize:main",
        ],
    },
)