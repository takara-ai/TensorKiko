import setuptools
import os

def get_version():
    version = os.environ.get("GITHUB_REF")
    if version:
        return version.split("/")[-1].lstrip("v")
    return "0.0.0"  # default version if not running in GitHub Actions

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="tensorkiko",
    version=get_version(),
    author="takara-ai",
    author_email="jordan@takara.ai",
    description="A fast and intuitive tool for visualizing and analyzing model structures from safetensors files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/takara-ai/TensorKiko",
    packages=setuptools.find_packages(),
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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "tensorkiko=tensorkiko.visualize:main",
        ],
    },
)