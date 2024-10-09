# TensorKiko

A fast and intuitive tool for visualizing and analyzing model structures from safetensors files, supporting tree-based visualizations and detailed parameter analysis.

## Installation

### Requirements

- Python 3.11 or higher

### Installation Methods

#### Using pip

To install TensorKiko with pip, run the following command:

```
pip install tensorkiko
```

#### Using Homebrew

To install TensorKiko using Homebrew, use the following commands:

```
brew tap takara-ai/tensorkiko https://github.com/takara-ai/TensorKiko
brew install tensorkiko
```

## Usage

After installation, you can use TensorKiko from the command line to visualize your model:

```
tensorkiko path/to/your/model.safetensors
```

TensorKiko can also attempt to convert `.ckpt` files to `.safetensors`. However, conversion will only succeed if the `.ckpt` file is in a standard model format without unique code elements such as SQLite.

For more options, use the help command:

```
tensorkiko --help
```

## Features

- Load and process safetensors files
- Generate interactive HTML visualizations of model structures
- Analyze model parameters, memory usage, and estimated FLOPs
- Search functionality for easy navigation of large models

## Contributing

We welcome contributions! If you'd like to contribute, please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
