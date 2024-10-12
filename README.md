# TensorKiko

[![Release TensorKiko](https://github.com/takara-ai/TensorKiko/actions/workflows/release.yml/badge.svg)](https://github.com/takara-ai/TensorKiko/actions/workflows/release.yml)

TensorKiko is a powerful and intuitive tool for visualizing and analyzing machine learning model structures. It supports various model formats and provides detailed insights into model architecture, parameters, and tensor statistics.

## Features

- **Multi-format Support**: Load and process models in .safetensors, .pt, .pth, .pb, and .h5 formats.
- **Interactive Visualization**: Generate HTML-based visualizations of model structures with a tree-based layout.
- **Detailed Analysis**:
  - Model parameters, memory usage, and estimated FLOPs.
  - Tensor statistics (mean, std dev, min/max values, zero count) with histograms.
  - SVG representation of tensor shapes.
- **Anomaly Detection**: Automatically detect and highlight potential issues in model tensors.
- **Search Functionality**: Easily navigate large models.
- **Custom Layer Filtering**: Include or exclude specific layers using regex patterns.
- **Precision Information**: Display the data type of model parameters.
- **Web-based Interface**: User-friendly interface with collapsible sections.

## UI Examples

### Model Overview

![TensorKiko Model Overview](https://github.com/takara-ai/TensorKiko/blob/main/media/images/ui/detail_tab.jpg)
This image shows the main interface, displaying:

- Model name and overall details
- Breakdown of layer types
- Beginning of the model's hierarchical structure

### Layer Details

![TensorKiko Layer Details](https://github.com/takara-ai/TensorKiko/blob/main/media/images/ui/layer_detail.jpg)
This image demonstrates the detailed view of a specific layer, including:

- Hierarchical model structure
- Layer information (parameters, shape, statistics)
- Histogram of weight distribution

## Installation and Usage

### Requirements

- Python 3.11 or higher

### Installation

```bash
pip install tensorkiko
```

### Basic Usage

```bash
tensorkiko path/to/your/model.safetensors
```

For multiple input files:

```bash
tensorkiko path/to/model1.pt path/to/model2.safetensors path/to/model3.h5
```

### Command-line Options

- `--debug`: Enable debug mode
- `--no-tree`: Disable tree visualization
- `--port PORT`: Specify HTTP server port (default: 8000)
- `--output-dir DIR`: Set output directory
- `--include-layers REGEX`: Include only specific layers
- `--exclude-layers REGEX`: Exclude specific layers

Example:

```bash
tensorkiko path/to/model.pt --debug --port 8080 --include-layers "conv|linear"
```

### Web Interface Guide

1. Run TensorKiko on your model(s).
2. A web browser will open with the visualization.
3. Explore the collapsible header for model information and layer type statistics.
4. Click tree nodes to view detailed layer information.
5. Use the search bar to find specific layers or parameters.
6. Examine tensor statistics, histograms, and shape visualizations.
7. Check for highlighted anomalies in the interface.

## Contributing

We welcome contributions! Please submit a Pull Request on our GitHub repository.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

TensorKiko is developed by the open-source community and takara.ai staff. The project is sponsored by [takara.ai](https://takara.ai).
