# TensorKiko

A fast and intuitive tool for visualizing and analyzing model structures from safetensors files, supporting tree-based visualizations and detailed parameter analysis.

# Python Install

## Creating and Setting Up the Virtual Environment

### For macOS and Linux

1. **Open your terminal.**
2. **Navigate to your project directory** (if you're not already there):
   ```bash
   cd /path/to/your/project
   ```
3. **Create the virtual environment:**
   ```bash
   python3.11 -m venv TensorKIKO
   ```
4. **Activate the virtual environment:**
   ```bash
   source TensorKIKO/bin/activate
   ```

### For Windows

1. **Open your Command Prompt or PowerShell.**
2. **Navigate to your project directory** (if you're not already there):
   ```bash
   cd \path\to\your\project
   ```
3. **Create the virtual environment:**
   ```bash
   python -m venv TensorKIKO
   ```
4. **Activate the virtual environment:**
   ```bash
   TensorKIKO\Scripts\activate
   ```

## Installing Packages

After activating the virtual environment, you can install the required packages using the `requirements.txt` file:

1. **Ensure you are in the virtual environment** (the command prompt should show `(TensorKIKO)` at the beginning).
2. **Install the packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Deactivating the Virtual Environment

To deactivate the virtual environment, simply run:

```bash
deactivate
```

# Using the Safetensors Visualization Script

The `visualize.py` script allows you to analyze and visualize the structure of models stored in safetensors format. Here's how to use it:

## Basic Usage

To run a basic analysis of a safetensors model file:

```bash
python visualize.py path/to/your/model.safetensors
```

This will output a summary of the model's structure and parameter count.

## Generating a Tree Diagram

To generate a tree diagram of the model structure, use the `--tree` flag:

```bash
python visualize.py path/to/your/model.safetensors --tree
```

This will create a dot file representing the model's tree structure, which can be visualized using tools like Graphviz.

## Debug Mode

To get detailed information about the script's execution and save it to a JSON file, use the `--debug` flag:

```bash
python visualize.py path/to/your/model.safetensors --debug
```

This will create a JSON file in the same directory as your model, containing detailed information about the analysis process.

## Combining Options

You can combine the `--tree` and `--debug` flags:

```bash
python visualize.py path/to/your/model.safetensors --tree --debug
```

This will generate both the tree diagram and the debug JSON file.

## Output

- Without any flags, the script will print the model analysis to the console.
- With `--tree`, it will also generate a `.dot` file for the tree structure.
- With `--debug`, it will create a `_debug.json` file with detailed execution information.

Note: Make sure you're in the activated virtual environment before running the script to ensure all dependencies are available.
