# TensorKiko Development Instructions

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/takara-ai/TensorKiko.git
   cd TensorKiko
   ```

2. **Install the package locally:**

   To install the package in development mode (with editable source code), run:

   ```bash
   pip install -e .
   ```

3. **Install the required dependencies:**

   Make sure to install the project dependencies from the `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Running as a Package

To run the project as a package (recommended for development):

```bash
python -m tensorkiko.visualize
```

This method ensures that Python can resolve the package structure correctly and allows you to use absolute imports.

### Running Scripts Directly

If you prefer to run scripts directly (e.g., `visualize.py`), you can use one of the following approaches:

1. **Modify `PYTHONPATH`:**

   Add the project root to your `PYTHONPATH` environment variable:

   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   python tensorkiko/visualize.py
   ```

2. **Relative Imports:**

   The project is set up to work with relative imports when running as a package. However, if running directly, ensure you have set the correct module path.

3. **Compatibility Helper:**

   The `visualize.py` script includes a helper to adjust `sys.path` automatically if run directly. No extra setup is needed, but running as a package is still recommended for consistent behavior.

## Building the Package

To build the distribution files for the project (source and wheel):

```bash
python setup.py sdist bdist_wheel
```

## Contribution Guidelines

- Ensure all imports are relative or absolute according to the context.
- Test the package by installing locally with `pip install -e .` after any code changes.
- Make sure to run tests after modifying key parts of the project.
