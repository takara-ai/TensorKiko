import torch
import tensorflow as tf
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import os
import logging
from tqdm import tqdm
import time
import io

logger = logging.getLogger(__name__)

def load_pytorch_model(file_path):
    """
    Load a PyTorch model (.pt or .pth) and return its state dict.
    """
    try:
        file_size = os.path.getsize(file_path)
        
        # Custom file-like object to track progress
        class ProgressReader(io.BytesIO):
            def __init__(self, file_path):
                self.file_path = file_path
                self.file = open(file_path, 'rb')
                self.progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading PyTorch model")
                self.len = file_size

            def read(self, size=-1):
                chunk = self.file.read(size)
                self.progress_bar.update(len(chunk))
                return chunk

            def tell(self):
                return self.file.tell()

            def seek(self, offset, whence=io.SEEK_SET):
                return self.file.seek(offset, whence)

            def close(self):
                self.file.close()
                self.progress_bar.close()

        with ProgressReader(file_path) as f:
            state_dict = torch.load(f, map_location='cpu', weights_only=True)

        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            return state_dict['state_dict']
        return state_dict
    except Exception as e:
        logger.error(f"Failed to load PyTorch model from {file_path}: {e}")
        return None

def load_tensorflow_model(file_path):
    """
    Load a TensorFlow model (.pb or .h5) and return a dict of its weights.
    """
    try:
        file_size = os.path.getsize(file_path)
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading TensorFlow model") as pbar:
            def progress_hook(_, current, total):
                pbar.update(current - pbar.n)
            model = tf.keras.models.load_model(file_path, custom_objects={'progress_hook': progress_hook})
        return {weight.name: weight.numpy() for layer in model.layers for weight in layer.weights}
    except Exception as e:
        logger.error(f"Failed to load TensorFlow model from {file_path}: {e}")
        return None

def load_safetensors_model(file_path):
    """
    Load a safetensors model and return its state dict.
    """
    try:
        file_size = os.path.getsize(file_path)
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading safetensors model") as pbar:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                state_dict = {}
                for key in tqdm(f.keys(), desc="Loading tensors"):
                    state_dict[key] = f.get_tensor(key)
                    pbar.update(f.get_tensor(key).nbytes)
        return state_dict
    except Exception as e:
        logger.error(f"Failed to load safetensors model from {file_path}: {e}")
        return None

def convert_to_safetensors(state_dict, output_path):
    """
    Convert a state dict to safetensors format and save it.
    """
    try:
        # Convert numpy arrays to PyTorch tensors
        converted_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, np.ndarray):
                converted_state_dict[key] = torch.from_numpy(value)
            elif isinstance(value, torch.Tensor):
                converted_state_dict[key] = value
            else:
                raise ValueError(f"Unsupported type for tensor conversion: {type(value)}")

        total_size = sum(tensor.numel() * tensor.element_size() for tensor in converted_state_dict.values())
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Converting to safetensors") as pbar:
            save_file(converted_state_dict, output_path)
            pbar.update(total_size)
        return output_path
    except Exception as e:
        logger.error(f"Failed to convert model to safetensors: {e}")
        return None

def load_model(file_path):
    """
    Load a model from various formats and return its state dict and the path to a safetensors version.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension in ['.pt', '.pth']:
        state_dict = load_pytorch_model(file_path)
    elif file_extension in ['.pb', '.h5']:
        state_dict = load_tensorflow_model(file_path)
    elif file_extension == '.safetensors':
        return load_safetensors_model(file_path), file_path
    else:
        logger.error(f"Unsupported file format: {file_extension}")
        return None, None

    if state_dict is None:
        return None, None

    safetensors_path = os.path.splitext(file_path)[0] + '.safetensors'
    
    if not os.path.exists(safetensors_path):
        user_input = input(f"Do you want to convert the model to safetensors format? (y/n): ").lower()
        if user_input == 'y':
            converted_path = convert_to_safetensors(state_dict, safetensors_path)
        else:
            logger.info("User chose not to convert the model to safetensors format.")
            converted_path = None
    else:
        logger.info(f"Safetensors file already exists at {safetensors_path}")
        converted_path = safetensors_path
    
    return state_dict, converted_path

def is_supported_format(file_path):
    """
    Check if the given file is in a supported format.
    """
    supported_extensions = ['.pt', '.pth', '.pb', '.h5', '.safetensors']
    return os.path.splitext(file_path)[1].lower() in supported_extensions