import torch
from safetensors.torch import save_file
import os
import pickle
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def convert_ckpt_to_safetensors(ckpt_path, output_path=None):
    logging.info(f"Loading checkpoint from {ckpt_path}")
    try:
        # First, try loading the checkpoint without specifying any encoding
        state_dict = torch.load(ckpt_path, map_location="cpu")
        logging.info("Checkpoint loaded successfully without specifying encoding.")
    except pickle.UnpicklingError:
        logging.error("Unable to load the checkpoint. It might be corrupted or saved in an incompatible format.")
        logging.info("Attempting to load with 'latin1' encoding...")
        try:
            with open(ckpt_path, 'rb') as f:
                state_dict = torch.load(f, map_location="cpu", encoding='latin1')
            logging.info("Checkpoint loaded successfully with 'latin1' encoding.")
        except Exception as e:
            logging.error(f"Failed to load the checkpoint even with 'latin1' encoding. Details: {str(e)}")
            # Attempt to fix the newlines and retry loading
            logging.info("Attempting to fix newline issues...")
            fixed_path = fix_newlines(ckpt_path)
            if fixed_path:
                try:
                    state_dict = torch.load(fixed_path, map_location="cpu")
                    logging.info("Checkpoint loaded successfully after fixing newlines.")
                except Exception as e:
                    logging.error(f"Failed to load the checkpoint after newline fix. Details: {str(e)}")
                    return None
            else:
                return None
    except Exception as e:
        logging.error(f"Failed to load the checkpoint. Details: {str(e)}")
        return None

    # Handle different dictionary structures within the checkpoint
    if isinstance(state_dict, dict):
        logging.info(f"Checkpoint contains keys: {list(state_dict.keys())}")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            logging.info("Extracted 'state_dict' from checkpoint.")
        elif "model" in state_dict:
            state_dict = state_dict["model"]
            logging.info("Extracted 'model' from checkpoint.")
    else:
        logging.error("The loaded checkpoint is not a dictionary.")
        return None

    # Filter out non-tensor entries
    tensor_keys = [k for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
    non_tensor_keys = [k for k in state_dict.keys() if k not in tensor_keys]

    if non_tensor_keys:
        logging.warning(f"Filtering out non-tensor entries: {non_tensor_keys}")

    state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    logging.info(f"Number of tensor entries to save: {len(state_dict)}")

    if output_path is None:
        output_path = os.path.splitext(ckpt_path)[0] + ".safetensors"
        logging.info(f"No output path provided. Using default: {output_path}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logging.error(f"Failed to create output directory '{output_dir}'. Details: {str(e)}")
            return None

    try:
        logging.info(f"Saving safetensors to {output_path}")
        save_file(state_dict, output_path)
        logging.info(f"Conversion completed: {ckpt_path} -> {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to save the safetensors file. Details: {str(e)}")
        return None

def fix_newlines(file_path):
    """Convert CRLF to LF if needed and save to a new file"""
    logging.info(f"Fixing newline characters in {file_path}")
    try:
        fixed_path = file_path.replace('.ckpt', '_unix.ckpt')
        with open(file_path, 'rb') as infile:
            content = infile.read()
        with open(fixed_path, 'wb') as output:
            for line in content.splitlines():
                output.write(line + b'\n')
        logging.info(f"Newlines fixed. Saved to {fixed_path}")
        return fixed_path
    except Exception as e:
        logging.error(f"Failed to fix newlines in {file_path}. Details: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .ckpt files to .safetensors")
    parser.add_argument("input_path", help="Path to the input .ckpt file")
    parser.add_argument("--output_path", help="Path to the output .safetensors file (optional)")
    args = parser.parse_args()

    # Prompt for confirmation
    user_input = input(f"Convert {args.input_path} to safetensors? (y/n): ").strip().lower()
    if user_input != 'y':
        print("Conversion cancelled.")
        exit(0)

    convert_ckpt_to_safetensors(args.input_path, args.output_path)
