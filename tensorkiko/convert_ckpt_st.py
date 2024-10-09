import torch
from safetensors.torch import save_file
import sys
import pickle

def convert_ckpt_to_safetensors(ckpt_path, output_path):
    try:
        # Attempt to load the checkpoint normally
        state_dict = torch.load(ckpt_path, map_location="cpu")
    except pickle.UnpicklingError:
        print("UnpicklingError encountered. Attempting to load with 'latin1' encoding...")
        try:
            with open(ckpt_path, 'rb') as f:
                state_dict = torch.load(f, map_location="cpu", encoding='latin1')
        except Exception as e:
            print(f"Failed to load the checkpoint. Error: {e}")
            return
    except Exception as e:
        print(f"Failed to load the checkpoint. Error: {e}")
        return

    # If the state_dict is nested, extract the model weights
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

    # Filter out non-tensor entries
    state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    try:
        # Save as safetensors
        save_file(state_dict, output_path)
        print(f"Conversion completed: {ckpt_path} -> {output_path}")
    except Exception as e:
        print(f"Failed to save the safetensors file. Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_ckpt_path> <output_safetensors_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    convert_ckpt_to_safetensors(input_path, output_path)