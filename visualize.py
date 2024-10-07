"""
Visualize and analyze the structure of a model stored in a safetensors file.
Utilizes async I/O and multithreading for improved performance.
Tree visualization is optional and toggled by a command-line argument.
"""

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Any

import torch
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from safetensors import safe_open

async def load_safetensors(file_path: str) -> Dict[str, torch.Tensor]:
    """
    Asynchronously load a safetensors file and return its contents as a dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    
    if not file_path.lower().endswith('.safetensors'):
        raise ValueError("The provided file is not a .safetensors file.")

    try:
        loop = asyncio.get_running_loop()
        with safe_open(file_path, framework="pt", device="cpu") as f:
            return await loop.run_in_executor(None, lambda: {key: f.get_tensor(key) for key in f.keys()})
    except Exception as e:
        raise ValueError(f"Failed to load the safetensors file: {str(e)}") from e

def create_tree_structure(state_dict: Dict[str, torch.Tensor]) -> Node:
    """Create a tree structure from the state dictionary."""
    root = Node("Model")
    nodes: Dict[str, Node] = {"": root}

    for key in state_dict:
        parts = key.split('.')
        for i in range(len(parts)):
            parent_path = '.'.join(parts[:i])
            current_path = '.'.join(parts[:i+1])
            
            if current_path not in nodes:
                nodes[current_path] = Node(parts[i], parent=nodes.get(parent_path, root))

    return root

def print_tree(root: Node) -> str:
    """Generate a string representation of the tree structure."""
    def _print_tree(node: Node, prefix: str = "", is_last: bool = True) -> list[str]:
        lines = [f"{prefix}{'`-' if is_last else '|-'} {node.name}"]
        prefix += "   " if is_last else "|  "
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            lines.extend(_print_tree(child, prefix, i == child_count - 1))
        return lines

    return '\n'.join(_print_tree(root))

async def export_tree_to_dot(root: Node, filename: str) -> str:
    """Asynchronously export the tree structure to a dot file for visualization."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, DotExporter(root).to_dotfile, filename)
    return f"Dot file '{filename}' created. You can use this file with Graphviz for visualization."

def analyze_model_structure(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, Dict[str, int]]:
    """Analyze the model structure and count parameters."""
    total_params = sum(tensor.numel() for tensor in state_dict.values())
    layer_types = defaultdict(int)
    
    for key in state_dict:
        parts = key.split('.')
        layer_type = parts[-2] if parts[-1] in {'weight', 'bias'} else parts[-1]
        layer_types[layer_type] += 1

    return total_params, dict(layer_types)

async def main(file_path: str, debug: bool, tree: bool) -> None:
    """Main function to process the safetensors file and generate analysis."""
    start_time = time.time()
    debug_info: Dict[str, Any] = {}

    try:
        print(f"Loading model from: {file_path}")
        state_dict = await load_safetensors(file_path)

        load_time = time.time()
        debug_info['load_time'] = load_time - start_time
        print(f"Model loaded in {debug_info['load_time']:.2f} seconds")

        with ThreadPoolExecutor() as executor:
            model_tree_future = executor.submit(create_tree_structure, state_dict) if tree else None
            total_params, layer_types = await asyncio.get_running_loop().run_in_executor(executor, analyze_model_structure, state_dict)
        
        if tree:
            model_tree = model_tree_future.result()
            tree_time = time.time()
            debug_info['tree_creation_time'] = tree_time - load_time
            print(f"Tree structure created in {debug_info['tree_creation_time']:.2f} seconds")

            tree_output = print_tree(model_tree)
            print("\nModel Structure:")
            print(tree_output)

            print_time = time.time()
            debug_info['tree_print_time'] = print_time - tree_time
            print(f"Tree printed in {debug_info['tree_print_time']:.2f} seconds")

            output_dir = os.path.dirname(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            dot_file = os.path.join(output_dir, f"{base_name}_structure.dot")
            dot_output = await export_tree_to_dot(model_tree, dot_file)
            print(dot_output)

            dot_time = time.time()
            debug_info['dot_export_time'] = dot_time - print_time
            print(f"Dot file exported in {debug_info['dot_export_time']:.2f} seconds")
        else:
            tree_output = None
            dot_output = None

        analysis_output = (
            f"\nModel Analysis:\n"
            f"Total parameters: {total_params:,}\n"
            "\nLayer type distribution:\n"
        )
        analysis_output += ''.join(f"  {layer_type}: {count}\n" for layer_type, count in sorted(layer_types.items(), key=lambda x: x[1], reverse=True))

        print(analysis_output)

        end_time = time.time()
        debug_info['analysis_time'] = end_time - (dot_time if tree else load_time)
        debug_info['total_execution_time'] = end_time - start_time
        print(f"\nAnalysis completed in {debug_info['analysis_time']:.2f} seconds")
        print(f"Total execution time: {debug_info['total_execution_time']:.2f} seconds")

        if debug:
            debug_info['tree_structure'] = tree_output if tree else "Tree visualization not requested"
            debug_info['dot_file_info'] = dot_output if tree else "Tree visualization not requested"
            debug_info['analysis'] = {
                'total_parameters': total_params,
                'layer_types': layer_types
            }
            
            output_dir = os.path.dirname(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            debug_file = os.path.join(output_dir, f"{base_name}_debug.json")
            with open(debug_file, 'w') as f:
                json.dump(debug_info, f, indent=2)
            print(f"\nDebug information saved to {debug_file}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and analyze the structure of a model stored in a safetensors file.")
    parser.add_argument("file_path", help="Path to the safetensors file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode and save detailed information to a JSON file")
    parser.add_argument("--tree", action="store_true", help="Generate and visualize the model tree structure")
    args = parser.parse_args()

    asyncio.run(main(args.file_path, args.debug, args.tree))