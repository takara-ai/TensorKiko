#!/usr/bin/env python3
"""
Model Visualizer: Visualize and analyze the structure of a model stored in a safetensors file.
Utilizes async I/O and multithreading for improved performance.
Tree visualization is optional and can be toggled by a command-line argument.
"""

import argparse
import asyncio
import os
import re
import sys
import logging
import webbrowser
import http.server
import socketserver
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Any

import torch
from anytree import Node
from safetensors import safe_open

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Class to handle model visualization."""

    def __init__(self, file_path: str, debug: bool, no_tree: bool, port: int, output_dir: str):
        self.file_path = file_path
        self.debug = debug
        self.no_tree = no_tree
        self.port = port
        self.output_dir = output_dir
        self.state_dict = {}
        self.model_tree = None
        self.tree_data = {}
        self.total_params = 0
        self.layer_types = {}

    async def load_safetensors(self) -> Dict[str, torch.Tensor]:
        """
        Asynchronously load a safetensors file and return its contents as a dictionary.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File '{self.file_path}' does not exist.")

        if not self.file_path.lower().endswith('.safetensors'):
            raise ValueError("The provided file is not a .safetensors file.")

        try:
            loop = asyncio.get_running_loop()
            with safe_open(self.file_path, framework="pt", device="cpu") as f:
                state_dict = await loop.run_in_executor(
                    None,
                    lambda: {key: f.get_tensor(key) for key in f.keys()}
                )
            logger.info(f"Successfully loaded safetensors file: {self.file_path}")
            return state_dict
        except Exception as e:
            raise ValueError(f"Failed to load the safetensors file: {str(e)}") from e

    def create_tree_structure(self, state_dict: Dict[str, torch.Tensor]) -> Node:
        """
        Create a hierarchical tree structure from the state dictionary.
        """
        root = Node("Model", type="root")
        nodes: Dict[str, Node] = {"": root}

        for key, tensor in state_dict.items():
            parts = key.split('.')
            for i in range(len(parts)):
                parent_path = '.'.join(parts[:i])
                current_path = '.'.join(parts[:i+1])

                if current_path not in nodes:
                    node_type = parts[-2] if i == len(parts) - 1 else "layer"
                    nodes[current_path] = Node(
                        parts[i],
                        parent=nodes.get(parent_path, root),
                        type=node_type,
                        params=tensor.numel() if i == len(parts) - 1 else 0,
                        shape=str(tensor.shape) if i == len(parts) - 1 else ""
                    )

        logger.info("Tree structure created successfully.")
        return root

    def analyze_model_structure(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, Dict[str, int]]:
        """
        Analyze the model structure and count parameters.
        """
        total_params = sum(tensor.numel() for tensor in state_dict.values())
        layer_types = defaultdict(int)

        for key in state_dict:
            parts = key.split('.')
            layer_type = parts[-2] if parts[-1] in {'weight', 'bias'} else parts[-1]
            layer_types[layer_type] += 1

        logger.info(f"Total parameters: {total_params:,}")
        logger.info("Layer types count:")
        for layer, count in layer_types.items():
            logger.info(f"  {layer}: {count}")
        return total_params, dict(layer_types)

    def tree_to_dict(self, node: Node) -> Dict[str, Any]:
        """
        Convert the anytree Node structure to a dictionary.
        """
        return {
            'name': node.name,
            'type': getattr(node, 'type', 'unknown'),
            'params': getattr(node, 'params', 0),
            'shape': getattr(node, 'shape', ''),
            'children': [self.tree_to_dict(child) for child in node.children] if node.children else None
        }

    def generate_html(self, tree_data: Dict[str, Any], total_params: int,
                      layer_types: Dict[str, int], model_name: str) -> str:
        """
        Generate the HTML content for visualization.
        """
        def format_shape(shape_str: str) -> str:
            torch_match = re.match(r'torch\.Size\(\[(.*)\]\)', shape_str)
            if torch_match:
                dims = torch_match.group(1).split(', ')
                return ' × '.join(dims)
            return shape_str

        def calculate_totals(node: Dict[str, Any]) -> Tuple[int, str]:
            if node is None or not isinstance(node, dict):
                return 0, ""

            params = node.get('params', 0)
            shape = node.get('shape', '')
            children = node.get('children', [])

            if children:
                child_params = sum(calculate_totals(child)[0] for child in children)
                if params == 0:
                    params = child_params
                    node['total_params'] = params
                else:
                    node['total_params'] = child_params
            else:
                node['total_params'] = params

            return params, shape

        def generate_tree_html(node: Dict[str, Any], depth=0) -> str:
            if node is None or not isinstance(node, dict):
                return ""

            node_name = node.get('name', 'Unknown')
            children = node.get('children', [])
            params = node.get('params', 0)
            total_params = node.get('total_params', 0)
            shape = node.get('shape', '')

            param_display = f"{params:,}" if params > 0 else f"Total: {total_params:,}"
            formatted_shape = format_shape(shape)

            html = f'<li><div class="node" data-params="{param_display}" data-shape="{formatted_shape}">{node_name}</div>'
            if children:
                html += '<ul>'
                for child in children:
                    html += generate_tree_html(child, depth + 1)
                html += '</ul>'
            html += '</li>'
            return html

        calculate_totals(tree_data)
        tree_html = generate_tree_html(tree_data)

        layer_type_html = ''.join(
            f'<li>{layer}: {count}</li>' for layer, count in sorted(layer_types.items(), key=lambda x: x[1], reverse=True)
        )

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Model Visualizer - {model_name}</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f0f0f0;
                    color: #333;
                }}
                #header {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    background-color: #fff;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 10px 20px;
                    z-index: 1000;
                }}
                #model-info {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                #layer-types {{
                    font-size: 0.8em;
                    max-height: 100px;
                    overflow-y: auto;
                    list-style-type: none;
                    padding: 0;
                    margin: 0;
                }}
                #tree {{
                    margin-top: 150px;
                    padding: 20px;
                }}
                .tree {{
                    --spacing: 1.5rem;
                    --radius: 10px;
                }}
                .tree li {{
                    display: block;
                    position: relative;
                    padding-left: calc(2 * var(--spacing) - var(--radius) - 2px);
                }}
                .tree ul {{
                    margin-left: calc(var(--radius) - var(--spacing));
                    padding-left: 0;
                }}
                .tree ul li {{
                    border-left: 2px solid #ddd;
                }}
                .tree ul li:last-child {{
                    border-color: transparent;
                }}
                .tree ul li::before {{
                    content: '';
                    display: block;
                    position: absolute;
                    top: calc(var(--spacing) / -2);
                    left: -2px;
                    width: calc(var(--spacing) + 2px);
                    height: calc(var(--spacing) + 1px);
                    border: solid #ddd;
                    border-width: 0 0 2px 2px;
                }}
                .tree .node {{
                    display: inline-block;
                    cursor: pointer;
                    background-color: #fff;
                    border: 2px solid #ddd;
                    border-radius: var(--radius);
                    padding: 0.5rem 1rem;
                    margin: 0.5rem 0;
                    transition: all 0.3s;
                }}
                .tree .node:hover {{
                    background-color: #f0f0f0;
                    transform: translateY(-2px);
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                #layer-info {{
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background-color: #fff;
                    border-radius: 10px;
                    padding: 15px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    display: none;
                    max-width: 300px;
                }}
            </style>
        </head>
        <body>
            <div id="header">
                <div id="model-info">
                    <h1>{model_name}</h1>
                    <div>
                        <p>Total Parameters: {total_params:,}</p>
                        <ul id="layer-types">
                            {layer_type_html}
                        </ul>
                    </div>
                </div>
            </div>
            {"<div id='tree'><ul class='tree'>" + tree_html + "</ul></div>" if not self.no_tree else ""}
            <div id="layer-info"></div>
            <script>
                document.addEventListener('DOMContentLoaded', () => {{
                    const nodes = document.querySelectorAll('.node');
                    const layerInfo = document.getElementById('layer-info');

                    nodes.forEach(node => {{
                        node.addEventListener('click', function(e) {{
                            e.stopPropagation();
                            const params = this.dataset.params;
                            const shape = this.dataset.shape;
                            let infoHTML = `<h3>${{this.textContent}}</h3>`;
                            infoHTML += `<p>Parameters: ${{params}}</p>`;
                            if (shape) {{
                                infoHTML += `<p>Shape: ${{shape}}</p>`;
                                infoHTML += `<div id="shape-svg">${{generateShapeSVG(shape)}}</div>`;
                            }}
                            layerInfo.innerHTML = infoHTML;
                            layerInfo.style.display = 'block';
                        }});
                    }});

                    document.addEventListener('click', (e) => {{
                        if (!e.target.closest('.node') && !e.target.closest('#layer-info')) {{
                            layerInfo.style.display = 'none';
                        }}
                    }});

                    function generateShapeSVG(shape) {{
                        const torchMatch = shape.match(/(\d+)\s*×\s*(\d+)\s*×\s*(\d+)\s*×\s*(\d+)/);
                        if (torchMatch) {{
                            const [_, n, c, h, w] = torchMatch;
                            return `
                                <svg width="100" height="100" viewBox="0 0 100 100">
                                    <rect x="10" y="10" width="80" height="80" fill="#f0f0f0" stroke="#333"/>
                                    <text x="50" y="50" text-anchor="middle" dominant-baseline="middle" font-size="14">${{n}}</text>
                                    <rect x="20" y="20" width="60" height="60" fill="#e0e0e0" stroke="#333"/>
                                    <text x="50" y="50" text-anchor="middle" dominant-baseline="middle" font-size="12">${{c}}</text>
                                    <rect x="30" y="30" width="40" height="40" fill="#d0d0d0" stroke="#333"/>
                                    <text x="50" y="50" text-anchor="middle" dominant-baseline="middle" font-size="10">${{h}}×${{w}}</text>
                                </svg>
                            `;
                        }}
                        const generalMatch = shape.match(/(\d+(?:\s*×\s*\d+)*)/);
                        if (generalMatch) {{
                            const dims = generalMatch[1].split('×').map(d => d.trim());
                            return `
                                <svg width="100" height="50" viewBox="0 0 100 50">
                                    <rect x="5" y="5" width="90" height="40" fill="#f0f0f0" stroke="#333"/>
                                    <text x="50" y="25" text-anchor="middle" dominant-baseline="middle" font-size="14">${{dims.join(' × ')}}</text>
                                </svg>
                            `;
                        }}
                        return '';
                    }}
                }});
            </script>
        </body>
        </html>
        """
        return html_content

    def save_html(self, html_content: str, model_name: str) -> str:
        """
        Save the generated HTML content to a file.
        """
        output_dir = self.output_dir or os.path.dirname(os.path.abspath(self.file_path))
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        html_file = os.path.join(output_dir, f"{base_name}_visualization.html")

        try:
            with open(html_file, 'w') as f:
                f.write(html_content)
            logger.info(f"HTML visualization generated: {html_file}")
            return html_file
        except Exception as e:
            raise IOError(f"Failed to write HTML file: {str(e)}") from e

    def serve_html(self, html_file: str) -> None:
        """
        Serve the HTML file using a simple HTTP server and open it in the default web browser.
        """
        try:
            os.chdir(self.output_dir or os.path.dirname(os.path.abspath(self.file_path)))
            handler = http.server.SimpleHTTPRequestHandler
            with socketserver.TCPServer(("localhost", self.port), handler) as httpd:
                port = httpd.server_address[1]
                url = f"http://localhost:{port}/{os.path.basename(html_file)}"
                logger.info(f"Serving visualization at {url}")
                webbrowser.open(url)
                logger.info("Press Ctrl+C to stop the server and exit.")
                httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("\nServer stopped by user.")
        except Exception as server_error:
            logger.error(f"Error starting the server: {str(server_error)}")
            logger.info(f"You can manually open the HTML file: {html_file}")

    async def generate_visualization(self) -> None:
        """
        Main method to generate the visualization.
        """
        try:
            logger.info(f"Loading model from: {self.file_path}")
            self.state_dict = await self.load_safetensors()

            with ThreadPoolExecutor() as executor:
                loop = asyncio.get_running_loop()
                self.model_tree = await loop.run_in_executor(
                    executor,
                    self.create_tree_structure,
                    self.state_dict
                )
                self.total_params, self.layer_types = await loop.run_in_executor(
                    executor,
                    self.analyze_model_structure,
                    self.state_dict
                )

            self.tree_data = self.tree_to_dict(self.model_tree)
            model_name = os.path.splitext(os.path.basename(self.file_path))[0]

            html_content = self.generate_html(
                self.tree_data,
                self.total_params,
                self.layer_types,
                model_name
            )

            html_file = self.save_html(html_content, model_name)
            self.serve_html(html_file)

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            if self.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Visualize and analyze the structure of a model stored in a safetensors file."
    )
    parser.add_argument("file_path", help="Path to the safetensors file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed error information")
    parser.add_argument("--no-tree", action="store_true", help="Disable tree visualization")
    parser.add_argument("--port", type=int, default=8000, help="Port number for the local HTTP server (default: 8000)")
    parser.add_argument("--output-dir", type=str, default="", help="Directory to save the HTML visualization")
    return parser.parse_args()


def main():
    """
    Entry point of the script.
    """
    args = parse_arguments()
    visualizer = ModelVisualizer(
        file_path=args.file_path,
        debug=args.debug,
        no_tree=args.no_tree,
        port=args.port,
        output_dir=args.output_dir
    )
    asyncio.run(visualizer.generate_visualization())


if __name__ == "__main__":
    main()
