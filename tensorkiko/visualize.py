import argparse
import os
import sys
import logging
import webbrowser
import http.server
from http.server import SimpleHTTPRequestHandler
import socketserver
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import torch
from anytree import Node
from safetensors import safe_open
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm  # For progress bars
import re
import threading
from urllib.parse import urlparse, parse_qs
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pkg_resources import resource_filename
from tensorkiko.convert_ckpt_st import convert_ckpt_to_safetensors
from tensorkiko.tensor_processing import process_tensors

@dataclass
class ModelVisualizer:
    file_paths: List[str]
    debug: bool = False
    no_tree: bool = False
    port: int = 8000
    output_dir: str = ''
    include_layers: Optional[str] = None
    exclude_layers: Optional[str] = None
    state_dicts: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    model_trees: Dict[str, Any] = field(default_factory=dict)
    tree_data: Dict[str, Any] = field(default_factory=dict)
    layer_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    model_info: Dict[str, Any] = field(default_factory=lambda: {
        'total_params': 0,
        'memory_usage': 0,
        'precisions': set(),
        'estimated_flops': 0
    })
    tensor_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    anomalies: Dict[str, str] = field(default_factory=dict)
    html_content: str = ''  # Store the generated HTML content

    template_env: Environment = field(init=False)

    def __post_init__(self):
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Use pkg_resources to find the template directory
        template_dir = resource_filename('tensorkiko', 'static/templates')
        
        self.template_env = Environment(
            loader=FileSystemLoader(searchpath=template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )

        # Verify that the template file exists
        template_path = os.path.join(template_dir, 'index.html')
        if not os.path.exists(template_path):
            self.logger.error(f"Template file not found: {template_path}")
            raise FileNotFoundError(f"Template file not found: {template_path}")
        else:
            self.logger.debug(f"Template file found: {template_path}")

    def load_and_process_safetensors(self) -> None:
        for file_path in self.file_paths:
            if not os.path.exists(file_path) or not file_path.lower().endswith('.safetensors'):
                self.logger.error(f"Invalid file: {file_path}")
                continue
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    state_dict = {key: f.get_tensor(key) for key in f.keys()}
                self.logger.debug(f"Loaded {len(state_dict)} tensors from {file_path}")
                self.state_dicts[file_path] = state_dict
            except Exception as e:
                self.logger.error(f"Failed to load the safetensors file {file_path}: {e}")
                continue

            root = Node("Model", type="root", full_name="Model")
            for key, tensor in state_dict.items():
                if self.include_layers and not re.search(self.include_layers, key):
                    continue
                if self.exclude_layers and re.search(self.exclude_layers, key):
                    continue

                parent = root
                parts = key.split('.')
                full_name = ""
                for i, part in enumerate(parts):
                    full_name = f"{full_name}.{part}" if full_name else part
                    child = next((c for c in parent.children if c.name == part), None)
                    if not child:
                        node_type = "layer" if i == len(parts) - 1 else "module"
                        node_attrs = {
                            "type": node_type,
                            "params": tensor.numel() if node_type == "layer" else 0,
                            "shape": str(tensor.shape) if node_type == "layer" else "",
                            "full_name": full_name,
                        }
                        child = Node(part, parent=parent, **node_attrs)
                    parent = child
            self.model_trees[file_path] = root

            # Update model_info
            self.update_model_info(state_dict)

            # Process tensors
            process_tensors(self,state_dict)

    def update_model_info(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model_info['total_params'] += sum(t.numel() for t in state_dict.values())
        self.model_info['memory_usage'] += round(sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024 * 1024), 2)
        self.model_info['precisions'].update(str(t.dtype) for t in state_dict.values())

    def safe_to_numpy(self, tensor):
        if tensor.dtype == torch.bfloat16:
            return tensor.to(torch.float32).cpu().numpy()
        return tensor.cpu().numpy()

    def calculate_tensor_stats(self, tensor_data: np.ndarray) -> Dict[str, Any]:
        stats = {
            'mean': None,
            'std': None,
            'min': float(np.min(tensor_data)),
            'max': float(np.max(tensor_data)),
            'num_zeros': int(np.sum(tensor_data == 0)),
            'num_elements': tensor_data.size,
            'histogram': None
        }

        # Safely calculate mean and std
        with np.errstate(all='ignore'):
            try:
                stats['mean'] = float(np.mean(tensor_data))
                stats['std'] = float(np.std(tensor_data))
            except Exception:
                # If calculation fails, try with float64
                tensor_data_64 = tensor_data.astype(np.float64)
                try:
                    stats['mean'] = float(np.mean(tensor_data_64))
                    stats['std'] = float(np.std(tensor_data_64))
                except Exception:
                    # If still failing, use robust statistics
                    stats['mean'] = float(np.median(tensor_data_64))
                    stats['std'] = float(np.median(np.abs(tensor_data_64 - stats['mean'])))

        # Only compute histogram if tensor is small to avoid performance issues
        if tensor_data.size <= 1e6:
            try:
                hist_counts, bin_edges = np.histogram(tensor_data, bins=50)
                stats['histogram'] = [hist_counts.tolist(), bin_edges.tolist()]
            except Exception as e:
                self.logger.warning(f"Could not compute histogram: {e}")

        return stats

    def detect_anomalies(self, tensor_data: np.ndarray, stats: Dict[str, Any]) -> Optional[str]:
        if np.isnan(tensor_data).any() or np.isinf(tensor_data).any():
            return 'Contains NaN or Inf values'
        elif stats['std'] == 0:
            return 'Zero variance'
        else:
            # Advanced anomaly detection using z-scores
            if stats['std'] != 0:
                z_scores = np.abs((tensor_data - stats['mean']) / stats['std'])
                outliers = np.where(z_scores > 6)[0]  # Threshold of 6 standard deviations
                if len(outliers) > 0:
                    return f'Outliers detected: {len(outliers)} values beyond 6 std dev'
            else:
                return 'Standard deviation is zero; cannot compute z-scores'
        return None

    def tree_to_dict(self, node: Node) -> Dict[str, Any]:
        return {
            'name': node.name,
            'type': getattr(node, 'type', 'unknown'),
            'params': getattr(node, 'params', 0),
            'shape': getattr(node, 'shape', ''),
            'full_name': getattr(node, 'full_name', ''),
            'children': [self.tree_to_dict(child) for child in node.children] if node.children else []
        }

    def generate_html(self, tree_data: Dict[str, Any], model_name: str) -> str:
        def format_shape(s):
            m = re.match(r'torch\.Size\(\[(.*)\]\)', s)
            return ' × '.join(m.group(1).split(', ')) if m else s

        def calculate_totals(node):
            if not node.get('children'):
                node['total_params'] = node.get('params', 0)
                return node['total_params']
            node['total_params'] = sum(calculate_totals(child) for child in node['children'])
            return node['total_params']

        def generate_tree_html(node):
            if not isinstance(node, dict):
                return ""
            name, children = node.get('name', 'Unknown'), node.get('children', [])
            params, total_params, shape = node.get('params', 0), node.get('total_params', 0), format_shape(node.get('shape', ''))
            param_display = f"{params:,}" if params > 0 else f"Total: {total_params:,}"
            caret = '<span class="caret"></span>' if children else ''
            anomaly = self.anomalies.get(node.get('full_name', ''), '')
            anomaly_class = 'anomaly' if anomaly else ''
            html = f'<li><div class="node {anomaly_class}" data-params="{param_display}" data-shape="{shape}" data-name="{name}" data-fullname="{node.get("full_name", "")}">{caret}{name}</div>'
            if children:
                html += f'<ul class="nested">{"".join(generate_tree_html(child) for child in children)}</ul>'
            return html + '</li>'

        calculate_totals(tree_data)
        tree_html = generate_tree_html(tree_data) if not self.no_tree else "<p>Tree visualization is disabled.</p>"
        layer_type_html = ''.join(f'<li>{layer}: {count}</li>' for layer, count in sorted(self.layer_types.items(), key=lambda x: x[1], reverse=True))

        # Prepare tensor stats and anomalies for JavaScript
        tensor_stats_json = json.dumps(self.tensor_stats)
        anomalies_json = json.dumps(self.anomalies)

        # Aggregate precisions
        precisions = ', '.join(self.model_info['precisions'])

        # Prepare model_info for the template
        model_info = {
            'total_params': self.model_info['total_params'],
            'memory_usage': self.model_info['memory_usage'],
            'estimated_flops': self.model_info['estimated_flops']
        }

        # Render the template with context
        template = self.template_env.get_template("index.html")
        html_content = template.render(
            model_name=model_name,
            tree_html=tree_html,
            layer_type_html=layer_type_html,
            tensor_stats_json=tensor_stats_json,
            anomalies_json=anomalies_json,
            model_info=model_info,
            precisions=precisions
        )
        return html_content


    def serve_html(self) -> None:
        class DebugHTTPServer(socketserver.TCPServer):
            def __init__(self, *args, **kwargs):
                self.debug = kwargs.pop('debug', False)
                self.html_content = kwargs.pop('html_content', '')
                self.tensor_stats = kwargs.pop('tensor_stats', {})
                self.anomalies = kwargs.pop('anomalies', {})
                super().__init__(*args, **kwargs)

        with DebugHTTPServer(("", self.port), CustomHandler, debug=self.debug, html_content=self.html_content, tensor_stats=self.tensor_stats, anomalies=self.anomalies) as httpd:
            url = f"http://localhost:{self.port}/"
            self.logger.info(f"Serving visualization at {url}")
            webbrowser.open(url)
            self.logger.info("Press Ctrl+C to stop the server and exit.")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                self.logger.info("\nServer stopped by user.")
            except Exception as e:
                self.logger.error(f"Error starting the server: {e}")
                self.logger.info(f"You can manually open the HTML file: {url}")

            with DebugHTTPServer(("", self.port), CustomHandler, debug=self.debug, html_content=self.html_content, tensor_stats=self.tensor_stats, anomalies=self.anomalies) as httpd:
                url = f"http://localhost:{self.port}/"
                self.logger.info(f"Serving visualization at {url}")
                webbrowser.open(url)
                self.logger.info("Press Ctrl+C to stop the server and exit.")
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    self.logger.info("\nServer stopped by user.")
                except Exception as e:
                    self.logger.error(f"Error starting the server: {e}")
                    self.logger.info(f"You can manually open the HTML file: {url}")

    def generate_visualization(self) -> None:
        try:
            self.logger.info("Loading models...")
            self.load_and_process_safetensors()
            for file_path, model_tree in self.model_trees.items():
                if not self.no_tree:
                    self.tree_data = self.tree_to_dict(model_tree)
                else:
                    self.tree_data = {}
                model_name = os.path.basename(file_path)
                self.html_content = self.generate_html(self.tree_data, model_name)
                # Directly serve the HTML from memory
                server_thread = threading.Thread(target=self.serve_html, daemon=True)
                server_thread.start()
                server_thread.join()
        except Exception as e:
            self.logger.error(f"Error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)

class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(self.server.html_content.encode('utf-8'))
        elif self.path.startswith('/static/'):
            # Serve static files
            static_dir = resource_filename('tensorkiko', 'static')
            file_path = os.path.join(static_dir, self.path[8:])  # Remove '/static/' prefix
            if os.path.exists(file_path) and os.path.isfile(file_path):
                with open(file_path, 'rb') as file:
                    content = file.read()
                self.send_response(200)
                if self.path.endswith('.css'):
                    self.send_header("Content-type", "text/css")
                elif self.path.endswith('.js'):
                    self.send_header("Content-type", "application/javascript")
                else:
                    self.send_header("Content-type", self.guess_type(file_path))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, f"File not found: {self.path}")
        elif self.path == '/api/data':
            # API endpoint to serve data
            data = {
                'tensor_stats': self.server.tensor_stats,
                'anomalies': self.server.anomalies
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            self.send_error(404, f"File not found: {self.path}")

    def log_message(self, format, *args):
        if self.server.debug:
            super().log_message(format, *args)
def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze safetensors models.")
    parser.add_argument("file_paths", nargs='+', help="Paths to the model files (.safetensors or .ckpt)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-tree", action="store_true", help="Disable tree visualization")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port (default: 8000)")
    parser.add_argument("--output-dir", default="", help="Directory to save the output files")
    parser.add_argument("--include-layers", help="Regex to include specific layers")
    parser.add_argument("--exclude-layers", help="Regex to exclude specific layers")
    parser.add_argument("--convert", action="store_true", help="Convert .ckpt files to .safetensors")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger("TensorKiko")

    safetensors_files = []

    for file_path in args.file_paths:
        if file_path.lower().endswith('.ckpt'):
            if args.convert or input(f"Convert {file_path} to safetensors? (y/n): ").lower() == 'y':
                output_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(file_path))[0] + ".safetensors") if args.output_dir else None
                converted_path = convert_ckpt_to_safetensors(file_path, output_path)
                if converted_path:
                    safetensors_files.append(converted_path)
                else:
                    logger.error(f"Failed to convert {file_path}. Skipping this file.")
            else:
                logger.warning(f"Skipping {file_path} as it's not in safetensors format.")
        elif file_path.lower().endswith('.safetensors'):
            safetensors_files.append(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_path}")

    if not safetensors_files:
        logger.error("No valid safetensors files to process.")
        sys.exit(1)

    visualizer = ModelVisualizer(
        file_paths=safetensors_files,
        debug=args.debug,
        no_tree=args.no_tree,
        port=args.port,
        output_dir=args.output_dir,
        include_layers=args.include_layers,
        exclude_layers=args.exclude_layers
    )
    visualizer.generate_visualization()

if __name__ == "__main__":
    main()
