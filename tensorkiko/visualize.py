from .modules.ascii_logo import display_logo
display_logo()
import argparse, os, sys, logging, webbrowser, http.server, socketserver, threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import torch
from anytree import Node
from safetensors import safe_open
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm
import re
from urllib.parse import urlparse, parse_qs
from jinja2 import Environment, FileSystemLoader, select_autoescape
import importlib.resources as pkg_resources
from importlib.resources import as_file, files
from .modules.model_loader import load_model, is_supported_format
from .modules.tensor_processing import process_tensors, get_param_size

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
    model_info: Dict[str, Any] = field(default_factory=lambda: {'total_params': 0, 'memory_usage': 0, 'precisions': set(), 'estimated_flops': 0})
    tensor_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    anomalies: Dict[str, str] = field(default_factory=dict)
    html_content: str = ''
    template_env: Environment = field(init=False)

    def __post_init__(self):
        logging.basicConfig(level=logging.DEBUG if self.debug else logging.INFO, format='[%(levelname)s] %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)
        # Use importlib.resources to get the template directory
        template_dir = pkg_resources.files('tensorkiko') / 'static' / 'templates'
        self.template_env = Environment(loader=FileSystemLoader(searchpath=str(template_dir)), autoescape=select_autoescape(['html', 'xml']))
        
        template_path = template_dir / 'index.html'
        if not template_path.is_file():
            raise FileNotFoundError(f"Template file not found: {template_path}")

    def load_and_process_safetensors(self):
        for file_path in self.file_paths:
            if not os.path.exists(file_path) or not file_path.lower().endswith('.safetensors'):
                self.logger.error(f"Invalid file: {file_path}")
                continue
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    state_dict = {key: f.get_tensor(key) for key in f.keys()}
                self.state_dicts[file_path] = state_dict
                root = Node("Model", type="root", full_name="Model")

                # Use the process_tensors function from the imported script
                process_tensors(self, state_dict)

                for key, tensor in state_dict.items():
                    if (self.include_layers and not re.search(self.include_layers, key)) or (self.exclude_layers and re.search(self.exclude_layers, key)):
                        continue
                    self._build_tree(root, key, tensor)

                self.model_trees[file_path] = root
                self.update_model_info(state_dict)

            except Exception as e:
                self.logger.error(f"Failed to load the safetensors file {file_path}: {e}")

    def load_and_process_models(self):
        for file_path in self.file_paths:
            state_dict, safetensors_path = load_model(file_path)
            if state_dict is None:
                self.logger.error(f"Failed to load model: {file_path}")
                continue

            self.state_dicts[safetensors_path] = state_dict
            root = Node("Model", type="root", full_name="Model")

            process_tensors(self, state_dict)

            for key, tensor in state_dict.items():
                if (self.include_layers and not re.search(self.include_layers, key)) or (self.exclude_layers and re.search(self.exclude_layers, key)):
                    continue
                self._build_tree(root, key, tensor)

            self.model_trees[safetensors_path] = root
            self.update_model_info(state_dict)

    def update_model_info(self, state_dict):
        self.model_info['total_params'] = sum(get_param_size(t) for t in state_dict.values())
        self.model_info['memory_usage'] = round(self.model_info['total_params'] / (1024 * 1024), 2)
        self.model_info['precisions'].update(str(t.dtype) for t in state_dict.values())

    def _build_tree(self, root, key, tensor):
        parent = root
        parts = key.split('.')
        full_name = ""
        for i, part in enumerate(parts):
            full_name = f"{full_name}.{part}" if full_name else part
            child = next((c for c in parent.children if c.name == part), None)
            if not child:
                node_type = "layer" if i == len(parts) - 1 else "module"
                node_attrs = {"type": node_type, "params": tensor.numel() if node_type == "layer" else 0,
                              "shape": str(tensor.shape) if node_type == "layer" else "", "full_name": full_name}
                child = Node(part, parent=parent, **node_attrs)
            parent = child

    def safe_to_numpy(self, tensor):
        return tensor.to(torch.float32).cpu().numpy() if tensor.dtype == torch.bfloat16 else tensor.cpu().numpy()

    def tree_to_dict(self, node):
        return {'name': node.name, 'type': getattr(node, 'type', 'unknown'), 'params': getattr(node, 'params', 0),
                'shape': getattr(node, 'shape', ''), 'full_name': getattr(node, 'full_name', ''),
                'children': [self.tree_to_dict(child) for child in node.children] if node.children else []}

    def generate_html(self, tree_data, model_name):
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
        tensor_stats_json, anomalies_json = json.dumps(self.tensor_stats), json.dumps(self.anomalies)
        precisions = ', '.join(self.model_info['precisions'])
        model_info = {k: v for k, v in self.model_info.items() if k in ['total_params', 'memory_usage', 'estimated_flops']}
        template = self.template_env.get_template("index.html")
        return template.render(model_name=model_name, tree_html=tree_html, layer_type_html=layer_type_html,
                               tensor_stats_json=tensor_stats_json, anomalies_json=anomalies_json,
                               model_info=model_info, precisions=precisions)

    def serve_html(self):
        class DebugHTTPServer(socketserver.TCPServer):
            def __init__(self, *args, **kwargs):
                self.debug = kwargs.pop('debug', False)
                self.html_content = kwargs.pop('html_content', '')
                self.tensor_stats = kwargs.pop('tensor_stats', {})
                self.anomalies = kwargs.pop('anomalies', {})
                super().__init__(*args, **kwargs)

        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path in ['/', '/index.html']:
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(self.server.html_content.encode('utf-8'))
                elif self.path.startswith('/static/'):
                    # Use importlib.resources to access static files
                    static_file = files('tensorkiko') / 'static' / self.path[8:]
                    if static_file.is_file():
                        with as_file(static_file) as file_path:
                            with open(file_path, 'rb') as file:
                                content = file.read()
                        self.send_response(200)
                        self.send_header("Content-type", self.guess_type(str(static_file)))
                        self.end_headers()
                        self.wfile.write(content)
                    else:
                        self.send_error(404, f"File not found: {self.path}")
                elif self.path == '/api/data':
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({'tensor_stats': self.server.tensor_stats, 'anomalies': self.server.anomalies}).encode('utf-8'))
                else:
                    self.send_error(404, f"File not found: {self.path}")

            def log_message(self, format, *args):
                if self.server.debug:
                    super().log_message(format, *args)

        with DebugHTTPServer(("", self.port), CustomHandler, debug=self.debug, html_content=self.html_content,
                             tensor_stats=self.tensor_stats, anomalies=self.anomalies) as httpd:
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

    def generate_visualization(self):
        try:
            self.logger.info("Loading models...")
            self.load_and_process_models()
            for file_path, model_tree in self.model_trees.items():
                self.tree_data = self.tree_to_dict(model_tree) if not self.no_tree else {}
                model_name = os.path.basename(file_path)
                self.html_content = self.generate_html(self.tree_data, model_name)
                
                # Start the server in a separate thread
                server_thread = threading.Thread(target=self.serve_html, daemon=True)
                server_thread.start()
                
                # Wait for the server thread to finish (i.e., when the user stops it)
                server_thread.join()
                
                # If there are more models, ask the user if they want to continue
                if file_path != list(self.model_trees.keys())[-1]:
                    if input("Press Enter to visualize the next model, or type 'q' to quit: ").lower() == 'q':
                        break
        except Exception as e:
            self.logger.error(f"Error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze models in various formats.")
    parser.add_argument("file_paths", nargs='+', help="Paths to the model files (.safetensors, .pt, .pth, .pb, .h5)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-tree", action="store_true", help="Disable tree visualization")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port (default: 8000)")
    parser.add_argument("--output-dir", default="", help="Directory to save the output files")
    parser.add_argument("--include-layers", help="Regex to include specific layers")
    parser.add_argument("--exclude-layers", help="Regex to exclude specific layers")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger("TensorKiko")

    valid_files = []
    for file_path in args.file_paths:
        if is_supported_format(file_path):
            valid_files.append(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_path}")

    if not valid_files:
        logger.error("No valid model files to process.")
        sys.exit(1)

    visualizer = ModelVisualizer(
        file_paths=valid_files,
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