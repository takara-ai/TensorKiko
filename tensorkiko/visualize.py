import argparse
import os
import sys
import logging
import webbrowser
import http.server
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

    def __post_init__(self):
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

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
            self.process_tensors(state_dict)

    def update_model_info(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model_info['total_params'] += sum(t.numel() for t in state_dict.values())
        self.model_info['memory_usage'] += sum(t.numel() * t.element_size() for t in state_dict.values())
        self.model_info['precisions'].update(str(t.dtype) for t in state_dict.values())

    def process_tensors(self, state_dict: Dict[str, torch.Tensor]) -> None:
        def process_tensor(key_tensor):
            key, tensor = key_tensor
            if self.include_layers and not re.search(self.include_layers, key):
                return
            if self.exclude_layers and re.search(self.exclude_layers, key):
                return

            parts = key.split('.')
            layer_type = parts[-2] if parts[-1] in {'weight', 'bias', 'running_mean', 'running_var'} else parts[-1]
            self.layer_types[layer_type] += 1

            # Enhanced FLOPs estimation
            if 'weight' in key:
                if tensor.dim() == 4:
                    n, c, h, w = tensor.shape
                    input_size = 224  # Placeholder; consider making this configurable
                    self.model_info['estimated_flops'] += 2 * n * c * h * w * input_size * input_size
                elif tensor.dim() == 2:
                    m, n = tensor.shape
                    self.model_info['estimated_flops'] += 2 * m * n

            # Calculate tensor statistics
            tensor_data = tensor.cpu().numpy()
            stats = self.calculate_tensor_stats(tensor_data)
            self.tensor_stats[key] = stats

            # Advanced anomaly detection
            anomaly = self.detect_anomalies(tensor_data, stats)
            if anomaly:
                self.anomalies[key] = anomaly

        # Use ThreadPoolExecutor with progress bar
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            key_tensor_pairs = list(state_dict.items())
            futures = {executor.submit(process_tensor, kt): kt for kt in key_tensor_pairs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tensors"):
                try:
                    future.result()
                except Exception as e:
                    kt = futures[future]
                    self.logger.error(f"Error processing tensor {kt[0]}: {e}")
                    self.anomalies[kt[0]] = f"Error during processing: {e}"
                    self.tensor_stats[kt[0]] = {
                        'mean': None,
                        'std': None,
                        'min': None,
                        'max': None,
                        'num_zeros': None,
                        'num_elements': kt[1].numel(),
                        'histogram': None
                    }

        self.model_info['memory_usage'] /= (1024 * 1024)  # Convert to MB
    

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
        with np.errstate(all='raise'):
            try:
                stats['mean'] = float(np.mean(tensor_data))
                stats['std'] = float(np.std(tensor_data))
            except FloatingPointError:
                # If overflow occurs, try with float64
                tensor_data_64 = tensor_data.astype(np.float64)
                try:
                    stats['mean'] = float(np.mean(tensor_data_64))
                    stats['std'] = float(np.std(tensor_data_64))
                except FloatingPointError:
                    # If still overflowing, use robust statistics
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

        # Embed your custom CSS directly into the HTML
        embedded_css = """
        <style>
            :root {
              --spacing: 1.5rem;
              --radius: 10px;
              --primary-color: #3498db;
              --bg-color: #f0f0f0;
              --text-color: #333;
              --border-color: #ddd;
            }
            
            body {
              font-family: Arial, sans-serif;
              margin: 0;
              padding: 0;
              background-color: var(--bg-color);
              color: var(--text-color);
            }
            
            #header {
              position: fixed;
              top: 0;
              left: 0;
              right: 0;
              background-color: rgba(255, 255, 255, 0.95);
              box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
              padding: 10px 20px;
              z-index: 1000;
              display: flex;
              flex-direction: column;
            }
            
            #model-info {
              display: flex;
              flex-wrap: wrap;
              justify-content: space-between;
              align-items: flex-start;
            }
            
            #model-details,
            #layer-types-container {
              flex: 1;
              min-width: 200px;
              margin-right: 20px;
            }
            
            #layer-types {
              font-size: 0.9em;
              max-height: 150px;
              overflow-y: auto;
              list-style-type: none;
              padding: 0;
              margin: 0;
              display: flex;
              flex-wrap: wrap;
            }
            
            #layer-types li {
              margin: 0 10px 5px 0;
              background-color: #e0e0e0;
              padding: 2px 5px;
              border-radius: 3px;
            }
            
            #search-container {
              margin-top: 10px;
              position: relative;
            }
            
            #search {
              width: 100%;
              padding: 5px;
            }
            
            #search-results {
              position: absolute;
              right: 5px;
              top: 50%;
              transform: translateY(-50%);
              font-size: 0.8em;
              color: #666;
            }
            
            #tree {
              padding: 20px;
            }
            
            .tree li {
              display: block;
              position: relative;
              padding-left: calc(2 * var(--spacing) - var(--radius) - 2px);
            }
            
            .tree ul {
              margin-left: calc(var(--radius) - var(--spacing));
              padding-left: 0;
            }
            
            .tree ul li {
              border-left: 2px solid var(--border-color);
            }
            
            .tree ul li:last-child {
              border-color: transparent;
            }
            
            .tree ul li::before {
              content: "";
              display: block;
              position: absolute;
              top: calc(var(--spacing) / -2);
              left: -2px;
              width: calc(var(--spacing) + 2px);
              height: calc(var(--spacing) + 1px);
              border: solid var(--border-color);
              border-width: 0 0 2px 2px;
            }
            
            .tree .node {
              display: inline-block;
              cursor: pointer;
              background-color: #fff;
              border: 2px solid var(--border-color);
              border-radius: var(--radius);
              padding: 0.5rem 1rem;
              margin: 0.5rem 0;
              transition: all 0.3s;
            }
            
            .tree .node:hover {
              background-color: var(--bg-color);
              transform: translateY(-2px);
              box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .tree .node.selected {
              background-color: #e6f3ff;
              border-color: var(--primary-color);
            }
            
            .caret {
              cursor: pointer;
              user-select: none;
              display: inline-block;
              width: 0;
              height: 0;
              margin-right: 6px;
              vertical-align: middle;
              border: 6px solid transparent;
              border-left-color: var(--text-color);
              transition: transform 0.2s;
            }
            
            .caret-down {
              transform: rotate(90deg);
            }
            
            .nested {
              display: none;
            }
            
            .active {
              display: block;
            }
            
            #layer-info {
              position: fixed;
              bottom: 20px;
              right: 20px;
              background-color: #fff;
              border-radius: var(--radius);
              padding: 15px;
              box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
              display: none;
              max-width: 300px;
              overflow: auto;
              max-height: 80vh;
              z-index: 1001;
            }
            
            .highlight {
              background-color: yellow;
            }
            
            .current-highlight {
              background-color: orange;
            }
            
            .anomaly {
              background-color: #ffe6e6;
            }
            
            #histogram-container {
              margin-top: 20px;
            }
            
            .histogram-bar {
              display: inline-block;
              width: 2px;
              background-color: var(--primary-color);
              vertical-align: bottom;
              margin-right: 1px;
            }
            
            #anomaly-info {
              color: red;
              font-weight: bold;
            }
        </style>
        """

        # HTML content with embedded CSS and JavaScript
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Model Visualizer - {model_name}</title>
            {embedded_css}
        </head>
        <body>
            <div id="header">
                <h1>{model_name}</h1>
                <div id="model-info">
                    <div id="model-details">
                        <h3>Model Details</h3>
                        <p>Total Parameters: {self.model_info['total_params']:,}</p>
                        <p>Memory Usage: {self.model_info['memory_usage']:.2f} MB</p>
                        <p>Precisions: {precisions}</p>
                        <p>Estimated FLOPs: {self.model_info['estimated_flops']:,}</p>
                    </div>
                    <div id="layer-types-container">
                        <h3>Layer Types</h3>
                        <ul id="layer-types">{layer_type_html}</ul>
                    </div>
                </div>
                <div id="search-container">
                    <input type="text" id="search" placeholder="Search for layers...">
                    <span id="search-results"></span>
                </div>
            </div>
            <div id="tree"><ul class="tree">{tree_html}</ul></div>
            <div id="layer-info"></div>
            <script>
                document.addEventListener('DOMContentLoaded', () => {{
                    const header = document.getElementById('header');
                    const tree = document.getElementById('tree');
                    const nodes = document.querySelectorAll('.node');
                    const layerInfo = document.getElementById('layer-info');
                    const searchInput = document.getElementById('search');
                    const searchResults = document.getElementById('search-results');
                    const tensorStats = {tensor_stats_json};
                    const anomalies = {anomalies_json};

                    // Function to generate SVG for tensor shapes
                    function generateShapeSVG(shape) {{
                        if (!shape) {{
                            return '<p>No shape information available.</p>';
                        }}
                        // Match shapes in the format "N × C × H × W"
                        const torchMatch = shape.match(/(\\d+)\\s*×\\s*(\\d+)\\s*×\\s*(\\d+)\\s*×\\s*(\\d+)/);
                        if (torchMatch) {{
                            const [_, n, c, h, w] = torchMatch;
                            return `
                                <svg width="100" height="100" viewBox="0 0 100 100">
                                    <rect x="10" y="10" width="80" height="80" fill="#f0f0f0" stroke="#333"/>
                                    <text x="50" y="30" text-anchor="middle" font-size="12">N: ${{n}}</text>
                                    <text x="50" y="50" text-anchor="middle" font-size="12">C: ${{c}}</text>
                                    <text x="50" y="70" text-anchor="middle" font-size="12">H×W: ${{h}}×${{w}}</text>
                                </svg>
                            `;
                        }}

                        // Match general shapes with any number of dimensions
                        const generalMatch = shape.match(/(\\d+(?:\\s*×\\s*\\d+)*)/);
                        if (generalMatch) {{
                            const dims = generalMatch[1].split('×').map(d => d.trim());
                            const dimsText = dims.join(' × ');
                            return `
                                <svg width="200" height="50" viewBox="0 0 200 50">
                                    <rect x="5" y="5" width="190" height="40" fill="#f0f0f0" stroke="#333"/>
                                    <text x="100" y="30" text-anchor="middle" font-size="14">${{dimsText}}</text>
                                </svg>
                            `;
                        }}

                        // If shape does not match expected formats
                        return '<p>No shape information available.</p>';
                    }}

                    // Function to generate histogram HTML
                    function generateHistogram(histogramData) {{
                        if (!histogramData) {{
                            return '<p>No histogram available.</p>';
                        }}
                        const [counts, bins] = histogramData;
                        const maxCount = Math.max(...counts);
                        const histogramHTML = counts.map(count => {{
                            const height = (count / maxCount) * 100;
                            return `<div class="histogram-bar" style="height: ${{height}}px;"></div>`;
                        }}).join('');
                        return `<div style="display: flex; align-items: flex-end; height: 100px;">${{histogramHTML}}</div>`;
                    }}

                    // Set initial top margin for tree
                    tree.style.marginTop = `${{header.offsetHeight + 20}}px`;

                    // Update tree margin on window resize
                    window.addEventListener('resize', () => {{
                        tree.style.marginTop = `${{header.offsetHeight + 20}}px`;
                    }});

                    // Node click event
                    tree.addEventListener('click', function(e) {{
                        if (e.target.classList.contains('caret') || e.target.classList.contains('node')) {{
                            const nodeElement = e.target.classList.contains('caret') ? e.target.parentElement : e.target;
                            const nestedUl = nodeElement.nextElementSibling;

                            // Toggle expand/collapse
                            if (nestedUl) {{
                                nestedUl.classList.toggle('active');
                                const caret = nodeElement.querySelector('.caret');
                                if (caret) caret.classList.toggle('caret-down');
                            }}

                            // Show node info
                            nodes.forEach(n => n.classList.remove('selected'));
                            nodeElement.classList.add('selected');
                            const params = nodeElement.dataset.params;
                            const shape = nodeElement.dataset.shape;
                            const name = nodeElement.dataset.name;
                            const key = nodeElement.dataset.fullname;
                            let infoHTML = `<h3>${{name}}</h3><p>Parameters: ${{params}}</p>`;
                            if (shape) {{
                                // Use SVG for shape representation
                                const shapeSVG = generateShapeSVG(shape);
                                infoHTML += `<div class="shape-svg">${{shapeSVG}}</div>`;
                            }}
                            if (tensorStats[key]) {{
                                const stats = tensorStats[key];
                                if (stats.mean !== null) {{
                                    infoHTML += `<h4>Statistics:</h4><p>Mean: ${{stats.mean.toFixed(4)}}, Std: ${{stats.std.toFixed(4)}}, Min: ${{stats.min.toFixed(4)}}, Max: ${{stats.max.toFixed(4)}}, Zeros: ${{stats.num_zeros}}</p>`;
                                    if (stats.histogram) {{
                                        infoHTML += `<div id="histogram-container">${{generateHistogram(stats.histogram)}}</div>`;
                                    }}
                                }} else {{
                                    infoHTML += `<p>Statistics: Unable to calculate.</p>`;
                                }}
                            }}
                            if (anomalies[key]) {{
                                infoHTML += `<div id="anomaly-info">Anomaly Detected: ${{anomalies[key]}}</div>`;
                            }}
                            layerInfo.innerHTML = infoHTML;
                            layerInfo.style.display = 'block';

                            e.stopPropagation();
                        }}
                    }});

                    document.addEventListener('click', (e) => {{
                        if (!e.target.closest('.node') && !e.target.closest('#layer-info')) {{
                            layerInfo.style.display = 'none';
                            nodes.forEach(n => n.classList.remove('selected'));
                        }}
                    }});

                    // Search functionality
                    searchInput.addEventListener('input', function() {{
                        const searchTerm = this.value.toLowerCase();
                        let matchCount = 0;
                        let firstMatch = null;

                        nodes.forEach(node => {{
                            node.classList.remove('highlight', 'current-highlight');
                            const nodeText = node.textContent.toLowerCase();

                            if (nodeText.includes(searchTerm)) {{
                                matchCount++;
                                node.classList.add('highlight');
                                if (!firstMatch) {{
                                    firstMatch = node;
                                    node.classList.add('current-highlight');
                                }}
                                expandParents(node);
                            }} else {{
                                node.classList.remove('highlight', 'current-highlight');
                            }}
                        }});

                        searchResults.textContent = searchTerm ? `${{matchCount}} result${{matchCount !== 1 ? 's' : ''}} found` : '';

                        if (firstMatch) {{
                            firstMatch.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                        }}
                    }});

                    function expandParents(node) {{
                        let parent = node.parentElement.parentElement;
                        while (parent && parent.id !== 'tree') {{
                            if (parent.tagName.toLowerCase() === 'ul') {{
                                parent.classList.add('active');
                                const parentNode = parent.previousElementSibling;
                                if (parentNode && parentNode.classList.contains('node')) {{
                                    const caret = parentNode.querySelector('.caret');
                                    if (caret) caret.classList.add('caret-down');
                                }}
                            }}
                            parent = parent.parentElement;
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """

        return html_content


    def serve_html(self) -> None:
        handler_class = self.create_handler()

        with socketserver.TCPServer(("", self.port), handler_class) as httpd:
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

    def create_handler(self):
        html_content = self.html_content  # Capture the current HTML content

        class CustomHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(html_content.encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'404 Not Found')

            def log_message(self, format, *args):
                return  # Suppress logging to keep the output clean

        return CustomHandler

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
                webbrowser.open_new_tab(f"http://localhost:{self.port}/")
                server_thread.join()
        except Exception as e:
            self.logger.error(f"Error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze safetensors models.")
    parser.add_argument("file_paths", nargs='+', help="Paths to the safetensors files")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-tree", action="store_true", help="Disable tree visualization")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port (default: 8000)")
    parser.add_argument("--output-dir", default="", help="Directory to save the HTML")
    parser.add_argument("--include-layers", help="Regex to include specific layers")
    parser.add_argument("--exclude-layers", help="Regex to exclude specific layers")
    args = parser.parse_args()

    visualizer = ModelVisualizer(
        file_paths=args.file_paths,
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
